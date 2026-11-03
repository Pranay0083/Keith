package worker

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/store"
)

const (
	gitTokenPath  = github.GitTokenPath
	gitTokenDir   = "/run/keith"
	cloneTimeout  = 15 * time.Minute
	gitCmdTimeout = 2 * time.Minute
)

// loadSecrets decrypts the org's and the session repos' secrets into the
// sandbox environment and the redactor. A repo secret overrides an org
// secret of the same name.
func (s *session) loadSecrets(ctx context.Context) error {
	repoIDs := make([]string, 0, len(s.sess.Repos))
	for _, r := range s.sess.Repos {
		repoIDs = append(repoIDs, r.RepoID)
	}
	cts, err := s.w.d.Store.LoadSecretCiphertexts(ctx, s.sess.OrgID, repoIDs...)
	if errors.Is(err, store.ErrNotFound) {
		cts, err = nil, nil
	}
	if err != nil {
		return err
	}
	if len(cts) > 0 && s.w.d.Keyring == nil {
		return errors.New("the worker has no SECRETS_KEY")
	}
	env := map[string]string{}
	var values []string
	for _, ct := range cts {
		v, err := s.w.d.Keyring.Open(ct.Ciphertext, ct.AAD())
		if err != nil {
			return fmt.Errorf("secret %s: %w", ct.Name, err)
		}
		env[ct.Name] = string(v)
		values = append(values, string(v))
	}
	s.env = env
	s.em.addSecrets(values...)
	return nil
}

// issueRef is the issue a session was started from.
type issueRef struct {
	URL    string
	Repo   string
	Number int
}

// issueURL returns sessions.trigger's issue_url, if any.
func issueURL(trigger json.RawMessage) string {
	if len(trigger) == 0 {
		return ""
	}
	var t struct {
		IssueURL string `json:"issue_url"`
	}
	if json.Unmarshal(trigger, &t) != nil {
		return ""
	}
	return strings.TrimSpace(t.IssueURL)
}

// ensureEnv gets the session a running sandbox with the repos checked
// out, fresh tokens and a shell manager: resuming the parked sandbox, or
// provisioning (and setting up) a new one.
func (s *session) ensureEnv(ctx context.Context) error {
	if len(s.sess.Repos) == 0 {
		return failf(ReasonProvisionFailed, "The session has no repositories.")
	}
	inst := s.sess.Repos[0].InstallationID
	var names []string
	for _, r := range s.sess.Repos {
		if r.InstallationID != inst {
			return failf(ReasonMultiInstallation,
				"This session's repositories belong to more than one GitHub App installation. Keith supports one installation per session for now: start one session per GitHub account or organization.")
		}
		names = append(names, r.FullName)
	}
	if err := s.loadRepos(ctx); err != nil {
		return err
	}

	// Credentials. The worker's scope adds the issue's repo when it is in
	// the same account (the API normally requires it to be tagged).
	hostRepos := slices.Clone(names)
	if url := issueURL(s.sess.Trigger); url != "" {
		if repo, n, err := github.ParseIssueURL(url); err == nil {
			s.issue = &issueRef{URL: url, Repo: repo, Number: n}
			owner, _, _ := strings.Cut(repo, "/")
			sameOwner := strings.EqualFold(owner, strings.Split(names[0], "/")[0])
			if sameOwner && !slices.ContainsFunc(hostRepos, func(r string) bool { return strings.EqualFold(r, repo) }) {
				hostRepos = append(hostRepos, repo)
			}
		}
	}
	var err error
	if s.sbTokens, err = s.w.d.GitHub.SandboxTokens(inst, names); err != nil {
		return failf(ReasonProvisionFailed, "Could not set up the sandbox's GitHub token: %v", err)
	}
	if s.host, s.hostTokens, err = s.w.d.GitHub.WorkerHost(inst, hostRepos); err != nil {
		return failf(ReasonProvisionFailed, "Could not set up the worker's GitHub token: %v", err)
	}
	s.tokenFile = &tokenFile{}
	s.sbTokens.OnRefresh(func(t github.Token) {
		s.em.addSecrets(t.Value)
		s.tokenFile.offer(t.Value)
	})

	// Sandbox: resume the parked one, or provision.
	resumed := false
	if s.sess.SandboxID != "" && s.sess.SandboxProvider == s.w.d.Sandboxes.Name() {
		sb, err := s.w.d.Sandboxes.Get(ctx, s.sess.SandboxID)
		if err == nil {
			err = sb.Resume(ctx)
		}
		switch {
		case err == nil:
			s.sb, resumed = sb, true
		case errors.Is(err, sandbox.ErrNotFound):
			s.log.Info("the session's sandbox is gone; provisioning a new one", "sandbox_id", s.sess.SandboxID)
		default:
			return fmt.Errorf("worker: resume sandbox %s: %w", s.sess.SandboxID, err)
		}
	}
	if !resumed {
		return s.provision(ctx)
	}
	s.attachShells()
	if err := s.startTokens(ctx); err != nil {
		return err
	}
	if _, err := s.shells.Reattach(ctx); err != nil {
		s.log.Warn("reattach shells", "err", err)
	}
	for _, r := range s.repos {
		if r.DefaultBranch == "" {
			out, err := s.execOK(ctx, r.Path, "git symbolic-ref --short refs/remotes/origin/HEAD", gitCmdTimeout)
			if err != nil {
				return s.provisionErr(ctx, ReasonProvisionFailed, "Could not find the default branch of "+r.FullName, err)
			}
			r.DefaultBranch = strings.TrimPrefix(strings.TrimSpace(out), "origin/")
		}
	}
	s.envReady = true
	return s.prepareRun(ctx)
}

// loadRepos fills s.repos with the session's repos and their profiles.
func (s *session) loadRepos(ctx context.Context) error {
	s.repos = s.repos[:0]
	for _, r := range s.sess.Repos {
		prof, err := s.w.d.Store.GetRepoProfile(ctx, r.RepoID)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			return err
		}
		_, name, _ := strings.Cut(r.FullName, "/")
		rs := &repoState{SessionRepo: r, Name: name, Path: path.Join(sandbox.WorkspaceDir, name), Profile: prof}
		if rs.Branch == "" {
			rs.Branch = store.SessionBranch(s.id)
		}
		s.repos = append(s.repos, rs)
	}
	return nil
}

// provision creates a sandbox, installs the credential helper, clones
// the repos and runs setup.
func (s *session) provision(ctx context.Context) error {
	if err := s.setStatus(ctx, events.StatusProvisioning, ""); err != nil {
		return err
	}
	spec := sandbox.Spec{
		Template:        s.w.o.SandboxTemplate,
		Env:             s.env,
		Labels:          s.labels(),
		EgressAllowlist: s.w.o.EgressAllowlist,
	}
	// A single-repo session starts from the repo's snapshot when there is
	// one for this provider.
	var snap store.Snapshot
	if len(s.repos) == 1 {
		if p := s.repos[0].Profile; p.Snapshot.ID != "" && p.Snapshot.Provider == s.w.d.Sandboxes.Name() {
			snap = p.Snapshot
			spec.SnapshotID = snap.ID
		}
	}
	sb, err := s.w.d.Sandboxes.Create(ctx, spec)
	if err != nil && spec.SnapshotID != "" && errors.Is(err, sandbox.ErrNotFound) {
		s.log.Warn("the repo's snapshot is gone; starting from the base image", "snapshot", snap.ID)
		_ = s.w.d.Store.SetRepoSnapshot(ctx, s.repos[0].RepoID, store.Snapshot{})
		s.repos[0].Profile.Snapshot = store.Snapshot{}
		snap, spec.SnapshotID = store.Snapshot{}, ""
		sb, err = s.w.d.Sandboxes.Create(ctx, spec)
	}
	if err != nil {
		if ctx.Err() != nil {
			return errInterrupted
		}
		return failf(ReasonProvisionFailed, "Could not create the sandbox: %v", err)
	}
	if old := s.sess.SandboxID; old != "" && old != sb.ID() {
		if oldSB, err := s.w.d.Sandboxes.Get(ctx, old); err == nil {
			_ = oldSB.Delete(ctx)
		}
	}
	s.sb = sb
	if err := s.w.d.Store.SetSessionSandbox(ctx, s.id, sb.ID(), s.w.d.Sandboxes.Name()); err != nil {
		_ = sb.Delete(context.WithoutCancel(ctx))
		s.sb = nil
		return err
	}
	s.sess.SandboxID, s.sess.SandboxProvider = sb.ID(), s.w.d.Sandboxes.Name()

	if err := s.installCredentialHelper(ctx); err != nil {
		return s.provisionErr(ctx, ReasonProvisionFailed, "Could not install the git credential helper", err)
	}
	s.attachShells()
	if err := s.startTokens(ctx); err != nil {
		return err
	}
	for _, r := range s.repos {
		if err := s.clone(ctx, r); err != nil {
			return s.provisionErr(ctx, ReasonCloneFailed, "Could not clone "+r.FullName, err)
		}
	}
	if err := s.setup(ctx, snap); err != nil {
		return err
	}
	s.envReady = true
	return s.prepareRun(ctx)
}

func (s *session) provisionErr(ctx context.Context, reason, what string, err error) error {
	if ctx.Err() != nil || !s.lease.Valid() {
		return errInterrupted
	}
	var fe *failure
	if errors.As(err, &fe) {
		return err
	}
	return failf(reason, "%s: %s", what, s.em.redact(err.Error()))
}

func (s *session) labels() map[string]string {
	l := map[string]string{"keith.session": s.id, "keith.org": s.sess.OrgID}
	for k, v := range s.w.o.SandboxLabels {
		l[k] = v
	}
	return l
}

// user returns the sandbox user for ExecRequest.User.
func (s *session) user() string { return s.w.o.SandboxUser }

// exec runs a command as the sandbox user.
func (s *session) exec(ctx context.Context, dir, cmd string, timeout time.Duration) (*sandbox.ExecResult, error) {
	return s.sb.Exec(ctx, sandbox.ExecRequest{
		Cmd: cmd, Dir: dir, Timeout: timeout, User: s.user(),
		Env: map[string]string{"GIT_TERMINAL_PROMPT": "0"},
	}, nil)
}

// execOK runs cmd and turns a non-zero exit into an error carrying the
// output tail.
func (s *session) execOK(ctx context.Context, dir, cmd string, timeout time.Duration) (string, error) {
	res, err := s.exec(ctx, dir, cmd, timeout)
	if err != nil {
		return "", err
	}
	if res.ExitCode != 0 {
		return res.Stdout, fmt.Errorf("exit %d: %s", res.ExitCode, tail(strings.TrimSpace(res.Stderr+"\n"+res.Stdout), 2000))
	}
	return res.Stdout, nil
}

// installCredentialHelper installs the git credential helper as root,
// creates /run/keith for the sandbox user and points git at the helper
// for the clone remote only (github.CredentialHelperScript).
func (s *session) installCredentialHelper(ctx context.Context) error {
	ids, err := s.execOK(ctx, "/", "id -u; id -g", time.Minute)
	if err != nil {
		return fmt.Errorf("id: %w", err)
	}
	f := strings.Fields(ids)
	if len(f) != 2 {
		return fmt.Errorf("id: unexpected output %q", ids)
	}
	uid, gid := f[0], f[1]
	if _, err := strconv.Atoi(uid); err != nil {
		return fmt.Errorf("id: unexpected uid %q", uid)
	}
	if _, err := strconv.Atoi(gid); err != nil {
		return fmt.Errorf("id: unexpected gid %q", gid)
	}
	script := base64.StdEncoding.EncodeToString([]byte(s.w.remote.CredentialHelperScript()))
	root := fmt.Sprintf(`set -e
printf %%s %s | base64 -d > %s
chown 0:0 %s
chmod 0755 %s
install -d -m 0700 -o %s -g %s %s`,
		shellQuote(script), github.CredentialHelperPath, github.CredentialHelperPath, github.CredentialHelperPath,
		uid, gid, gitTokenDir)
	res, err := s.sb.Exec(ctx, sandbox.ExecRequest{Cmd: root, Dir: "/", User: "root", Timeout: time.Minute}, nil)
	if err != nil {
		return err
	}
	if res.ExitCode != 0 {
		return fmt.Errorf("install as root: exit %d: %s", res.ExitCode, tail(res.Stderr, 1000))
	}
	key := "credential." + s.w.remote.CredentialKey() + ".helper"
	cfg := fmt.Sprintf(`git config --global --unset-all %[1]s >/dev/null 2>&1 || true
git config --global --add %[1]s '' && git config --global --add %[1]s %[2]s`, shellQuote(key), github.CredentialHelperPath)
	if _, err := s.execOK(ctx, "/", cfg, time.Minute); err != nil {
		return fmt.Errorf("git config: %w", err)
	}
	return nil
}

// tokenFile serializes writes of the sandbox token file; the latest
// offered token wins.
type tokenFile struct {
	mu      sync.Mutex
	latest  string
	written string
	signal  chan struct{}
}

func (t *tokenFile) offer(v string) {
	t.mu.Lock()
	t.latest = v
	sig := t.signal
	t.mu.Unlock()
	if sig != nil {
		select {
		case sig <- struct{}{}:
		default:
		}
	}
}

// startTokens mints the first sandbox token, writes it, and keeps it
// fresh: the source's Run refreshes it and a writer rewrites the file.
func (s *session) startTokens(ctx context.Context) error {
	tok, err := s.sbTokens.Token(ctx)
	if err != nil {
		if ctx.Err() != nil {
			return errInterrupted
		}
		return failf(ReasonProvisionFailed, "Could not mint the sandbox's GitHub token: %v", err)
	}
	s.em.addSecrets(tok.Value)
	if err := s.writeToken(ctx, tok.Value); err != nil {
		return s.provisionErr(ctx, ReasonProvisionFailed, "Could not write the sandbox's GitHub token", err)
	}
	tf := s.tokenFile
	tf.mu.Lock()
	tf.written = tok.Value
	tf.signal = make(chan struct{}, 1)
	tf.mu.Unlock()

	tctx, cancel := context.WithCancel(ctx)
	s.tokenStop = cancel
	s.tokenDone = make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_ = s.sbTokens.Run(tctx)
	}()
	go func() {
		defer wg.Done()
		for {
			select {
			case <-tctx.Done():
				return
			case <-tf.signal:
			}
			tf.mu.Lock()
			v, done := tf.latest, tf.written
			tf.mu.Unlock()
			if v == "" || v == done {
				continue
			}
			if err := s.writeToken(tctx, v); err != nil {
				if tctx.Err() == nil {
					s.log.Warn("rotate the sandbox token", "err", err)
				}
				continue
			}
			tf.mu.Lock()
			tf.written = v
			tf.mu.Unlock()
			s.log.Info("rotated the sandbox token")
		}
	}()
	go func() {
		wg.Wait()
		close(s.tokenDone)
	}()
	return nil
}

// writeToken replaces the token file atomically: write a temporary file,
// then rename it, so git never reads half a token.
func (s *session) writeToken(ctx context.Context, value string) error {
	tmp := gitTokenPath + ".tmp"
	if err := s.sb.WriteFile(ctx, tmp, []byte(value+"\n"), 0o600); err != nil {
		return err
	}
	_, err := s.execOK(ctx, "/", "mv -f "+tmp+" "+gitTokenPath, time.Minute)
	return err
}

// attachShells builds the shell manager, streaming output as
// tool_call.output for the agent's current call on that shell (or the
// worker's own synthetic call).
func (s *session) attachShells() {
	s.shells = s.w.d.NewShells(s.sb, s.user(), func(shell string, chunk []byte) {
		callID := ""
		if r := s.agentRun.Load(); r != nil {
			callID = r.CurrentCall(shell)
		}
		if callID == "" {
			s.synthMu.Lock()
			callID = s.synth[shell]
			s.synthMu.Unlock()
		}
		s.em.shellOutput(context.Background(), shell, chunk, callID)
	})
}

// clone clones a repo (or refreshes the copy a snapshot holds) and
// checks out the session branch: from the remote branch when it exists
// (a sandbox recreated mid-session), else from the default branch.
func (s *session) clone(ctx context.Context, r *repoState) error {
	url, err := s.w.remote.CloneURL(r.FullName)
	if err != nil {
		return err
	}
	if err := agent.ValidateAgentBranch(r.Branch); err != nil {
		return err
	}
	script := fmt.Sprintf(`set -e
url=%s
dir=%s
branch=%s
def=%s
if [ -d "$dir/.git" ]; then
	git -C "$dir" remote set-url origin "$url"
	git -C "$dir" fetch --quiet --prune origin
else
	git clone --quiet "$url" "$dir"
fi
cd "$dir"
if [ -z "$def" ]; then
	def=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')
fi
[ -n "$def" ] || { echo "cannot tell the default branch" >&2; exit 3; }
if git rev-parse --verify -q "refs/remotes/origin/$branch" >/dev/null; then
	start="origin/$branch"
else
	start="origin/$def"
fi
git checkout -q -f -B "$branch" --no-track "$start"
git clean -q -fd
echo "default=$def"
echo "start=$start"`, shellQuote(url), shellQuote(r.Path), shellQuote(r.Branch), shellQuote(r.DefaultBranch))
	out, err := s.execOK(ctx, sandbox.WorkspaceDir, script, cloneTimeout)
	if err != nil {
		return err
	}
	for _, line := range strings.Split(out, "\n") {
		if v, ok := strings.CutPrefix(line, "default="); ok && r.DefaultBranch == "" {
			r.DefaultBranch = strings.TrimSpace(v)
		}
		if v, ok := strings.CutPrefix(line, "start="); ok && strings.TrimSpace(v) == "origin/"+r.Branch {
			s.log.Info("continuing from the pushed session branch", "repo", r.FullName)
		}
	}
	return s.w.d.Store.SetSessionRepoBranch(ctx, s.id, r.RepoID, r.Branch)
}

func shellQuote(v string) string {
	return "'" + strings.ReplaceAll(v, "'", `'\''`) + "'"
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := len(s) - n
	for cut < len(s) && (s[cut]&0xC0) == 0x80 {
		cut++
	}
	return "…" + s[cut:]
}
