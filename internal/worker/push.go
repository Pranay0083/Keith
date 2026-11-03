package worker

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// The sandbox's GitHub token is read-only (github.SandboxPermissions), so
// nothing in a sandbox can push. Every push of a session branch, by
// open_pr (agent.Run.Push) or by the worker's own ship, goes through
// session.push:
//
//  1. Mint a push token (github.PushPermissions) for the one repo. It
//     stays in the worker and is revoked when the push returns.
//  2. Open the remote's receive-pack service and read the branch's
//     current value and the default branch's.
//  3. In the sandbox, as the sandbox user: resolve the branch's tip and
//     its remote-tracking ref (the lease), and write a git bundle of the
//     branch relative to the remote's commits the sandbox has.
//  4. Read the bundle (at most Options.MaxPushBytes) and accept it only
//     if it carries exactly one ref, refs/heads/agent/<session-id>, at
//     that tip.
//  5. Send one ref update, refs/heads/agent/<session-id> from its current
//     value to the tip, with the bundle's pack. No other ref can change.
//
// The branch is updated with force-with-lease semantics: the agent may
// rewrite its own branch, but a push is refused when the remote branch
// holds commits the sandbox has not seen (someone else pushed to it),
// unless the worker itself pushed them in this task. A branch that does
// not exist on the remote is created.

// DefaultMaxPushBytes caps the bundle one push reads from the sandbox.
const DefaultMaxPushBytes = 100 << 20

const (
	pushTimeout       = 5 * time.Minute
	pushRevokeTimeout = 10 * time.Second
)

// PushRemote is an open push to one repo: *github.ReceivePack.
type PushRemote interface {
	// Ref returns the object ID ref points at on the remote, or "" when
	// there is no such ref.
	Ref(name string) string
	// Update moves one ref, sending pack (github.RefUpdate).
	Update(ctx context.Context, u github.RefUpdate, pack []byte) error
}

// GitPusher opens a push to repo (owner/name) with token, a push token
// (contents: write) that never enters the sandbox.
type GitPusher func(ctx context.Context, repo, token string) (PushRemote, error)

// receivePackPusher is the production GitPusher: github's receive-pack
// client over remote.
func receivePackPusher(remote github.GitRemote) GitPusher {
	return func(ctx context.Context, repo, token string) (PushRemote, error) {
		rp, err := remote.OpenReceivePack(ctx, nil, repo, token)
		if err != nil {
			return nil, err
		}
		return rp, nil
	}
}

// localBranch is what the sandbox reports for a push.
type localBranch struct {
	tip   string // the branch's commit
	lease string // its remote-tracking ref, "" if none
	// bundled is set when a bundle of the new commits was written; else
	// the remote already has the tip's objects.
	bundled bool
}

// pushHook is agent.Run.Push: open_pr committed in the sandbox and asks
// the worker to push.
func (s *session) pushHook(ctx context.Context, rc agent.RepoContext) error {
	for _, r := range s.repos {
		if r.FullName == rc.FullName {
			return s.push(ctx, r)
		}
	}
	return fmt.Errorf("%s is not one of this session's repos", rc.FullName)
}

// push pushes r's session branch, refs/heads/agent/<session-id>, to the
// remote from the worker (see the comment at the top of this file). Its
// errors are redacted and suitable for the model and the user.
func (s *session) push(ctx context.Context, r *repoState) error {
	err := s.pushBranch(ctx, r)
	if err == nil || ctx.Err() != nil {
		return err
	}
	return errors.New(s.em.redact(err.Error()))
}

func (s *session) pushBranch(ctx context.Context, r *repoState) error {
	branch := store.SessionBranch(s.id)
	if r.Branch != branch {
		return fmt.Errorf("refusing to push %s: only the session branch %s may be pushed, not %q", r.FullName, branch, r.Branch)
	}
	if err := github.ValidateAgentBranch(branch); err != nil {
		return fmt.Errorf("refusing to push %s: %w", r.FullName, err)
	}
	ref := "refs/heads/" + branch
	ctx, cancel := context.WithTimeout(ctx, pushTimeout)
	defer cancel()

	ts, err := s.w.d.GitHub.PushTokens(s.installationID, []string{r.FullName})
	if err != nil {
		return fmt.Errorf("could not set up the push token for %s: %w", r.FullName, err)
	}
	defer func() {
		rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), pushRevokeTimeout)
		defer cancel()
		if err := ts.Revoke(rctx); err != nil {
			s.log.Warn("revoke the push token", "repo", r.FullName, "err", s.em.redact(err.Error()))
		}
	}()
	tok, err := ts.Token(ctx)
	if err != nil {
		return fmt.Errorf("could not mint the push token for %s: %w", r.FullName, err)
	}
	s.em.addSecrets(tok.Value)
	remote, err := s.w.d.GitPush(ctx, r.FullName, tok.Value)
	if err != nil {
		return err
	}
	cur := remote.Ref(ref)
	var have []string
	for _, oid := range []string{cur, remote.Ref("refs/heads/" + r.DefaultBranch)} {
		if github.IsObjectID(oid) && oid != github.ZeroOID {
			have = append(have, oid)
		}
	}

	bundlePath := "/tmp/keith-push-" + randomHex(8) + ".bundle"
	defer func() {
		_ = s.sb.RemoveFile(context.WithoutCancel(ctx), bundlePath)
	}()
	local, err := s.bundle(ctx, r, ref, bundlePath, have)
	if err != nil {
		return err
	}
	if cur != "" && cur != local.lease && cur != s.pushedTip(r.FullName) {
		return fmt.Errorf("%s on %s has commits this sandbox has not seen (it is at %.12s; the sandbox last saw %s). Fetch it (git fetch origin %s), integrate those commits, and push again",
			branch, r.FullName, cur, shortOID(local.lease), branch)
	}
	if cur == local.tip {
		s.trackPushed(ctx, r, local.tip)
		return nil
	}
	pack := github.EmptyPack()
	if local.bundled {
		if pack, err = s.readBundle(ctx, bundlePath, ref, local.tip); err != nil {
			return err
		}
	}
	if err := remote.Update(ctx, github.RefUpdate{Ref: ref, Old: cur, New: local.tip}, pack); err != nil {
		return err
	}
	s.setPushedTip(r.FullName, local.tip)
	s.trackPushed(ctx, r, local.tip)
	return nil
}

// pushScript runs in the repo as the sandbox user. It prints the
// branch's tip and its remote-tracking ref, then either "mode=empty"
// (the remote has every commit) or writes a bundle of the branch,
// excluding the remote's commits the sandbox has, and prints
// "mode=bundle".
const pushScript = `set -e
ref=%s
track=%s
def=%s
out=%s
tip=$(git rev-parse --verify -q "$ref^{commit}") || { echo "the branch ${ref#refs/heads/} does not exist in the sandbox" >&2; exit 3; }
have=%s
lease=$(git rev-parse --verify -q "$track^{commit}") || lease=
echo "tip=$tip"
echo "lease=$lease"
set --
for c in $have; do
	if git cat-file -e "$c^{commit}" 2>/dev/null; then set -- "$@" "$c"; fi
done
if [ -n "$def" ] && d=$(git rev-parse --verify -q "$def^{commit}"); then set -- "$@" "$d"; fi
if [ $# -gt 0 ]; then
	n=$(git rev-list --count "$tip" --not "$@")
	if [ "$n" = 0 ]; then echo "mode=empty"; exit 0; fi
fi
rm -f "$out"
if [ $# -gt 0 ]; then
	git bundle create -q "$out" "$ref" --not "$@"
else
	git bundle create -q "$out" "$ref"
fi
echo "mode=bundle"`

// bundle runs pushScript for r and parses its report.
func (s *session) bundle(ctx context.Context, r *repoState, ref, path string, have []string) (localBranch, error) {
	track := "refs/remotes/origin/" + strings.TrimPrefix(ref, "refs/heads/")
	def := ""
	if r.DefaultBranch != "" {
		def = "refs/remotes/origin/" + r.DefaultBranch
	}
	// have holds object IDs only (checked by the caller), so it splits
	// safely on spaces.
	script := fmt.Sprintf(pushScript, shellQuote(ref), shellQuote(track), shellQuote(def), shellQuote(path), shellQuote(strings.Join(have, " ")))
	out, err := s.execOK(ctx, r.Path, script, pushTimeout)
	if err != nil {
		if ctx.Err() != nil {
			return localBranch{}, ctx.Err()
		}
		return localBranch{}, fmt.Errorf("could not bundle %s for the push: %w", r.FullName, err)
	}
	var lb localBranch
	mode := ""
	for _, line := range strings.Split(out, "\n") {
		k, v, _ := strings.Cut(strings.TrimSpace(line), "=")
		switch k {
		case "tip":
			lb.tip = v
		case "lease":
			lb.lease = v
		case "mode":
			mode = v
		}
	}
	switch {
	case !github.IsObjectID(lb.tip) || lb.tip == github.ZeroOID:
		return localBranch{}, fmt.Errorf("could not read the tip of %s in the sandbox", ref)
	case lb.lease != "" && !github.IsObjectID(lb.lease):
		return localBranch{}, fmt.Errorf("could not read the remote-tracking ref of %s in the sandbox", ref)
	case mode != "empty" && mode != "bundle":
		return localBranch{}, fmt.Errorf("could not bundle %s for the push: unexpected output", r.FullName)
	}
	lb.bundled = mode == "bundle"
	return lb, nil
}

// readBundle reads the sandbox's bundle and returns its pack, if it
// carries exactly one ref, ref, at tip.
func (s *session) readBundle(ctx context.Context, path, ref, tip string) ([]byte, error) {
	limit := s.w.o.MaxPushBytes
	data, err := s.sb.ReadFileLimit(ctx, path, limit)
	if errors.Is(err, sandbox.ErrFileTooLarge) {
		return nil, fmt.Errorf("the push is larger than %d MiB, the most the worker accepts. Remove large or generated files (build output, dependencies, binaries) from the branch's commits and try again", limit>>20)
	}
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("could not read the push bundle from the sandbox: %w", err)
	}
	b, err := github.ParseBundle(data)
	if err != nil {
		return nil, fmt.Errorf("refusing the push: %w", err)
	}
	if len(b.Refs) != 1 || b.Refs[0].Name != ref {
		names := make([]string, len(b.Refs))
		for i, r := range b.Refs {
			names[i] = r.Name
		}
		return nil, fmt.Errorf("refusing the push: the bundle must carry exactly %s, and carries %q", ref, names)
	}
	if b.Refs[0].OID != tip {
		return nil, fmt.Errorf("refusing the push: the bundle's %s is at %.12s, not at the branch tip %.12s (did the branch change during the push?)", ref, b.Refs[0].OID, tip)
	}
	return b.Pack, nil
}

// trackPushed records a push in the sandbox, as git push -u would: the
// remote-tracking ref (the next push's lease, and the worker's count of
// unpushed commits) and the branch's upstream. It is best effort.
func (s *session) trackPushed(ctx context.Context, r *repoState, tip string) {
	cmd := fmt.Sprintf("git update-ref %s %s && git config %s origin && git config %s %s",
		shellQuote("refs/remotes/origin/"+r.Branch), shellQuote(tip),
		shellQuote("branch."+r.Branch+".remote"), shellQuote("branch."+r.Branch+".merge"), shellQuote("refs/heads/"+r.Branch))
	if _, err := s.execOK(ctx, r.Path, cmd, gitCmdTimeout); err != nil && ctx.Err() == nil {
		s.log.Warn("record the push in the sandbox", "repo", r.FullName, "err", err)
	}
}

func (s *session) pushedTip(repo string) string {
	s.pushMu.Lock()
	defer s.pushMu.Unlock()
	return s.pushed[repo]
}

func (s *session) setPushedTip(repo, tip string) {
	s.pushMu.Lock()
	defer s.pushMu.Unlock()
	if s.pushed == nil {
		s.pushed = map[string]string{}
	}
	s.pushed[repo] = tip
}

func shortOID(oid string) string {
	if oid == "" {
		return "no remote branch"
	}
	return oid[:min(12, len(oid))]
}
