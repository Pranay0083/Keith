package worker

import (
	"bytes"
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events/memevents"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/worker/memstore"
)

// pushEnv is a worker push against a real remote: git http-backend
// serving acme/web.git, and a memsandbox whose commands run on the host
// in a clone of it (/workspace/web).
type pushEnv struct {
	t      *testing.T
	home   string
	root   string // GIT_PROJECT_ROOT
	work   string // the "sandbox" clone
	sb     *memsandbox.Sandbox
	gh     *fakeGitHub
	s      *session
	repo   *repoState
	branch string

	mu       sync.Mutex
	auth     []string // Authorization headers the remote saw
	urls     []string
	swap     func(hostBundle string) // replaces the bundle the script wrote
	hostRuns int
}

const sandboxClone = "/workspace/web"

var bundlePathRE = regexp.MustCompile(`'(/tmp/keith-push-[0-9a-f]+\.bundle)'`)

func newPushEnv(t *testing.T, opts Options) *pushEnv {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	out, err := exec.Command("git", "--exec-path").Output()
	if err != nil {
		t.Fatal(err)
	}
	backend := filepath.Join(strings.TrimSpace(string(out)), "git-http-backend")
	if _, err := os.Stat(backend); err != nil {
		t.Skip("git-http-backend is not installed")
	}
	e := &pushEnv{t: t, home: t.TempDir(), root: t.TempDir()}
	h := &cgi.Handler{Path: backend, Root: "/git", Stderr: io.Discard, Env: []string{
		"GIT_PROJECT_ROOT=" + e.root, "GIT_HTTP_EXPORT_ALL=1", "REMOTE_USER=x-access-token",
		"GIT_CONFIG_NOSYSTEM=1", "HOME=" + e.home, "XDG_CONFIG_HOME=" + e.home,
	}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		e.mu.Lock()
		e.auth = append(e.auth, r.Header.Get("Authorization"))
		e.urls = append(e.urls, r.URL.String())
		e.mu.Unlock()
		user, pass, _ := r.BasicAuth()
		push := strings.HasSuffix(r.URL.Path, "/git-receive-pack") || r.URL.Query().Get("service") == "git-receive-pack"
		// Push tokens may push; the sandbox's tokens only fetch.
		if user != "x-access-token" || (push && !strings.HasPrefix(pass, "ghs_push")) {
			http.Error(w, "denied", http.StatusForbidden)
			return
		}
		r.Header.Del("Authorization")
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	// The remote: main, agent/other and tag v1.
	bare := e.bare()
	if err := os.MkdirAll(filepath.Dir(bare), 0o755); err != nil {
		t.Fatal(err)
	}
	e.git(e.root, "init", "-q", "--bare", bare)
	e.git(bare, "config", "http.receivepack", "true")
	seed := t.TempDir()
	e.git(seed, "init", "-q", ".")
	e.write(seed, "README.md", "# web\n")
	e.git(seed, "add", "-A")
	e.git(seed, "commit", "-qm", "init")
	e.git(seed, "branch", "agent/other")
	e.git(seed, "tag", "v1")
	e.git(seed, "push", "-q", bare, "main", "agent/other", "v1")

	// The sandbox's clone, on the session branch.
	sessID := "00000000-0000-4000-8000-0000000000f4"
	e.branch = store.SessionBranch(sessID)
	e.work = filepath.Join(t.TempDir(), "web")
	e.git(e.root, "clone", "-q", bare, e.work)
	e.git(e.work, "checkout", "-qb", e.branch)

	e.sb = memsandbox.New()
	e.sb.SetHandler(e.runOnHost)
	e.gh = &fakeGitHub{host: &agenttest.Host{}}
	opts.GitBaseURL = srv.URL + "/git/"
	w, err := New(Deps{
		Store: memstore.New(), Events: memevents.NewStore(), Bus: memevents.NewBus(), Leases: NewMemLeaser(),
		Sandboxes: memsandbox.NewProvider(), GitHub: e.gh,
		LLM:    func(string) (llm.Provider, error) { return nil, nil },
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}, opts)
	if err != nil {
		t.Fatal(err)
	}
	l, err := w.d.Leases.Acquire(context.Background(), sessID, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	e.s = newSession(w, sessID, l)
	e.s.sb, e.s.installationID = e.sb, 42
	e.repo = &repoState{
		SessionRepo: store.SessionRepo{FullName: "acme/web", Branch: e.branch, DefaultBranch: "main"},
		Name:        "web", Path: sandboxClone,
	}
	return e
}

func (e *pushEnv) bare() string { return filepath.Join(e.root, "acme", "web.git") }

func (e *pushEnv) git(dir string, args ...string) string {
	e.t.Helper()
	out, err := e.gitErr(dir, args...)
	if err != nil {
		e.t.Fatal(err)
	}
	return out
}

func (e *pushEnv) gitErr(dir string, args ...string) (string, error) {
	full := append([]string{"-c", "user.name=T", "-c", "user.email=t@example.test", "-c", "commit.gpgsign=false",
		"-c", "init.defaultBranch=main", "-c", "core.hooksPath=/dev/null"}, args...)
	cmd := exec.Command("git", full...)
	cmd.Dir = dir
	cmd.Env = e.env()
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, errb.String())
	}
	return strings.TrimSpace(out.String()), nil
}

func (e *pushEnv) env() []string {
	var env []string
	for _, kv := range os.Environ() {
		if k, _, _ := strings.Cut(kv, "="); !strings.HasPrefix(k, "GIT_") && k != "HOME" && k != "XDG_CONFIG_HOME" {
			env = append(env, kv)
		}
	}
	return append(env, "HOME="+e.home, "XDG_CONFIG_HOME="+e.home, "GIT_CONFIG_NOSYSTEM=1", "GIT_TERMINAL_PROMPT=0")
}

func (e *pushEnv) write(dir, name, content string) {
	e.t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		e.t.Fatal(err)
	}
}

// commit commits a file on the sandbox's branch and returns the tip.
func (e *pushEnv) commit(name, content string) string {
	e.t.Helper()
	e.write(e.work, name, content)
	e.git(e.work, "add", "-A")
	e.git(e.work, "commit", "-qm", "change "+name)
	return e.git(e.work, "rev-parse", "HEAD")
}

// refs returns the remote's refs as "name oid" lines.
func (e *pushEnv) refs() string {
	return e.git(e.bare(), "for-each-ref", "--format=%(refname) %(objectname)")
}

// runOnHost is the sandbox's Exec: the command runs with sh on the host,
// in the clone, with the bundle path mapped to a host file that is then
// copied into the memsandbox.
func (e *pushEnv) runOnHost(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
	if req.Dir != sandboxClone {
		return &sandbox.ExecResult{ExitCode: 127, Stderr: "unexpected dir " + req.Dir}, nil
	}
	cmdText := req.Cmd
	var sbPath, hostPath string
	if m := bundlePathRE.FindStringSubmatch(cmdText); m != nil {
		sbPath, hostPath = m[1], filepath.Join(e.t.TempDir(), "push.bundle")
		cmdText = strings.ReplaceAll(cmdText, m[1], hostPath)
	}
	cmd := exec.Command("sh", "-c", cmdText)
	cmd.Dir = e.work
	cmd.Env = e.env()
	for k, v := range req.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	err := cmd.Run()
	code := 0
	if ee, ok := err.(*exec.ExitError); ok {
		code = ee.ExitCode()
	} else if err != nil {
		return nil, err
	}
	e.mu.Lock()
	e.hostRuns++
	swap := e.swap
	e.mu.Unlock()
	if hostPath != "" {
		if _, err := os.Stat(hostPath); err == nil {
			if swap != nil {
				swap(hostPath)
			}
			data, err := os.ReadFile(hostPath)
			if err != nil {
				return nil, err
			}
			if err := e.sb.WriteFile(context.Background(), sbPath, data, 0o644); err != nil {
				return nil, err
			}
		}
	}
	return &sandbox.ExecResult{ExitCode: code, Stdout: out.String(), Stderr: errb.String()}, nil
}

func (e *pushEnv) push() error {
	e.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	return e.s.push(ctx, e.repo)
}

// checkTokens fails if a push token reached the sandbox (a command, its
// environment or a file), the worker's environment, a URL or err; and
// if any was not revoked.
func (e *pushEnv) checkTokens(errs ...error) {
	e.t.Helper()
	toks := e.gh.pushTokens()
	if len(toks) == 0 {
		e.t.Fatal("no push token was minted")
	}
	for _, ft := range toks {
		v := ft.val
		if !ft.isRevoked() {
			e.t.Errorf("push token %s was not revoked", v)
		}
		if !slicesEqual(ft.scope, []string{"acme/web"}) {
			e.t.Errorf("push token scope = %v", ft.scope)
		}
		for _, c := range e.sb.Calls() {
			if strings.Contains(c.Exec.Cmd, v) || strings.Contains(string(c.Data), v) || strings.Contains(c.Path, v) {
				e.t.Errorf("the push token reached the sandbox: %+v", c)
			}
			for k, val := range c.Exec.Env {
				if strings.Contains(k+"="+val, v) {
					e.t.Errorf("the push token is in a sandbox command's environment: %s", k)
				}
			}
		}
		for _, kv := range os.Environ() {
			if strings.Contains(kv, v) {
				e.t.Errorf("the push token is in the worker's environment")
			}
		}
		e.mu.Lock()
		for _, u := range e.urls {
			if strings.Contains(u, v) {
				e.t.Errorf("the push token is in a URL: %s", u)
			}
		}
		e.mu.Unlock()
		for _, err := range errs {
			if err != nil && strings.Contains(err.Error(), v) {
				e.t.Errorf("an error leaks the push token: %v", err)
			}
		}
		if got := e.s.em.redact("x " + v + " y"); strings.Contains(got, v) {
			e.t.Errorf("the redactor does not know the push token: %q", got)
		}
	}
}

// sameRefLines compares two ref listings as sets of lines.
func sameRefLines(a, b string) bool {
	as, bs := strings.Split(a, "\n"), strings.Split(b, "\n")
	slices.Sort(as)
	slices.Sort(bs)
	return slices.Equal(as, bs)
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPushUpdatesOnlyTheSessionBranch(t *testing.T) {
	e := newPushEnv(t, Options{})
	before := e.refs()
	tip := e.commit("fix.txt", "fixed\n")
	if err := e.push(); err != nil {
		t.Fatalf("push: %v", err)
	}
	if got, want := e.refs(), before+"\nrefs/heads/"+e.branch+" "+tip; !sameRefLines(got, want) {
		t.Fatalf("remote refs:\n%s\nwant\n%s", got, want)
	}
	if got := e.git(e.bare(), "show", e.branch+":fix.txt"); got != "fixed" {
		t.Fatalf("fix.txt = %q", got)
	}
	// The sandbox records the push like git push -u.
	if got := e.git(e.work, "rev-parse", "refs/remotes/origin/"+e.branch); got != tip {
		t.Errorf("tracking ref = %s", got)
	}
	if got := e.git(e.work, "config", "branch."+e.branch+".merge"); got != "refs/heads/"+e.branch {
		t.Errorf("upstream = %s", got)
	}
	// The bundle was read with the cap and removed.
	var read, removed bool
	for _, c := range e.sb.Calls() {
		if c.Op == memsandbox.OpRead && strings.HasPrefix(c.Path, "/tmp/keith-push-") && c.Limit == DefaultMaxPushBytes {
			read = true
		}
		if c.Op == memsandbox.OpRemove && strings.HasPrefix(c.Path, "/tmp/keith-push-") {
			removed = true
		}
	}
	if !read || !removed {
		t.Errorf("bundle read %v, removed %v: %+v", read, removed, e.sb.Calls())
	}
	for p := range e.sb.Files() {
		if strings.Contains(p, "keith-push") {
			t.Errorf("bundle left in the sandbox: %s", p)
		}
	}

	// Nothing new: no update. More commits: a fast-forward.
	if err := e.push(); err != nil {
		t.Fatalf("second push: %v", err)
	}
	tip2 := e.commit("more.txt", "more\n")
	if err := e.push(); err != nil {
		t.Fatalf("third push: %v", err)
	}
	if got := e.git(e.bare(), "rev-parse", e.branch); got != tip2 {
		t.Fatalf("branch = %s, want %s", got, tip2)
	}
	// The agent rewrote its own branch: force-with-lease lets it through.
	e.write(e.work, "more.txt", "rewritten\n")
	e.git(e.work, "commit", "-qa", "--amend", "-m", "more, rewritten")
	tip3 := e.git(e.work, "rev-parse", "HEAD")
	if err := e.push(); err != nil {
		t.Fatalf("push of a rewritten branch: %v", err)
	}
	if got := e.git(e.bare(), "rev-parse", e.branch); got != tip3 {
		t.Fatalf("branch = %s, want %s", got, tip3)
	}
	// The branch reset to main: the remote has every object.
	e.git(e.work, "reset", "-q", "--hard", "origin/main")
	if err := e.push(); err != nil {
		t.Fatalf("push of a commit the remote has: %v", err)
	}
	mainOID := e.git(e.bare(), "rev-parse", "main")
	if got, want := e.refs(), before+"\nrefs/heads/"+e.branch+" "+mainOID; !sameRefLines(got, want) {
		t.Fatalf("remote refs:\n%s\nwant\n%s", got, want)
	}
	e.checkTokens()
	e.mu.Lock()
	for _, a := range e.auth {
		if !strings.HasPrefix(a, "Basic ") {
			t.Errorf("request without the push token: %q", a)
		}
	}
	e.mu.Unlock()
}

func TestPushRefusesForeignRefsInTheBundle(t *testing.T) {
	e := newPushEnv(t, Options{})
	before := e.refs()
	e.commit("fix.txt", "fixed\n")
	other := t.TempDir()
	e.git(other, "clone", "-q", e.bare(), "x")
	otherDir := filepath.Join(other, "x")
	e.write(otherDir, "evil.txt", "evil\n")
	e.git(otherDir, "add", "-A")
	e.git(otherDir, "commit", "-qm", "evil")

	want := map[string]string{
		"two refs":       "must carry exactly refs/heads/" + e.branch,
		"another branch": `carries ["refs/heads/main"]`,
		"another tip":    "not at the branch tip",
		"garbage":        "malformed ref line",
	}
	cases := map[string]func(path string){
		// The session branch and main.
		"two refs": func(p string) {
			e.git(e.work, "bundle", "create", "-q", p, "refs/heads/"+e.branch, "refs/remotes/origin/main")
		},
		// main only.
		"another branch": func(p string) {
			e.git(otherDir, "bundle", "create", "-q", p, "refs/heads/main", "--not", "origin/main")
		},
		// The right name at another commit.
		"another tip": func(p string) {
			e.git(otherDir, "branch", "-f", e.branch, "HEAD")
			e.git(otherDir, "bundle", "create", "-q", p, "refs/heads/"+e.branch, "--not", "origin/main")
		},
		"garbage": func(p string) {
			if err := os.WriteFile(p, []byte("# v2 git bundle\nnonsense\n\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		},
	}
	var errs []error
	for name, swap := range cases {
		e.mu.Lock()
		e.swap = func(p string) {
			_ = os.Remove(p)
			swap(p)
		}
		e.mu.Unlock()
		err := e.push()
		if err == nil || !strings.Contains(err.Error(), "refusing the push") || !strings.Contains(err.Error(), want[name]) {
			t.Errorf("%s: push = %v", name, err)
		}
		errs = append(errs, err)
		if got := e.refs(); got != before {
			t.Fatalf("%s: remote refs changed:\n%s", name, got)
		}
	}
	e.checkTokens(errs...)
}

func TestPushRefusesAnOversizedBundle(t *testing.T) {
	e := newPushEnv(t, Options{MaxPushBytes: 32 << 10})
	before := e.refs()
	big := make([]byte, 256<<10)
	_, _ = rand.Read(big)
	e.commit("big.bin", string(big))
	err := e.push()
	if err == nil || !strings.Contains(err.Error(), "larger than") {
		t.Fatalf("push = %v", err)
	}
	if got := e.refs(); got != before {
		t.Fatalf("remote refs changed:\n%s", got)
	}
	for _, c := range e.sb.Calls() {
		if c.Op == memsandbox.OpRead && c.Limit != 32<<10 {
			t.Errorf("read with limit %d", c.Limit)
		}
	}
	e.checkTokens(err)
}

func TestPushLease(t *testing.T) {
	e := newPushEnv(t, Options{})
	e.commit("a.txt", "1\n")
	if err := e.push(); err != nil {
		t.Fatal(err)
	}
	// Someone else pushes to the session branch.
	other := t.TempDir()
	e.git(other, "clone", "-q", "-b", e.branch, e.bare(), "x")
	otherDir := filepath.Join(other, "x")
	e.write(otherDir, "theirs.txt", "theirs\n")
	e.git(otherDir, "add", "-A")
	e.git(otherDir, "commit", "-qm", "theirs")
	e.git(otherDir, "push", "-q", "origin", e.branch)
	theirs := e.git(otherDir, "rev-parse", "HEAD")

	e.commit("b.txt", "2\n")
	// This task pushed the branch before, but not their commit.
	err := e.push()
	if err == nil || !strings.Contains(err.Error(), "has commits this sandbox has not seen") || !strings.Contains(err.Error(), "git fetch origin "+e.branch) {
		t.Fatalf("push over someone else's commit = %v", err)
	}
	if got := e.git(e.bare(), "rev-parse", e.branch); got != theirs {
		t.Fatalf("their commit was overwritten: %s", got)
	}
	// After fetching and merging, the push goes through.
	e.git(e.work, "fetch", "-q", "origin")
	e.git(e.work, "merge", "-q", "--no-edit", "origin/"+e.branch)
	if err := e.push(); err != nil {
		t.Fatalf("push after merging: %v", err)
	}
	// A branch deleted on the remote is created again.
	e.git(e.bare(), "update-ref", "-d", "refs/heads/"+e.branch)
	tip := e.commit("c.txt", "3\n")
	if err := e.push(); err != nil {
		t.Fatalf("push after the branch was deleted: %v", err)
	}
	if got := e.git(e.bare(), "rev-parse", e.branch); got != tip {
		t.Fatalf("branch = %s", got)
	}
	e.checkTokens(err)
}

func TestPushOnlyTheSessionBranchName(t *testing.T) {
	e := newPushEnv(t, Options{})
	before := e.refs()
	e.commit("a.txt", "1\n")
	for _, b := range []string{"main", "agent/other", "agent/" + "00000000-0000-4000-8000-000000000001", ""} {
		e.repo.Branch = b
		if err := e.push(); err == nil || !strings.Contains(err.Error(), "only the session branch") {
			t.Errorf("push of %q = %v", b, err)
		}
	}
	if len(e.gh.pushTokens()) != 0 || e.hostRuns != 0 {
		t.Error("minted a token or ran the push script for another branch")
	}
	if got := e.refs(); got != before {
		t.Fatalf("remote refs changed:\n%s", got)
	}
	// A sandbox without the branch fails clearly.
	e.repo.Branch = e.branch
	e.git(e.work, "checkout", "-q", "main")
	e.git(e.work, "branch", "-D", e.branch)
	if err := e.push(); err == nil || !strings.Contains(err.Error(), "does not exist in the sandbox") {
		t.Fatalf("push without the branch = %v", err)
	}
	e.checkTokens()
}
