package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/memevents"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/worker/memstore"
)

const (
	testOrg     = "00000000-0000-4000-8000-0000000000aa"
	testUser    = "00000000-0000-4000-8000-0000000000bb"
	testRepo    = "00000000-0000-4000-8000-0000000000c1"
	testRepo2   = "00000000-0000-4000-8000-0000000000c2"
	testSecret  = "SUPERSECRETVALUE42"
	firstToken  = "ghs_sandbox1"
	workerToken = "ghs_worker1"
)

// ---- fake GitHub ----

type fakeTokens struct {
	mu      sync.Mutex
	val     string
	handed  bool
	hooks   []func(github.Token)
	revoked bool
	done    chan struct{}
	scope   []string
}

func newFakeTokens(v string, scope []string) *fakeTokens {
	return &fakeTokens{val: v, done: make(chan struct{}), scope: scope}
}

func (f *fakeTokens) Token(context.Context) (github.Token, error) {
	f.mu.Lock()
	if f.revoked {
		f.mu.Unlock()
		return github.Token{}, github.ErrTokenSourceClosed
	}
	first := !f.handed
	f.handed = true
	t := github.Token{Value: f.val, ExpiresAt: time.Now().Add(time.Hour)}
	hooks := slices.Clone(f.hooks)
	f.mu.Unlock()
	if first {
		for _, h := range hooks {
			h(t)
		}
	}
	return t, nil
}

// rotate mints a new token, as a refresh would.
func (f *fakeTokens) rotate(v string) {
	f.mu.Lock()
	f.val = v
	t := github.Token{Value: v, ExpiresAt: time.Now().Add(time.Hour)}
	hooks := slices.Clone(f.hooks)
	f.mu.Unlock()
	for _, h := range hooks {
		h(t)
	}
}

func (f *fakeTokens) Run(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-f.done:
		return nil
	}
}

func (f *fakeTokens) Revoke(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.revoked {
		f.revoked = true
		close(f.done)
	}
	return nil
}

func (f *fakeTokens) isRevoked() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.revoked
}

func (f *fakeTokens) OnRefresh(fn func(github.Token)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hooks = append(f.hooks, fn)
}

type fakeGitHub struct {
	host *agenttest.Host

	mu      sync.Mutex
	sandbox []*fakeTokens
	worker  []*fakeTokens
	push    []*fakeTokens
	n       int
}

func (g *fakeGitHub) PushTokens(inst int64, repos []string) (TokenSource, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	t := newFakeTokens(fmt.Sprintf("ghs_push%d", len(g.push)+1), repos)
	g.push = append(g.push, t)
	return t, nil
}

func (g *fakeGitHub) pushTokens() []*fakeTokens {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.push)
}

func (g *fakeGitHub) SandboxTokens(inst int64, repos []string) (TokenSource, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.n++
	v := firstToken
	if g.n > 1 {
		v = fmt.Sprintf("ghs_sandbox%d", g.n)
	}
	t := newFakeTokens(v, repos)
	g.sandbox = append(g.sandbox, t)
	return t, nil
}

func (g *fakeGitHub) WorkerHost(inst int64, repos []string) (agent.RepoHost, TokenSource, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	t := newFakeTokens(workerToken, repos)
	g.worker = append(g.worker, t)
	return g.host, t, nil
}

func (g *fakeGitHub) lastSandbox() *fakeTokens {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.sandbox[len(g.sandbox)-1]
}

func (g *fakeGitHub) lastWorker() *fakeTokens {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.worker[len(g.worker)-1]
}

// ---- fake shells ----

type fakeShells struct {
	*agenttest.Shells
	onOutput func(shell string, chunk []byte)

	mu          sync.Mutex
	interrupts  chan struct{}
	reattached  int
	closed      []string
	interrupted int
}

func (f *fakeShells) Reattach(context.Context) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reattached++
	return nil, nil
}

func (f *fakeShells) Close(_ context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = append(f.closed, name)
	return nil
}

func (f *fakeShells) Interrupt(ctx context.Context, name string) error {
	f.mu.Lock()
	f.interrupted++
	select {
	case f.interrupts <- struct{}{}:
	default:
	}
	f.mu.Unlock()
	return f.Shells.Interrupt(ctx, name)
}

// ---- environment ----

type tenv struct {
	t      *testing.T
	st     *memstore.Store
	store  Store // what the worker uses (st, possibly wrapped)
	evs    *memevents.Store
	bus    *memevents.Bus
	leases *MemLeaser
	ctl    *control.Mem
	sbx    *memsandbox.Provider
	gh     *fakeGitHub
	llm    *llmtest.Provider
	kr     *secrets.Keyring
	opts   Options
	w      *Worker

	mu       sync.Mutex
	shells   []*fakeShells
	onExec   agenttest.ExecFunc
	sbRules  []memsandbox.RouteRule
	dirty    bool
	ahead    int
	pushed   int
	lockOut  string
	diffText string
	cloneOut string
	branch   string

	// The fake remote the worker pushes to.
	remoteRefs map[string]string
	updates    []fakeUpdate
	updateErr  error
}

// fakeUpdate is one push the fake remote received.
type fakeUpdate struct {
	Repo, Token string
	github.RefUpdate
	Pack []byte
}

// fakeTip is the fake sandbox's branch tip after ahead commits.
func fakeTip(ahead int) string { return fmt.Sprintf("%040x", 0xc0ffee00+ahead) }

// gitPush is the worker's GitPusher: the fake remote.
func (e *tenv) gitPush(_ context.Context, repo, token string) (PushRemote, error) {
	return &fakeRemote{e: e, repo: repo, token: token}, nil
}

type fakeRemote struct {
	e           *tenv
	repo, token string
}

func (r *fakeRemote) Ref(name string) string {
	r.e.mu.Lock()
	defer r.e.mu.Unlock()
	if name == "refs/heads/main" {
		return fakeTip(0)
	}
	return r.e.remoteRefs[r.repo+" "+name]
}

func (r *fakeRemote) Update(_ context.Context, u github.RefUpdate, pack []byte) error {
	r.e.mu.Lock()
	defer r.e.mu.Unlock()
	if r.e.updateErr != nil {
		return r.e.updateErr
	}
	if r.e.remoteRefs == nil {
		r.e.remoteRefs = map[string]string{}
	}
	r.e.updates = append(r.e.updates, fakeUpdate{Repo: r.repo, Token: r.token, RefUpdate: u, Pack: slices.Clone(pack)})
	r.e.remoteRefs[r.repo+" "+u.Ref] = u.New
	r.e.pushed = 0
	return nil
}

// pushes returns the updates the fake remote received.
func (e *tenv) pushes() []fakeUpdate {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.updates)
}

func newEnv(t *testing.T) *tenv {
	t.Helper()
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	kr, err := secrets.NewKeyring(key)
	if err != nil {
		t.Fatal(err)
	}
	e := &tenv{
		t: t, st: memstore.New(), evs: memevents.NewStore(), bus: memevents.NewBus(),
		leases: NewMemLeaser(), ctl: control.NewMem(), sbx: memsandbox.NewProvider(),
		gh: &fakeGitHub{host: &agenttest.Host{Issues: map[string]agent.Issue{
			"acme/web#7": {Repo: "acme/web", Number: 7, Title: "Login is broken", Body: "It fails.", State: "open", Author: "reporter"},
		}}},
		llm:      llmtest.New(),
		kr:       kr,
		lockOut:  "100644 deadbeef 0\tpackage-lock.json\n",
		diffText: "diff --git a/hello.txt b/hello.txt\n+hello\n",
		cloneOut: "default=main\nstart=origin/main\n",
		opts:     Options{CancelPoll: 20 * time.Millisecond, LeaseTTL: time.Second},
	}
	e.store = e.st
	e.sbx.NewHandler = func(sandbox.Spec) memsandbox.Handler { return e.handle }
	return e
}

// handle is every memsandbox's Exec handler.
func (e *tenv) handle(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
	e.mu.Lock()
	rules := slices.Clone(e.sbRules)
	dirty, ahead, pushed, lockOut, diffText, cloneOut := e.dirty, e.ahead, e.pushed, e.lockOut, e.diffText, e.cloneOut
	branch := e.branch
	e.mu.Unlock()
	for _, r := range rules {
		if r.Match(req) {
			return r.Handler(req)
		}
	}
	c := req.Cmd
	ok := func(out string) (*sandbox.ExecResult, error) { return &sandbox.ExecResult{Stdout: out}, nil }
	switch {
	case c == "id -u; id -g":
		return ok("1000\n1000\n")
	case req.User == "root" && strings.Contains(c, "base64 -d"):
		return ok("")
	case strings.Contains(c, "git config --global"):
		return ok("")
	case strings.HasPrefix(c, "mv -f "), strings.HasPrefix(c, "rm -f /run/keith"):
		return ok("")
	case strings.Contains(c, "git clone"):
		return ok(cloneOut)
	case strings.Contains(c, "git ls-files -s"):
		return ok(lockOut)
	case strings.Contains(c, "git diff --cached"):
		return ok(diffText)
	case strings.Contains(c, "git status --porcelain"):
		d := 0
		if dirty {
			d = 1
		}
		return ok(fmt.Sprintf("%d\n%d\n%d\n", d, ahead, pushed))
	case strings.Contains(c, " commit -q "):
		e.mu.Lock()
		e.dirty, e.ahead, e.pushed = false, e.ahead+1, e.pushed+1
		e.mu.Unlock()
		return ok("")
	case strings.Contains(c, "git push"):
		// The sandbox's credentials are read-only.
		return &sandbox.ExecResult{ExitCode: 128, Stderr: "remote: Permission denied (read-only token)"}, nil
	case strings.Contains(c, "git bundle create"):
		// The worker's push script: the remote has the fake commits
		// already, so no bundle is needed. The sandbox's tracking ref
		// is the remote's value.
		e.mu.Lock()
		lease := ""
		for k, v := range e.remoteRefs {
			if strings.HasSuffix(k, " refs/heads/"+branch) {
				lease = v
			}
		}
		e.mu.Unlock()
		return ok(fmt.Sprintf("tip=%s\nlease=%s\nmode=empty\n", fakeTip(ahead), lease))
	case strings.HasPrefix(c, "git update-ref"):
		return ok("")
	case c == "git rev-parse --abbrev-ref HEAD":
		return ok(branch + "\n")
	case c == "git add -A":
		return ok("")
	case strings.Contains(c, "git symbolic-ref"):
		return ok("origin/main\n")
	}
	return &sandbox.ExecResult{ExitCode: 127, Stderr: "no fake for: " + c}, nil
}

func (e *tenv) setChanges(dirty bool, ahead int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.dirty, e.ahead, e.pushed = dirty, ahead, ahead
}

func (e *tenv) newShells(sb sandbox.Sandbox, user string, onOutput func(string, []byte)) Shells {
	fs := &fakeShells{onOutput: onOutput, interrupts: make(chan struct{}, 16)}
	fs.Shells = &agenttest.Shells{OnOutput: onOutput, OnExec: func(name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
		e.mu.Lock()
		f := e.onExec
		e.mu.Unlock()
		if f != nil {
			return f(name, cmd, timeout)
		}
		return sandbox.ShellResult{OutputTail: "ok\n"}, nil
	}}
	e.mu.Lock()
	e.shells = append(e.shells, fs)
	e.mu.Unlock()
	return fs
}

func (e *tenv) lastShells() *fakeShells {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.shells) == 0 {
		return nil
	}
	return e.shells[len(e.shells)-1]
}

// worker builds the Worker (once).
func (e *tenv) worker() *Worker {
	e.t.Helper()
	if e.w != nil {
		return e.w
	}
	w, err := New(Deps{
		Store: e.store, Events: e.evs, Bus: e.bus, Leases: e.leases, Control: e.ctl,
		Sandboxes: e.sbx, GitHub: e.gh, GitPush: e.gitPush,
		LLM: func(name string) (llm.Provider, error) {
			if name == "missing" {
				return nil, errors.New("no key")
			}
			return e.llm, nil
		},
		Keyring:   e.kr,
		NewShells: e.newShells,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	}, e.opts)
	if err != nil {
		e.t.Fatal(err)
	}
	e.w = w
	return w
}

type sessOpt func(*store.Session)

func withRepos(rs ...store.SessionRepo) sessOpt {
	return func(s *store.Session) { s.Repos = rs }
}

func withPrompt(p string) sessOpt { return func(s *store.Session) { s.Prompt = p } }

func withIssue(url string) sessOpt {
	return func(s *store.Session) { s.Trigger = json.RawMessage(`{"kind":"web","issue_url":"` + url + `"}`) }
}

var sessSeq int

func webRepo() store.SessionRepo {
	return store.SessionRepo{RepoID: testRepo, FullName: "acme/web", InstallationID: 1, DefaultBranch: "main"}
}

func (e *tenv) newSession(opts ...sessOpt) store.Session {
	sessSeq++
	s := store.Session{
		ID:        fmt.Sprintf("00000000-0000-4000-8000-%012d", sessSeq),
		OrgID:     testOrg,
		CreatedBy: testUser,
		Prompt:    "Fix the login bug",
		Provider:  "anthropic",
		Model:     "claude-opus-5-5",
		Repos:     []store.SessionRepo{webRepo()},
	}
	for _, o := range opts {
		o(&s)
	}
	e.mu.Lock()
	e.branch = store.SessionBranch(s.ID)
	e.mu.Unlock()
	return e.st.AddSession(s)
}

func (e *tenv) profile(p store.RepoProfile) { e.st.SetProfile(p) }

func (e *tenv) addSecret(name, value, repoID string) {
	e.t.Helper()
	aad := store.SecretAAD(testOrg, repoID, name)
	ct, err := e.kr.Seal([]byte(value), aad)
	if err != nil {
		e.t.Fatal(err)
	}
	e.st.AddSecret(store.SecretCiphertext{OrgID: testOrg, RepoID: repoID, Name: name, Ciphertext: ct})
}

func (e *tenv) run(id string) error {
	e.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return e.worker().HandleRunSession(ctx, queue.RunSessionPayload{SessionID: id})
}

func (e *tenv) resume(id string) error {
	e.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return e.worker().HandleResumeSession(ctx, queue.ResumeSessionPayload{SessionID: id, Reason: queue.ResumeMessage, InputID: "m_x"})
}

// types returns the stored event types.
func (e *tenv) types(id string) []string { return e.evs.Types(id) }

func (e *tenv) events(id string, typ string) []events.Event {
	var out []events.Event
	for _, ev := range e.evs.All(id) {
		if ev.Type == typ {
			out = append(out, ev)
		}
	}
	return out
}

func payload[T any](t *testing.T, ev events.Event) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(ev.Payload, &v); err != nil {
		t.Fatalf("decode %s: %v", ev.Type, err)
	}
	return v
}

func (e *tenv) statuses(id string) []string {
	var out []string
	for _, ev := range e.events(id, events.TypeSessionStatus) {
		out = append(out, string(payload[events.SessionStatusPayload](e.t, ev).Status))
	}
	return out
}

func (e *tenv) sandbox(id string) *memsandbox.Sandbox {
	e.t.Helper()
	s := e.st.Session(id)
	sb, ok := e.sbx.Sandbox(s.SandboxID)
	if !ok {
		e.t.Fatalf("no sandbox %q", s.SandboxID)
	}
	return sb
}

func (e *tenv) execs(sb *memsandbox.Sandbox, sub string) []sandbox.ExecRequest {
	var out []sandbox.ExecRequest
	for _, r := range sb.ExecRequests() {
		if strings.Contains(r.Cmd, sub) {
			out = append(out, r)
		}
	}
	return out
}

// writes returns the data written to path, in order.
func writes(sb *memsandbox.Sandbox, path string) []string {
	var out []string
	for _, c := range sb.Calls() {
		if c.Op == memsandbox.OpWrite && c.Path == path {
			out = append(out, string(c.Data))
		}
	}
	return out
}

func wantSeq(t *testing.T, what string, got, want []string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("%s:\n got  %v\n want %v", what, got, want)
	}
}

// noSecrets fails if a stored event of the session contains v.
func (e *tenv) noSecrets(id string, vs ...string) {
	e.t.Helper()
	for _, ev := range e.evs.All(id) {
		for _, v := range vs {
			if strings.Contains(string(ev.Payload), v) {
				e.t.Errorf("event %d (%s) leaks %q: %s", ev.Seq, ev.Type, v, ev.Payload)
			}
		}
	}
}

func toolCall(id, name string, input any) *llm.Response { return llmtest.ToolCall(id, name, input) }
