package e2e

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/hibiken/asynq"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

// sbExec runs cmd in sb and returns its result, failing t if it could
// not run.
func sbExec(t testing.TB, sb sandbox.Sandbox, req sandbox.ExecRequest) *sandbox.ExecResult {
	t.Helper()
	if req.Timeout == 0 {
		req.Timeout = 2 * time.Minute
	}
	res, err := sb.Exec(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("exec %q: %v", req.Cmd, err)
	}
	return res
}

// sbMust runs cmd in sb and fails t unless it exits 0.
func sbMust(t testing.TB, sb sandbox.Sandbox, req sandbox.ExecRequest) string {
	t.Helper()
	res := sbExec(t, sb, req)
	if res.ExitCode != 0 {
		t.Fatalf("exec %q: exit %d\nstdout: %s\nstderr: %s", req.Cmd, res.ExitCode, res.Stdout, res.Stderr)
	}
	return res.Stdout
}

// TestHarnessGitFromSandbox checks the git server, the fake GitHub's
// tokens and the credential helper together: a docker sandbox clones
// and pushes with a minted contents:write token and nothing else, and
// the sandbox token (SandboxPermissions, contents:read) fetches but
// cannot push.
func TestHarnessGitFromSandbox(t *testing.T) {
	storetest.Skip(t)
	ctx := context.Background()

	gh := newFakeGitHubAPI(t)
	app := newTestApp(t, gh)
	gs := newGitServer(t)
	gs.setAllow(func(token string, push bool) bool { return gh.tokenAllows(token, "contents", push) })
	gs.setScope(gh.tokenCovers)
	gs.seed(t, "acme/web", "main", map[string]string{"README.md": "# web\n", "src/app.txt": "v1\n"})
	gs.seed(t, "acme/other", "main", map[string]string{"README.md": "# other\n"})
	instID := randID()
	gh.addInstallation(instID, "acme", "acme/web", "acme/other")

	tok, err := app.InstallationToken(ctx, instID, []string{"acme/web"}, github.PushPermissions)
	if err != nil {
		t.Fatalf("mint a contents:write token: %v", err)
	}
	readTok, err := app.InstallationToken(ctx, instID, []string{"acme/web"}, github.SandboxPermissions)
	if err != nil {
		t.Fatalf("mint the sandbox token: %v", err)
	}
	if m := gh.mints(); len(m) != 2 || !slices.Equal(m[0].Repositories, []string{"acme/web"}) ||
		m[0].Permissions["contents"] != "write" || m[0].InstallationID != instID ||
		len(m[1].Permissions) != 1 || m[1].Permissions["contents"] != "read" {
		t.Fatalf("mints = %+v", m)
	}

	remote, err := github.ParseGitRemote(gs.BaseURL())
	if err != nil {
		t.Fatal(err)
	}
	cloneURL, err := remote.CloneURL("acme/web")
	if err != nil {
		t.Fatal(err)
	}

	sb, err := newSandboxProvider(t).Create(ctx, sandbox.Spec{})
	if err != nil {
		t.Fatalf("create sandbox: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if err := sb.Delete(ctx); err != nil {
			t.Errorf("delete sandbox %s: %v", sb.ID(), err)
		}
	})

	// The credential helper, installed as the worker does
	// (internal/github.CredentialHelperScript).
	if err := sb.WriteFile(ctx, "/tmp/git-credential-keith", []byte(remote.CredentialHelperScript()), 0o644); err != nil {
		t.Fatal(err)
	}
	sbMust(t, sb, sandbox.ExecRequest{User: "root", Cmd: "install -m 0755 -o root -g root /tmp/git-credential-keith " +
		github.CredentialHelperPath + " && rm /tmp/git-credential-keith && install -d -m 0700 -o agent -g agent /run/keith"})
	key := remote.CredentialKey()
	sbMust(t, sb, sandbox.ExecRequest{Cmd: "git config --global credential." + key + ".helper '' && " +
		"git config --global --add credential." + key + ".helper " + github.CredentialHelperPath + " && " +
		"git config --global user.name 'Keith Agent' && git config --global user.email agent@keith.test"})
	writeToken := func(v string) {
		t.Helper()
		if err := sb.WriteFile(ctx, "/run/keith/git-token.tmp", []byte(v+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		sbMust(t, sb, sandbox.ExecRequest{Cmd: "mv -f /run/keith/git-token.tmp " + github.GitTokenPath})
	}
	noPrompt := map[string]string{"GIT_TERMINAL_PROMPT": "0"}

	// Without a token file the helper answers nothing: 401.
	res := sbExec(t, sb, sandbox.ExecRequest{Cmd: "git clone " + cloneURL + " /workspace/denied", Env: noPrompt})
	if res.ExitCode == 0 {
		t.Fatal("clone without a token succeeded")
	}
	if !slices.ContainsFunc(gs.requests(), func(r gitRequest) bool {
		return r.Repo == "acme/web" && r.Token == "" && r.Status == http.StatusUnauthorized
	}) {
		t.Fatalf("no 401 for the tokenless clone: %+v", gs.requests())
	}

	// With the contents:write token: clone, commit on agent/test, push.
	writeToken(tok.Value)
	sbMust(t, sb, sandbox.ExecRequest{Cmd: "git clone " + cloneURL + " /workspace/web", Env: noPrompt})
	if got := strings.TrimSpace(sbMust(t, sb, sandbox.ExecRequest{Dir: "/workspace/web", Cmd: "git config --get remote.origin.url"})); got != cloneURL {
		t.Fatalf("origin = %q, want %q (no token in the URL)", got, cloneURL)
	}
	// big.bin (3 MB, incompressible) makes the pack exceed http.postBuffer,
	// so git sends it chunked.
	sbMust(t, sb, sandbox.ExecRequest{Dir: "/workspace/web", Env: noPrompt, Cmd: "git checkout -q -b agent/test && " +
		"printf 'v2\\n' > src/app.txt && head -c 3000000 /dev/urandom > big.bin && git add -A && " +
		"git commit -qm 'e2e: bump app' && git push -q origin agent/test"})
	if bs := gs.branches(t, "acme/web"); !slices.Equal(bs, []string{"agent/test", "main"}) {
		t.Fatalf("branches = %v", bs)
	}
	if got, ok := gs.show(t, "acme/web", "agent/test", "src/app.txt"); !ok || got != "v2\n" {
		t.Fatalf("agent/test:src/app.txt = %q, %v", got, ok)
	}
	if got, _ := gs.show(t, "acme/web", "main", "src/app.txt"); got != "v1\n" {
		t.Fatalf("main:src/app.txt = %q", got)
	}
	if got, ok := gs.show(t, "acme/web", "agent/test", "big.bin"); !ok || len(got) != 3000000 {
		t.Fatalf("agent/test:big.bin has %d bytes, %v", len(got), ok)
	}
	if lg := gs.log(t, "acme/web", "agent/test"); !slices.Equal(lg, []string{"e2e: bump app", "Initial commit"}) {
		t.Fatalf("log = %q", lg)
	}
	if !slices.ContainsFunc(gs.requests(), func(r gitRequest) bool {
		return r.Token == tok.Value && r.Push && r.Method == http.MethodPost && r.Chunked && r.Status == http.StatusOK
	}) {
		t.Fatalf("no authenticated chunked push: %+v", gs.requests())
	}

	// The token is scoped to acme/web: acme/other is refused.
	otherURL, _ := remote.CloneURL("acme/other")
	if res := sbExec(t, sb, sandbox.ExecRequest{Cmd: "git clone " + otherURL + " /workspace/other", Env: noPrompt}); res.ExitCode == 0 {
		t.Fatal("clone of a repo outside the token's scope succeeded")
	}

	// The sandbox token (contents:read) fetches but cannot push.
	writeToken(readTok.Value)
	sbMust(t, sb, sandbox.ExecRequest{Dir: "/workspace/web", Env: noPrompt, Cmd: "git fetch -q origin"})
	res = sbExec(t, sb, sandbox.ExecRequest{Dir: "/workspace/web", Env: noPrompt,
		Cmd: "git commit -q --allow-empty -m nope && git push -q origin HEAD:agent/denied"})
	if res.ExitCode == 0 {
		t.Fatal("push with a contents:read token succeeded")
	}
	if slices.Contains(gs.branches(t, "acme/web"), "agent/denied") {
		t.Fatal("agent/denied was pushed")
	}

	// A revoked token is refused.
	writeToken(tok.Value)
	if err := app.RevokeToken(ctx, tok.Value); err != nil {
		t.Fatal(err)
	}
	if res := sbExec(t, sb, sandbox.ExecRequest{Dir: "/workspace/web", Env: noPrompt, Cmd: "git fetch -q origin"}); res.ExitCode == 0 {
		t.Fatal("fetch with a revoked token succeeded")
	}
	if r := gh.revoked(); !slices.Equal(r, []string{tok.Value}) {
		t.Fatalf("revoked = %v", r)
	}
	if u := gh.unexpectedCalls(); len(u) > 0 {
		t.Fatalf("unexpected GitHub calls: %v", u)
	}
}

// TestHarnessFakeGitHub drives the fake GitHub API through the worker's
// real github.Host and TokenSource, so the fake answers what go-github
// sends.
func TestHarnessFakeGitHub(t *testing.T) {
	storetest.Skip(t)
	ctx := context.Background()
	gh := newFakeGitHubAPI(t)
	app := newTestApp(t, gh)
	instID := randID()
	gh.addInstallation(instID, "acme", "acme/web")
	gh.setHeadExists(func(repo, branch string) bool {
		return repo == "acme/web" && (branch == "main" || branch == "agent/test")
	})
	gh.addIssue("acme/web", 7, ghIssue{
		Title: "Fix the build", Body: "It is red.", Author: "octo", Labels: []string{"bug"},
		Comments: []ghComment{{Author: "a", Body: "first"}, {Author: "b", Body: "second"}},
	})

	ts, err := github.NewTokenSource(app, instID, []string{"acme/web"}, github.WorkerPermissions)
	if err != nil {
		t.Fatal(err)
	}
	host := github.NewHost(ts)
	is, err := host.ReadIssue(ctx, "acme/web", 7)
	if err != nil {
		t.Fatalf("ReadIssue: %v", err)
	}
	if is.Title != "Fix the build" || is.Body != "It is red." || is.State != "open" || is.Author != "octo" ||
		!slices.Equal(is.Labels, []string{"bug"}) || len(is.Comments) != 2 || is.Comments[1].Body != "second" ||
		is.URL != "https://github.com/acme/web/issues/7" {
		t.Fatalf("issue = %+v", is)
	}
	if _, err := host.ReadIssue(ctx, "acme/web", 8); !errors.Is(err, github.ErrNotFound) {
		t.Fatalf("missing issue: %v", err)
	}

	req := agent.PRRequest{Repo: "acme/web", Head: "agent/test", Base: "main", Title: "Fix the build", Body: "Fixes #7"}
	pr, err := host.OpenPR(ctx, req)
	if err != nil {
		t.Fatalf("OpenPR: %v", err)
	}
	if pr.Number != 8 || pr.Head != "agent/test" || pr.Base != "main" || pr.URL != "https://github.com/acme/web/pull/8" {
		t.Fatalf("pr = %+v", pr)
	}
	again, err := host.OpenPR(ctx, req)
	if err != nil || again != pr {
		t.Fatalf("second OpenPR = %+v, %v; want the same PR", again, err)
	}
	prs := gh.prs()
	tok, _ := ts.Token(ctx)
	if len(prs) != 1 || prs[0].Title != req.Title || prs[0].Body != req.Body || prs[0].Token != tok.Value {
		t.Fatalf("prs = %+v", prs)
	}
	if _, err := host.OpenPR(ctx, agent.PRRequest{Repo: "acme/web", Head: "agent/missing", Base: "main", Title: "x"}); err == nil {
		t.Fatal("OpenPR from a branch that does not exist succeeded")
	}

	// The sandbox token can neither read issues nor open PRs.
	sbts, err := github.NewTokenSource(app, instID, []string{"acme/web"}, github.SandboxPermissions)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := github.NewHost(sbts).ReadIssue(ctx, "acme/web", 7); err == nil {
		t.Fatal("ReadIssue with the sandbox token succeeded")
	}
	if _, err := github.NewHost(sbts).OpenPR(ctx, agent.PRRequest{Repo: "acme/web", Head: "agent/other", Base: "main", Title: "x"}); err == nil {
		t.Fatal("OpenPR with the sandbox token succeeded")
	}

	// Mint checks: repos outside the installation, unknown installations.
	if _, err := app.InstallationToken(ctx, instID, []string{"acme/nope"}, github.SandboxPermissions); err == nil {
		t.Fatal("minted a token for a repo outside the installation")
	}
	if _, err := app.InstallationToken(ctx, instID+1, []string{"acme/web"}, github.SandboxPermissions); !errors.Is(err, github.ErrNotFound) {
		t.Fatalf("mint for an unknown installation: %v", err)
	}
	if m := gh.mints(); len(m) != 2 || m[0].Permissions["pull_requests"] != "write" || m[1].Permissions["contents"] != "read" ||
		len(m[1].Permissions) != 1 {
		t.Fatalf("mints = %+v", m)
	}

	if err := ts.Revoke(ctx); err != nil {
		t.Fatal(err)
	}
	if gh.isLive(tok.Value) || !slices.Equal(gh.revoked(), []string{tok.Value}) {
		t.Fatalf("revoked = %v, live = %v", gh.revoked(), gh.isLive(tok.Value))
	}
	if u := gh.unexpectedCalls(); len(u) > 0 {
		t.Fatalf("unexpected GitHub calls: %v", u)
	}
}

// TestHarnessLogin checks newEnv: sign-in through the real OAuth flow,
// a session created and read back over the API, its run_session task
// on the e2e Redis database, and the events WebSocket.
func TestHarnessLogin(t *testing.T) {
	e := newEnv(t)
	c := e.login(t, 4242, "octo")
	u := e.user(t, 4242)
	if u.GitHubLogin != "octo" {
		t.Fatalf("user = %+v", u)
	}
	org := e.org(t, u.ID)
	repo := e.seedRepo(t, org.ID, "octo/web", "main")

	e.apiJSON(t, c, http.MethodGet, "/api/me", nil, http.StatusOK)
	e.apiJSON(t, http.DefaultClient, http.MethodGet, "/api/me", nil, http.StatusUnauthorized)

	b := e.apiJSON(t, c, http.MethodPost, "/api/orgs/"+org.ID+"/sessions",
		map[string]any{"prompt": "x", "repo_ids": []string{repo.ID}}, http.StatusCreated)
	id := decodeJSON[struct {
		ID string `json:"id"`
	}](t, b).ID
	s := e.session(t, c, id)
	if s.Status != events.StatusQueued || s.Prompt != "x" || s.OrgID != org.ID ||
		len(s.Repos) != 1 || s.Repos[0].FullName != "octo/web" || s.Repos[0].RepoID != repo.ID {
		t.Fatalf("session = %+v", s)
	}
	if s := e.waitStatus(t, c, id, time.Second, events.StatusQueued); s.ID != id {
		t.Fatalf("waitStatus = %+v", s)
	}

	ti, err := asynq.NewInspectorFromRedisClient(e.Redis).GetTaskInfo(queue.QueueSessions, queue.RunSessionTaskID(id))
	if err != nil {
		t.Fatalf("run_session task on Redis database %d: %v", e2eRedisDB(t), err)
	}
	if ti.State != asynq.TaskStatePending {
		t.Fatalf("run_session task state = %v", ti.State)
	}

	ws := e.events(t, c, id, 0)
	if ev, err := ws.next(300 * time.Millisecond); err != errWSTimeout {
		t.Fatalf("events of a session no worker ran: %+v, %v", ev, err)
	}
	ws.close()
}
