package e2e

import (
	"context"
	"encoding/json"
	"net/http"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/worker"
)

// The e2e secret: an org secret the sandbox gets as $API_TOKEN, which
// must never show up in a stored event.
const e2eSecretValue = "e2e-Secret-Value-7f3a9c"

// The seeded repo: greet.sh forgets the exclamation mark test.sh wants.
var webFiles = map[string]string{
	"README.md": "# web\n\nGreets people.\n",
	"greet.sh":  "#!/bin/sh\necho \"Hello, $1\"\n",
	"test.sh":   "#!/bin/sh\nout=$(sh greet.sh World)\nif [ \"$out\" = \"Hello, World!\" ]; then echo PASS; else echo \"FAIL: got '$out'\"; exit 1; fi\n",
}

// otherSessionBranch is another session's branch in the fixture repo,
// which this session must not be able to touch.
const otherSessionBranch = "agent/00000000-0000-4000-8000-0000000000ff"

// sandboxPushes tries, from a shell in the sandbox, to push a new commit
// to main, to another session's branch and to its own, to delete that
// other branch and the tag, and to force-push. Each push would change a
// ref. It prints ALL-PUSHES-REFUSED and exits 0 only if every push
// failed.
func sandboxPushes(ownBranch string) string {
	var refspecs []string
	for _, r := range []string{
		"$c:refs/heads/main", "+$c:refs/heads/main", "$c:refs/heads/" + otherSessionBranch,
		"+$c:refs/heads/" + otherSessionBranch, ":refs/heads/" + otherSessionBranch, ":refs/tags/v1",
		"$c:refs/heads/" + ownBranch, "$c:refs/heads/new-branch",
	} {
		refspecs = append(refspecs, `"`+r+`"`)
	}
	return `cd /workspace/web && export GIT_TERMINAL_PROMPT=0 && ` +
		`c=$(git -c user.name=x -c user.email=x@example.test commit-tree "HEAD^{tree}" -p HEAD -m injected) && ` +
		"for r in " + strings.Join(refspecs, " ") +
		`; do if git push -q origin "$r" >/dev/null 2>&1; then echo "PUSHED $r"; exit 1; fi; done; echo ALL-PUSHES-REFUSED`
}

// scripted is the session's scripted model: a provider shared by every
// worker task of a test, with a step queue the test fills.
type scripted struct {
	*llmtest.Provider
	mu    sync.Mutex
	calls int
}

func (s *scripted) Generate(ctx context.Context, req llm.Request, on func(llm.StreamEvent)) (*llm.Response, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return s.Provider.Generate(ctx, req, on)
}

func (s *scripted) Calls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// startWorker runs the real worker handlers on the e2e Redis, the way
// `keith worker` does, until the test ends.
func startWorker(t *testing.T, e *env, model llm.Provider) *worker.Worker {
	t.Helper()
	opts := worker.OptionsFromConfig(e.Config)
	// The worker pushes from the host, where host.docker.internal (the
	// sandboxes' name for the git server) may not resolve.
	opts.GitPushBaseURL = e.Git.LocalURL()
	opts.CancelPoll = 300 * time.Millisecond
	opts.LeaseTTL = 10 * time.Second
	w, err := worker.New(worker.Deps{
		Store:     e.Store,
		Events:    e.Store.Events(),
		Bus:       e.Bus,
		Leases:    worker.NewRedisLeaser(lease.NewManager(e.Redis)),
		Control:   e.Control,
		Sandboxes: e.Sandboxes,
		GitHub:    worker.NewAppGitHub(e.App),
		LLM:       func(string) (llm.Provider, error) { return model, nil },
		Keyring:   e.Keyring,
		Logger:    e.Logger.With("component", "worker"),
	}, opts)
	if err != nil {
		t.Fatal(err)
	}
	srv := queue.NewServer(e.Redis, queue.ServerConfig{
		Concurrency: 2, Queues: map[string]int{queue.QueueSessions: 1},
		Logger: e.Logger.With("component", "asynq"), ShutdownTimeout: 20 * time.Second,
	})
	if err := srv.Start(queue.NewMux(w.Handlers())); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Shutdown)
	return w
}

// fixture signs a user in, seeds acme/web (git repo with main, another
// session's branch and a tag; App installation, issue #1, profile and an
// org secret) and returns the client and IDs.
func fixture(t *testing.T, e *env) (*http.Client, store.Org, store.Repo) {
	t.Helper()
	c := e.login(t, 1001, "octo")
	org := e.org(t, e.user(t, 1001).ID)
	repo := e.seedRepo(t, org.ID, "acme/web", "main")
	e.Git.seed(t, "acme/web", "main", webFiles)
	bare := e.Git.bareDir("acme/web")
	e.Git.mustGit(t, bare, "branch", otherSessionBranch, "main")
	e.Git.mustGit(t, bare, "tag", "v1", "main")
	e.GitHub.addIssue("acme/web", 1, ghIssue{
		Title: "greet.sh forgets the exclamation mark",
		Body:  "`sh greet.sh World` prints `Hello, World` but should print `Hello, World!`.",
	})
	e.apiJSON(t, c, http.MethodPut, "/api/repos/"+repo.ID+"/profile", map[string]string{
		"install_cmd": "echo installing deps", "build_cmd": "", "test_cmd": "sh test.sh", "run_cmd": "",
		"knowledge": "Plain POSIX sh.",
	}, http.StatusOK)
	e.apiJSON(t, c, http.MethodPut, "/api/orgs/"+org.ID+"/secrets/API_TOKEN",
		map[string]any{"value": e2eSecretValue}, http.StatusOK)
	return c, org, repo
}

func newSession(t *testing.T, e *env, c *http.Client, org store.Org, repo store.Repo, prompt string) string {
	t.Helper()
	b := e.apiJSON(t, c, http.MethodPost, "/api/orgs/"+org.ID+"/sessions", map[string]any{
		"prompt": prompt, "repo_ids": []string{repo.ID}, "issue_url": "https://github.com/acme/web/issues/1",
	}, http.StatusCreated)
	return decodeJSON[struct {
		ID string `json:"id"`
	}](t, b).ID
}

// shape is an event type sequence with each run of tool_call.output
// collapsed to one entry (how a command's output is chunked depends on
// timing) and the status of session.status spelled out.
func shape(t *testing.T, evs []events.Event) []string {
	t.Helper()
	var out []string
	for _, ev := range evs {
		s := ev.Type
		switch ev.Type {
		case events.TypeToolCallOutput:
			if len(out) > 0 && out[len(out)-1] == s {
				continue
			}
		case events.TypeSessionStatus:
			var p events.SessionStatusPayload
			_ = json.Unmarshal(ev.Payload, &p)
			s += ":" + string(p.Status)
		case events.TypeToolCallStarted:
			var p events.ToolCallStartedPayload
			_ = json.Unmarshal(ev.Payload, &p)
			s += ":" + p.Tool
		}
		out = append(out, s)
	}
	return out
}

func storedEvents(t *testing.T, e *env, id string) []events.Event {
	t.Helper()
	evs, err := e.Store.Events().After(context.Background(), id, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	return evs
}

// TestEndToEnd drives a whole session through the real API, queue and
// worker, a docker sandbox, a local git server and a fake GitHub, with a
// scripted model: read the issue, explore, fix, ask the user, get the
// answer, run the tests, open the PR.
func TestEndToEnd(t *testing.T) {
	e := newEnv(t)
	c, org, repo := fixture(t, e)
	refsBefore := e.Git.refs(t, "acme/web")
	model := &scripted{Provider: llmtest.New()}
	startWorker(t, e, model)

	model.Push(
		llmtest.ToolCall("issue_1", "read_issue", map[string]any{"number": 1}),
		llmtest.ToolCall("shell_1", "create_shell", map[string]any{"name": "main"}),
		llmtest.ToolCall("exec_1", "shell_exec", map[string]any{"shell": "main",
			"command": `cd /workspace/web && cat greet.sh && echo "token=$API_TOKEN"`}),
		llmtest.ToolCall("edit_1", "edit_file", map[string]any{"path": "/workspace/web/greet.sh",
			"old_string": `Hello, $1"`, "new_string": `Hello, $1!"`}),
		llmtest.ToolCall("ask_1", "ask_user", map[string]any{"question": "Update the README too?", "choices": []string{"yes", "no"}}),
	)

	// A live connection from the start: it sees the first half, then
	// reconnects with after_seq for the rest.
	id := newSession(t, e, c, org, repo, "Fix the greeting")
	live := e.events(t, c, id, 0)
	firstHalf := live.until(t, 3*time.Minute, isStatus(events.StatusWaitingForUser))
	live.close()

	s := e.waitStatus(t, c, id, 30*time.Second, events.StatusWaitingForUser)
	if s.Repos[0].Branch != "agent/"+id {
		t.Fatalf("session repo = %+v", s.Repos[0])
	}
	// The diff endpoint already has the edit.
	diff := e.apiJSON(t, c, http.MethodGet, "/api/sessions/"+id+"/diff", nil, http.StatusOK)
	if !strings.Contains(string(diff), `Hello, $1!`) {
		t.Fatalf("diff while waiting = %s", diff)
	}

	model.Push(
		llmtest.ToolCall("exec_2", "shell_exec", map[string]any{"shell": "main", "command": "sh test.sh"}),
		// A prompt injection's push: the sandbox's token is read-only.
		llmtest.ToolCall("push_1", "shell_exec", map[string]any{"shell": "main", "command": sandboxPushes("agent/" + id),
			"timeout_seconds": 300}),
		llmtest.ToolCall("pr_1", "open_pr", map[string]any{"title": "Add the missing exclamation mark",
			"body": "Fixes #1. `sh test.sh` passes."}),
		llmtest.Text("Fixed greet.sh and opened the pull request."),
	)
	e.apiJSON(t, c, http.MethodPost, "/api/sessions/"+id+"/answers",
		map[string]string{"question_id": "ask_1", "text": "no"}, http.StatusAccepted)

	again := e.events(t, c, id, firstHalf[len(firstHalf)-1].Seq)
	secondHalf := again.until(t, 3*time.Minute, isStatus(events.StatusCompleted))
	again.close()
	final := e.waitStatus(t, c, id, 30*time.Second, events.StatusCompleted)

	// Replay from zero equals what the two connections saw, and has the
	// exact shape of the session.
	replay := e.events(t, c, id, 0)
	all := replay.until(t, 30*time.Second, isStatus(events.StatusCompleted))
	replay.close()
	joined := append(slices.Clone(firstHalf), secondHalf...)
	if len(joined) != len(all) {
		t.Fatalf("live+reconnect saw %d events, replay %d", len(joined), len(all))
	}
	for i := range all {
		if all[i].Seq != joined[i].Seq || all[i].Type != joined[i].Type || string(all[i].Payload) != string(joined[i].Payload) {
			t.Fatalf("event %d differs: live %+v, replay %+v", i, joined[i], all[i])
		}
		if all[i].Seq != int64(i+1) {
			t.Fatalf("seq gap at %d: %d", i, all[i].Seq)
		}
	}
	want := []string{
		"message.user",
		"session.status:provisioning",
		"session.status:setting_up",
		"tool_call.started:setup", "shell.created", "tool_call.output", "tool_call.finished",
		"session.status:running",
		"tool_call.started:read_issue", "tool_call.finished",
		"tool_call.started:create_shell", "shell.created", "tool_call.finished",
		"tool_call.started:shell_exec", "tool_call.output", "tool_call.finished",
		"tool_call.started:edit_file", "file.changed", "tool_call.finished",
		"tool_call.started:ask_user", "tool_call.finished", "question.asked",
		"session.status:waiting_for_user",
		"message.user",
		"session.status:running",
		"tool_call.started:shell_exec", "tool_call.output", "tool_call.finished",
		"tool_call.started:shell_exec", "tool_call.output", "tool_call.finished",
		"tool_call.started:open_pr", "pr.opened", "tool_call.finished",
		"message.assistant",
		"tool_call.started:run_tests", "shell.created", "tool_call.output", "tool_call.finished",
		"session.status:completed",
	}
	if got := shape(t, all); !slices.Equal(got, want) {
		t.Fatalf("event shape:\n got  %q\n want %q", got, want)
	}

	// The answer's message.user carries the question; the shell output
	// streamed the file and a redacted secret.
	for _, ev := range all {
		switch ev.Type {
		case events.TypeMessageUser:
			var p events.MessageUserPayload
			_ = json.Unmarshal(ev.Payload, &p)
			if p.Text == "no" && p.QuestionID != "ask_1" {
				t.Errorf("answer message.user = %+v", p)
			}
		case events.TypeFileChanged:
			var p events.FileChangedPayload
			_ = json.Unmarshal(ev.Payload, &p)
			if p.Repo != "acme/web" || p.Path != "/workspace/web/greet.sh" || p.Action != events.FileModified {
				t.Errorf("file.changed = %+v", p)
			}
		}
	}
	var out, pushOut strings.Builder
	for _, ev := range all {
		if ev.Type == events.TypeToolCallOutput {
			var p events.ToolCallOutputPayload
			_ = json.Unmarshal(ev.Payload, &p)
			out.WriteString(p.Chunk)
			if p.CallID == "push_1" {
				pushOut.WriteString(p.Chunk)
			}
		}
	}
	// Every push from the sandbox failed.
	for _, ev := range all {
		if ev.Type != events.TypeToolCallFinished {
			continue
		}
		var p events.ToolCallFinishedPayload
		_ = json.Unmarshal(ev.Payload, &p)
		if p.CallID == "push_1" && (p.ExitCode != 0 || p.TimedOut || !strings.Contains(pushOut.String(), "ALL-PUSHES-REFUSED")) {
			t.Errorf("a push from the sandbox succeeded: %+v\n%s", p, pushOut.String())
		}
	}
	if !strings.Contains(out.String(), "token=[redacted]") || !strings.Contains(out.String(), "PASS") {
		t.Errorf("shell output lacks the redacted token or PASS:\n%s", out.String())
	}

	// No secret, and no token, in any stored event.
	stored := storedEvents(t, e, id)
	secretsSeen := []string{e2eSecretValue}
	for _, m := range e.GitHub.mints() {
		secretsSeen = append(secretsSeen, m.Token)
	}
	for _, ev := range stored {
		for _, v := range secretsSeen {
			if v != "" && strings.Contains(string(ev.Payload), v) {
				t.Errorf("event %d (%s) leaks %q", ev.Seq, ev.Type, v)
			}
		}
	}

	// The worker pushed the branch, which holds the fix; the PR was
	// opened by the worker's own token. No other ref changed.
	branch := "agent/" + id
	refsAfter := e.Git.refs(t, "acme/web")
	wantRefs := append(slices.Clone(refsBefore), "refs/heads/"+branch+" "+e.Git.revParse(t, "acme/web", branch))
	slices.Sort(wantRefs)
	if !slices.Equal(refsAfter, wantRefs) {
		t.Errorf("refs after the session:\n%s\nwant\n%s", strings.Join(refsAfter, "\n"), strings.Join(wantRefs, "\n"))
	}
	if !slices.Contains(e.Git.branches(t, "acme/web"), branch) {
		t.Fatalf("branches = %v", e.Git.branches(t, "acme/web"))
	}
	if body, ok := e.Git.show(t, "acme/web", branch, "greet.sh"); !ok || !strings.Contains(body, `Hello, $1!`) {
		t.Fatalf("greet.sh on %s = %q", branch, body)
	}
	if subj := e.Git.log(t, "acme/web", branch); len(subj) == 0 || subj[0] != "Add the missing exclamation mark" {
		t.Errorf("log = %v", subj)
	}
	prs := e.GitHub.prs()
	if len(prs) != 1 || prs[0].Head != branch || prs[0].Base != "main" || prs[0].Title != "Add the missing exclamation mark" {
		t.Fatalf("PRs = %+v", prs)
	}
	for _, m := range e.GitHub.mints() {
		if m.Token == prs[0].Token && (m.Permissions["pull_requests"] != "write" || m.Permissions["contents"] == "write") {
			t.Errorf("PR opened with token permissions %v; want the worker token", m.Permissions)
		}
	}
	if final.Repos[0].PRNumber == nil || *final.Repos[0].PRNumber != prs[0].Number || final.FinishedAt == nil {
		t.Errorf("final session = %+v", final)
	}
	// Token scoping: the sandbox tokens carry contents:read only, the
	// push tokens contents:write only, the worker's tokens never
	// contents:write; all on acme/web only.
	var sandboxMints, workerMints, pushMints int
	kind := map[string]string{} // token -> sandbox, push or worker
	for _, m := range e.GitHub.mints() {
		if !slices.Equal(m.Repositories, []string{"acme/web"}) {
			t.Errorf("mint scope = %v", m.Repositories)
		}
		switch {
		case len(m.Permissions) == 1 && m.Permissions["contents"] == "read":
			sandboxMints++
			kind[m.Token] = "sandbox"
		case len(m.Permissions) == 1 && m.Permissions["contents"] == "write":
			pushMints++
			kind[m.Token] = "push"
		case m.Permissions["pull_requests"] == "write" && m.Permissions["contents"] == "read":
			workerMints++
			kind[m.Token] = "worker"
		default:
			t.Errorf("unexpected mint permissions %v", m.Permissions)
		}
	}
	if sandboxMints != 2 || workerMints != 2 || pushMints != 1 {
		t.Errorf("mints: %d sandbox, %d worker (want 2 each: run + resume), %d push (want 1: open_pr)", sandboxMints, workerMints, pushMints)
	}
	// The git server: the sandbox's tokens were refused every push; only
	// the push token pushed, from the worker.
	var sandboxPushReqs, pushed int
	for _, r := range e.Git.requests() {
		if !r.Push || r.Token == "" {
			continue
		}
		switch kind[r.Token] {
		case "sandbox":
			sandboxPushReqs++
			if r.Status != http.StatusUnauthorized {
				t.Errorf("a sandbox token's push request got %d: %+v", r.Status, r)
			}
		case "push":
			if r.Status == http.StatusOK && r.Method == http.MethodPost {
				pushed++
			}
		default:
			t.Errorf("push request with a %q token: %+v", kind[r.Token], r)
		}
	}
	if sandboxPushReqs == 0 || pushed != 1 {
		t.Errorf("push requests: %d refused from the sandbox, %d pushes by the worker (want 1)", sandboxPushReqs, pushed)
	}
	// Everything minted is revoked when the session parks, right after
	// the final status.
	deadline := time.Now().Add(15 * time.Second)
	for _, m := range e.GitHub.mints() {
		for e.GitHub.isLive(m.Token) && time.Now().Before(deadline) {
			time.Sleep(50 * time.Millisecond)
		}
		if e.GitHub.isLive(m.Token) {
			t.Errorf("token %s still live", m.Token)
		}
	}
	if u := e.GitHub.unexpectedCalls(); len(u) > 0 {
		t.Errorf("unexpected GitHub calls: %v", u)
	}

	// The diff endpoint returns the change.
	var d struct {
		Repos []struct {
			FullName string `json:"full_name"`
			Diff     string `json:"diff"`
		} `json:"repos"`
	}
	_ = json.Unmarshal(e.apiJSON(t, c, http.MethodGet, "/api/sessions/"+id+"/diff", nil, http.StatusOK), &d)
	if len(d.Repos) != 1 || !strings.Contains(d.Repos[0].Diff, `+echo "Hello, $1!"`) || strings.Contains(d.Repos[0].Diff, e2eSecretValue) {
		t.Errorf("diff = %+v", d)
	}

	// The first setup was snapshotted, without the secret in the image.
	prof, err := e.Store.GetRepoProfile(context.Background(), repo.ID)
	if err != nil || prof.Snapshot.ID == "" || prof.Snapshot.Provider != "docker" {
		t.Fatalf("profile snapshot = %+v, %v", prof.Snapshot, err)
	}
	img, err := exec.Command("docker", "image", "inspect", "--format", "{{json .Config.Env}}", prof.Snapshot.ID).Output()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(img), e2eSecretValue) {
		t.Errorf("snapshot image env holds the secret: %s", img)
	}
	if model.Remaining() != 0 {
		t.Errorf("%d scripted model steps unused", model.Remaining())
	}
}

// TestEndToEndCancel stops a session while a long command runs.
func TestEndToEndCancel(t *testing.T) {
	e := newEnv(t)
	c, org, repo := fixture(t, e)
	model := &scripted{Provider: llmtest.New()}
	startWorker(t, e, model)
	model.Push(llmtest.ToolCall("exec_1", "shell_exec", map[string]any{"shell": "main", "command": "sleep 300", "timeout_seconds": 600}))

	id := newSession(t, e, c, org, repo, "Wait a long time")
	ws := e.events(t, c, id, 0)
	ws.until(t, 3*time.Minute, func(ev events.Event) bool {
		if ev.Type != events.TypeToolCallStarted {
			return false
		}
		var p events.ToolCallStartedPayload
		_ = json.Unmarshal(ev.Payload, &p)
		return p.Tool == "shell_exec"
	})
	time.Sleep(time.Second) // let the command start
	start := time.Now()
	e.apiJSON(t, c, http.MethodPost, "/api/sessions/"+id+"/stop", nil, http.StatusAccepted)
	rest := ws.until(t, time.Minute, isStatus(events.StatusCancelled))
	ws.close()
	if d := time.Since(start); d > 30*time.Second {
		t.Errorf("cancel took %s", d)
	}
	var finished *events.ToolCallFinishedPayload
	for _, ev := range rest {
		if ev.Type == events.TypeToolCallFinished {
			var p events.ToolCallFinishedPayload
			_ = json.Unmarshal(ev.Payload, &p)
			finished = &p
		}
	}
	if finished == nil || finished.CallID != "exec_1" || finished.TimedOut || finished.ExitCode == 0 {
		t.Errorf("shell_exec finished = %+v", finished)
	}
	s := e.waitStatus(t, c, id, 10*time.Second, events.StatusCancelled)
	if s.CancelRequested {
		t.Error("cancel flag left set")
	}
	if model.Calls() != 1 {
		t.Errorf("model called %d times", model.Calls())
	}
	// Stop on a finished session is a 409.
	e.apiJSON(t, c, http.MethodPost, "/api/sessions/"+id+"/stop", nil, http.StatusConflict)
}
