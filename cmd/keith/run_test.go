package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/worker"
)

const testPAT = "ghp_localruntesttoken0123456789"

func envFrom(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestResolveRun(t *testing.T) {
	good := map[string]string{"GITHUB_TOKEN": testPAT, "ANTHROPIC_API_KEY": "sk-ant-test"}
	base := runOptions{Local: true, Repo: "acme/web", Task: "Fix it"}
	tests := []struct {
		name string
		mod  func(*runOptions)
		env  map[string]string
		want string // error substring; "" means success
	}{
		{"ok", nil, good, ""},
		{"not local", func(o *runOptions) { o.Local = false }, good, "only --local is supported; use the web app or the API"},
		{"no repo", func(o *runOptions) { o.Repo = "" }, good, "--repo is required"},
		{"bad repo", func(o *runOptions) { o.Repo = "acme" }, good, `--repo "acme" is not owner/name`},
		{"bad repo chars", func(o *runOptions) { o.Repo = "acme/we b" }, good, "is not owner/name"},
		{"repo url", func(o *runOptions) { o.Repo = "https://github.com/acme/web" }, good, "is not owner/name"},
		{"no task", func(o *runOptions) { o.Task = "" }, good, "give the task as arguments, or an --issue"},
		{"bad issue", func(o *runOptions) { o.Issue = "https://gitlab.com/acme/web/issues/1" }, good, "--issue"},
		{"issue elsewhere", func(o *runOptions) { o.Issue = "https://github.com/acme/api/issues/3" }, good, "--issue is in acme/api, not in --repo acme/web"},
		{"issue only", func(o *runOptions) { o.Task = ""; o.Issue = "https://github.com/ACME/Web/issues/3" }, good, ""},
		{"no token", nil, map[string]string{"ANTHROPIC_API_KEY": "k"}, "GITHUB_TOKEN is not set"},
		{"no anthropic key", nil, map[string]string{"GITHUB_TOKEN": testPAT}, "ANTHROPIC_API_KEY is not set"},
		{"no gemini key", func(o *runOptions) { o.Provider = "gemini" }, good, "GEMINI_API_KEY is not set"},
		{"bad provider", func(o *runOptions) { o.Provider = "openai" }, good, `--provider "openai"`},
		{"gemini without model", func(o *runOptions) { o.Provider = "gemini" },
			map[string]string{"GITHUB_TOKEN": testPAT, "GEMINI_API_KEY": "g"}, "no default model for gemini"},
		{"malformed env", nil, map[string]string{"GITHUB_TOKEN": testPAT, "ANTHROPIC_API_KEY": "k", "MAX_SESSION_DURATION": "soon"}, "MAX_SESSION_DURATION"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := base
			if tt.mod != nil {
				tt.mod(&o)
			}
			plan, cfg, token, err := resolveRun(o, envFrom(tt.env))
			if tt.want != "" {
				if err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("err = %v, want it to contain %q", err, tt.want)
				}
				if err != nil && strings.Contains(err.Error(), testPAT) {
					t.Fatalf("error leaks the token: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if cfg == nil || token != testPAT {
				t.Fatalf("cfg %v, token %q", cfg, token)
			}
			if plan.Provider != "anthropic" || plan.Model != "claude-opus-5-5" || plan.Options.SandboxLabels[localLabel] != "1" ||
				plan.Options.EgressAllowlist != nil || plan.Options.NoPR {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}

	// Flags override the environment's defaults.
	plan, _, _, err := resolveRun(runOptions{Local: true, Repo: "acme/web", Task: "x", Provider: "gemini", Model: "gemini-x", NoPR: true, Keep: true, Install: " npm ci ", Test: "npm test"},
		envFrom(map[string]string{"GITHUB_TOKEN": testPAT, "GEMINI_API_KEY": "g", "SANDBOX_PROVIDER": "daytona", "SANDBOX_TEMPLATE": "img:1"}))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Provider != "gemini" || plan.Model != "gemini-x" || !plan.Options.NoPR || !plan.Keep || plan.Install != "npm ci" ||
		plan.Test != "npm test" || plan.Options.SandboxTemplate != "img:1" || plan.Options.EgressAllowlist != nil {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestRunCommandRequiresLocal(t *testing.T) {
	root := newRootCmd()
	root.SetArgs([]string{"run", "--repo", "acme/web", "fix", "it"})
	root.SetOut(io.Discard)
	root.SetErr(io.Discard)
	err := root.Execute()
	if err == nil || !strings.Contains(err.Error(), "only --local is supported") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseAnswer(t *testing.T) {
	choices := []string{"dayjs", "luxon"}
	for _, c := range []struct {
		in      string
		choices []string
		want    string
		ok      bool
	}{
		{"1", choices, "dayjs", true},
		{" 2 \n", choices, "luxon", true},
		{"3", choices, "3", true},
		{"0", choices, "0", true},
		{"02x", choices, "02x", true},
		{"use moment instead", choices, "use moment instead", true},
		{"1", nil, "1", true},
		{"   ", choices, "", false},
		{"", nil, "", false},
	} {
		got, ok := parseAnswer(c.in, c.choices)
		if got != c.want || ok != c.ok {
			t.Errorf("parseAnswer(%q) = %q, %v; want %q, %v", c.in, got, ok, c.want, c.ok)
		}
	}
}

func ev(seq int64, typ string, payload any) events.Event {
	raw, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	return events.Event{SessionID: "s", Seq: seq, TS: time.Unix(0, 0).UTC(), Type: typ, Payload: raw}
}

func TestEventPrinter(t *testing.T) {
	var out bytes.Buffer
	p := newEventPrinter(&out, nil, false)
	for _, e := range []events.Event{
		ev(1, events.TypeMessageUser, events.MessageUserPayload{MessageID: "u1", Text: "Fix the bug\nin utils"}),
		ev(2, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusProvisioning}),
		ev(3, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c0", Tool: "setup", Input: json.RawMessage(`{"repo":"acme/web","command":"npm ci"}`)}),
		ev(4, events.TypeShellCreated, events.ShellCreatedPayload{ShellID: "setup", CallID: "c0"}),
		ev(5, events.TypeToolCallOutput, events.ToolCallOutputPayload{CallID: "c0", ShellID: "setup", Chunk: "added 3 \x1b[32mpackages\x1b[0m"}),
		ev(6, events.TypeToolCallFinished, events.ToolCallFinishedPayload{CallID: "c0", Tool: "setup", DurationMS: 1234, Summary: "acme/web: ok"}),
		ev(7, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning}),
		ev(0, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m1", Text: "I'll fix "}),
		ev(0, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m1", Text: "it."}),
		ev(8, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m1", Text: "I'll fix it.\n\nDone"}),
		ev(0, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m1", Text: "late delta"}),
		ev(9, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c1", Tool: "shell_exec", Input: json.RawMessage(`{"shell":"main","command":"go test ./...\nmore"}`)}),
		ev(10, events.TypeToolCallOutput, events.ToolCallOutputPayload{CallID: "c1", ShellID: "main", Chunk: "FAIL\n"}),
		ev(11, events.TypeToolCallFinished, events.ToolCallFinishedPayload{CallID: "c1", Tool: "shell_exec", IsError: true, ExitCode: 1, DurationMS: 2500, Summary: "exit 1"}),
		ev(12, events.TypeFileChanged, events.FileChangedPayload{Repo: "acme/web", Path: "/workspace/web/a.go", Action: events.FileModified}),
		ev(13, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c2", Tool: "read_issue", Input: json.RawMessage(`{"repo":"acme/web","number":42}`)}),
		ev(14, events.TypeToolCallFinished, events.ToolCallFinishedPayload{CallID: "c2", Tool: "read_issue", DurationMS: 65000, TimedOut: true}),
		ev(15, events.TypeQuestionAsked, events.QuestionAskedPayload{QuestionID: "q1", Question: "Which lib?", Choices: []string{"dayjs", "luxon"}}),
		ev(16, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusWaitingForUser}),
		ev(16, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusWaitingForUser}), // a replayed duplicate
		ev(17, events.TypePROpened, events.PROpenedPayload{Repo: "acme/web", Number: 7, URL: "https://github.com/acme/web/pull/7"}),
		ev(18, events.TypeError, events.ErrorPayload{Code: "ship_failed", Message: "boom"}),
		ev(19, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusFailed, Reason: "ship_failed"}),
		ev(20, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m2", Text: "Full text without deltas"}),
		ev(0, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m3", Text: "partial "}),
		ev(21, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m3", Text: "different final text"}),
		ev(22, "x.custom", map[string]int{"a": 1}),
		ev(23, events.TypeError, events.ErrorPayload{Message: "no code"}),
	} {
		p.Print(e)
	}
	want := strings.Join([]string{
		"› Fix the bug",
		"  in utils",
		"● provisioning",
		"▶ setup  npm ci",
		"added 3 \x1b[32mpackages\x1b[0m",
		"✓ setup (1.2s) acme/web: ok",
		"● running",
		"I'll fix it.",
		"",
		"Done",
		"▶ shell_exec  go test ./... …",
		"FAIL",
		"✗ shell_exec (exit 1, 2.5s) exit 1",
		"✎ modified /workspace/web/a.go",
		"▶ read_issue  acme/web#42",
		"✗ read_issue (timed out, 1m5s)",
		"? Which lib?",
		"  1) dayjs",
		"  2) luxon",
		"● waiting_for_user",
		"⇪ pull request acme/web#7 opened: https://github.com/acme/web/pull/7",
		"✗ error (ship_failed): boom",
		"● failed (ship_failed)",
		"Full text without deltas",
		"partial ",
		"different final text",
		`· x.custom {"a":1}`,
		"✗ error: no code",
		"",
	}, "\n")
	if got := out.String(); got != want {
		t.Errorf("output:\n%s\nwant:\n%s", got, want)
	}
	if p.Last() != 23 || p.Status() != events.StatusFailed {
		t.Errorf("last %d, status %s", p.Last(), p.Status())
	}
}

func TestFailureHint(t *testing.T) {
	for _, c := range []struct{ reason, msg, want string }{
		{"provision_failed", `Could not create the sandbox: docker: create sandbox from keith-sandbox:latest: sandbox: not found (docker run: exit 125: Unable to find image 'keith-sandbox:latest' locally)`,
			"docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox"},
		{"provision_failed", `Could not create the sandbox: docker: create sandbox from my:img: pull access denied for my`, "SANDBOX_TEMPLATE"},
		{"provision_failed", `docker run: exec: "docker": executable file not found in $PATH`, "not installed"},
		{"provision_failed", `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?`, "start Docker"},
		{"clone_failed", "Could not clone acme/web: exit 128", "GITHUB_TOKEN can read it"},
		{"ship_failed", "Could not push", "--no-pr"},
		{"llm_unavailable", "no key", "API key"},
		{"step_cap", "too many steps", ""},
	} {
		got := failureHint(c.reason, c.msg, nil)
		if (c.want == "" && got != "") || !strings.Contains(got, c.want) {
			t.Errorf("failureHint(%s, %q) = %q, want %q", c.reason, c.msg, got, c.want)
		}
	}
}

// ---- whole local sessions, in-process ----

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// snapProvider is a memsandbox.Provider that can fail Create.
type snapProvider struct {
	*memsandbox.Provider
	createErr error
}

func (p *snapProvider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Sandbox, error) {
	if p.createErr != nil {
		return nil, p.createErr
	}
	return p.Provider.Create(ctx, spec)
}

// localHandler fakes the sandbox's git and setup commands.
func localHandler(spec sandbox.Spec) memsandbox.Handler {
	branch := "agent/" + spec.Labels["keith.session"]
	return func(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		ok := func(s string) (*sandbox.ExecResult, error) { return &sandbox.ExecResult{Stdout: s}, nil }
		c := req.Cmd
		switch {
		case c == "id -u; id -g":
			return ok("1000\n1000\n")
		case strings.Contains(c, "git clone"):
			return ok("default=main\nstart=origin/main\n")
		case strings.Contains(c, "git diff --cached"):
			return ok("diff --git a/hello.txt b/hello.txt\n+hello\n")
		case strings.Contains(c, "git status --porcelain"):
			return ok("1\n0\n0\n") // dirty, nothing committed
		case strings.Contains(c, "git symbolic-ref"):
			return ok("origin/main\n")
		case c == "git rev-parse --abbrev-ref HEAD":
			return ok(branch + "\n")
		}
		return ok("")
	}
}

type testShells struct{ *agenttest.Shells }

func (testShells) Reattach(context.Context) ([]string, error) { return nil, nil }
func (testShells) Close(context.Context, string) error        { return nil }

func newTestShells(_ sandbox.Sandbox, _ string, onOutput func(string, []byte)) worker.Shells {
	return testShells{&agenttest.Shells{OnOutput: onOutput, OnExec: func(string, string, time.Duration) (sandbox.ShellResult, error) {
		return sandbox.ShellResult{OutputTail: "ok\n"}, nil
	}}}
}

// fakeGitHubAPI is the REST API the PAT host talks to.
type fakeGitHubAPI struct {
	srv *httptest.Server

	mu    sync.Mutex
	auth  []string
	paths []string
	prs   int
}

func newFakeGitHubAPI(t *testing.T) *fakeGitHubAPI {
	f := &fakeGitHubAPI{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /repos/acme/web/pulls", func(w http.ResponseWriter, r *http.Request) {
		f.record(r)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, "[]")
	})
	mux.HandleFunc("POST /repos/acme/web/pulls", func(w http.ResponseWriter, r *http.Request) {
		f.record(r)
		var body struct{ Head, Base string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.prs++
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]any{
			"number": 7, "html_url": "https://github.com/acme/web/pull/7",
			"head": map[string]any{"ref": body.Head}, "base": map[string]any{"ref": body.Base},
		})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		f.record(r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"message":"Not Found"}`)
	})
	f.srv = httptest.NewServer(mux)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeGitHubAPI) record(r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	f.paths = append(f.paths, r.Method+" "+r.URL.Path)
}

type localFixture struct {
	t    *testing.T
	sbx  *snapProvider
	llm  *llmtest.Provider
	gh   *fakeGitHubAPI
	plan runPlan
	deps localDeps
	run  *localRun
	out  *syncBuffer
}

// newFixture resolves o (on acme/web) the way the command does, and
// wires memsandbox, llmtest, fake shells and the PAT GitHub against a
// fake API.
func newFixture(t *testing.T, o runOptions) *localFixture {
	t.Helper()
	o.Local, o.Repo = true, "acme/web"
	if o.Task == "" && o.Issue == "" {
		o.Task = "Add hello"
	}
	if o.Model == "" {
		o.Model = "test-model"
	}
	plan, _, token, err := resolveRun(o, envFrom(map[string]string{"GITHUB_TOKEN": testPAT, "ANTHROPIC_API_KEY": "sk-ant-test"}))
	if err != nil {
		t.Fatal(err)
	}
	plan.Options.CancelPoll = 20 * time.Millisecond
	f := &localFixture{t: t, sbx: &snapProvider{Provider: memsandbox.NewProvider()}, llm: llmtest.New(), gh: newFakeGitHubAPI(t), plan: plan, out: &syncBuffer{}}
	f.sbx.NewHandler = localHandler
	f.deps = localDeps{
		Sandboxes: f.sbx,
		LLM: func(name string) (llm.Provider, error) {
			if name != "anthropic" {
				return nil, errors.New("unexpected provider " + name)
			}
			return f.llm, nil
		},
		GitHub:    worker.NewPATGitHub(token, github.PATHostOptions{APIBaseURL: f.gh.srv.URL + "/", HTTPClient: f.gh.srv.Client()}),
		NewShells: newTestShells,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	return f
}

func (f *localFixture) build() *localRun {
	f.t.Helper()
	if f.run == nil {
		r, err := newLocalRun(f.plan, f.deps)
		if err != nil {
			f.t.Fatal(err)
		}
		f.run = r
	}
	return f.run
}

func (f *localFixture) start(stdin string, interrupts <-chan os.Signal) error {
	f.t.Helper()
	r := f.build()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return r.Run(ctx, strings.NewReader(stdin), f.out, nil, interrupts)
}

func (f *localFixture) sandbox() *memsandbox.Sandbox {
	f.t.Helper()
	id := f.run.ms.Session(f.run.sess.ID).SandboxID
	sb, ok := f.sbx.Sandbox(id)
	if !ok {
		f.t.Fatalf("no sandbox %q", id)
	}
	return sb
}

func (f *localFixture) wantOutput(wants ...string) {
	f.t.Helper()
	out := f.out.String()
	for _, w := range wants {
		if !strings.Contains(out, w) {
			f.t.Errorf("output lacks %q:\n%s", w, out)
		}
	}
}

func writeHello(id string) *llm.Response {
	return llmtest.ToolCall(id, "write_file", map[string]any{"path": "/workspace/web/hello.txt", "content": "hello\n"})
}

func TestLocalRunShipsAndPrintsDiff(t *testing.T) {
	f := newFixture(t, runOptions{Install: "npm ci", Test: "npm test"})
	f.llm.Push(writeHello("c1"), llmtest.Text("Added hello.txt."))
	if err := f.start("", nil); err != nil {
		t.Fatalf("run: %v\n%s", err, f.out)
	}
	id := f.run.sess.ID
	f.wantOutput(
		"Session "+id+" on acme/web (anthropic test-model, pull request at the end)",
		"› Add hello",
		"● provisioning\n● setting_up\n▶ setup  npm ci\nok\n✓ setup (",
		"● running",
		"▶ write_file  /workspace/web/hello.txt",
		"✎ created /workspace/web/hello.txt",
		"▶ run_tests  npm test",
		"⇪ pull request acme/web#7 opened: https://github.com/acme/web/pull/7",
		"● completed",
		"=== Session completed ===\nPull request for acme/web: https://github.com/acme/web/pull/7\n",
		"--- diff acme/web ---\ndiff --git a/hello.txt b/hello.txt\n+hello\n",
	)
	out := f.out.String()
	if n := strings.Count(out, "Added hello.txt."); n != 1 {
		t.Errorf("final message printed %d times:\n%s", n, out)
	}
	if strings.Contains(out, testPAT) {
		t.Error("output leaks the GitHub token")
	}

	sess := f.run.ms.Session(id)
	if sess.Status != events.StatusCompleted || sess.Repos[0].DefaultBranch != "" || sess.Repos[0].Branch != "agent/"+id {
		t.Errorf("session = %+v", sess)
	}
	sb := f.sandbox()
	if !sb.Deleted() {
		t.Error("sandbox not deleted")
	}
	tokenWritten := false
	for _, c := range sb.Calls() {
		if c.Op == memsandbox.OpWrite && c.Path == "/run/keith/git-token.tmp" && string(c.Data) == testPAT+"\n" {
			tokenWritten = true
		}
	}
	if !tokenWritten {
		t.Error("the PAT was not written to the credential helper's token file")
	}
	if specs := f.sbx.Specs(); len(specs) != 1 || specs[0].Labels[localLabel] != "1" || specs[0].Labels["keith.session"] != id {
		t.Errorf("specs = %+v", specs)
	}
	for _, c := range sb.Calls() {
		if c.Op == memsandbox.OpSnapshot {
			t.Error("a local run snapshotted its sandbox")
		}
	}
	f.gh.mu.Lock()
	defer f.gh.mu.Unlock()
	if f.gh.prs != 1 {
		t.Errorf("PRs opened = %d (%v)", f.gh.prs, f.gh.paths)
	}
	for _, a := range f.gh.auth {
		if !strings.Contains(a, testPAT) {
			t.Errorf("GitHub API call without the PAT: %q", a)
		}
	}
}

func lastToolResult(req llm.Request) string {
	return agenttest.BlockText(agenttest.ToolResults(req)...)
}

func TestLocalRunAsksForAnswers(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true, Keep: true})
	f.llm.Push(llmtest.ToolCall("q1", "ask_user", map[string]any{"question": "Which date library?", "choices": []string{"dayjs", "luxon"}}))
	f.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		if res := lastToolResult(req); !strings.Contains(res, "luxon") {
			t.Errorf("the answer is not the tool result: %q", res)
		}
		return llmtest.Text("Using luxon."), nil
	})
	if err := f.start("\n2\n", nil); err != nil {
		t.Fatalf("run: %v\n%s", err, f.out)
	}
	prompt := "answer (1-2 or text, Ctrl-D to stop)> "
	f.wantOutput(
		"(anthropic test-model, no pull request)",
		"? Which date library?\n  1) dayjs\n  2) luxon\n● waiting_for_user\n",
		prompt+"\n"+prompt+"2\n",
		"› luxon",
		"Using luxon.",
		"=== Session completed ===",
		"Kept the sandbox mem-p1 (paused).",
	)
	id := f.run.sess.ID
	ins := f.run.ms.Inputs(id)
	if len(ins) != 1 || ins[0].Text != "luxon" || ins[0].QuestionID != "q1" || ins[0].ConsumedAt == nil {
		t.Errorf("inputs = %+v", ins)
	}
	if f.sandbox().Deleted() {
		t.Error("--keep: sandbox deleted")
	}
	if f.gh.prs != 0 {
		t.Error("--no-pr: a PR was opened")
	}
}

func TestLocalRunEOFWhileWaitingCancels(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true})
	f.llm.Push(llmtest.ToolCall("q1", "ask_user", map[string]any{"question": "Proceed?"}))
	err := f.start("", nil)
	if err == nil || !strings.Contains(err.Error(), "cancelled") {
		t.Fatalf("err = %v\n%s", err, f.out)
	}
	f.wantOutput("? Proceed?", "answer (Ctrl-D to stop)> \nStopping the session.", "● cancelled (cancelled_by_user)",
		"=== Session cancelled (cancelled_by_user) ===", "--- diff acme/web ---\n(no changes)")
	if s := f.run.ms.Session(f.run.sess.ID); s.Status != events.StatusCancelled {
		t.Errorf("status = %s", s.Status)
	}
	if !f.sandbox().Deleted() {
		t.Error("sandbox not deleted")
	}
}

func TestLocalRunCtrlCStops(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true})
	r := f.build()
	sigs := make(chan os.Signal, 4)
	f.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		sigs <- os.Interrupt
		for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
			if c, _ := r.ms.IsCancelRequested(context.Background(), r.sess.ID); c {
				break
			}
		}
		return writeHello("c1"), nil
	})
	err := f.start("", sigs)
	if err == nil || !strings.Contains(err.Error(), "cancelled") {
		t.Fatalf("err = %v\n%s", err, f.out)
	}
	f.wantOutput("^C Stopping the session (press Ctrl-C again to abort)...", "● cancelled (cancelled_by_user)")
	if p := r.ctl.Published(); len(p) != 1 || p[0].SessionID != r.sess.ID {
		t.Errorf("control signals = %+v", p)
	}
	if !f.sandbox().Deleted() {
		t.Error("sandbox not deleted")
	}
}

// blockingLLM blocks every call until its context ends.
type blockingLLM struct{ started chan struct{} }

func (b *blockingLLM) Name() string { return "anthropic" }

func (b *blockingLLM) Generate(ctx context.Context, _ llm.Request, _ func(llm.StreamEvent)) (*llm.Response, error) {
	select {
	case b.started <- struct{}{}:
	default:
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestLocalRunSecondCtrlCAborts(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true})
	bl := &blockingLLM{started: make(chan struct{}, 1)}
	f.deps.LLM = func(string) (llm.Provider, error) { return bl, nil }
	sigs := make(chan os.Signal, 4)
	go func() {
		<-bl.started
		sigs <- os.Interrupt
		sigs <- os.Interrupt
	}()
	err := f.start("", sigs)
	if err == nil || !strings.Contains(err.Error(), "aborted") {
		t.Fatalf("err = %v\n%s", err, f.out)
	}
	f.wantOutput("^C Stopping the session", "^C Aborting", "=== Session running ===")
	if !f.sandbox().Deleted() {
		t.Error("sandbox not deleted after abort")
	}
}

func TestLocalRunMissingImageHint(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true})
	f.sbx.createErr = errors.New("docker: create sandbox from keith-sandbox:latest: sandbox: not found (docker run: exit 125: " +
		"Unable to find image 'keith-sandbox:latest' locally\ndocker: Error response from daemon: pull access denied for keith-sandbox)")
	err := f.start("", nil)
	if err == nil || !strings.Contains(err.Error(), "failed (provision_failed)") {
		t.Fatalf("err = %v\n%s", err, f.out)
	}
	f.wantOutput("✗ error (provision_failed): Could not create the sandbox",
		"=== Session failed (provision_failed) ===\nHint: build the sandbox image first:\n  docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox\n",
		"(no changes)")
}

func TestLocalRunIssueOnly(t *testing.T) {
	f := newFixture(t, runOptions{NoPR: true, Issue: "https://github.com/acme/web/issues/42", JSON: true})
	f.llm.Push(llmtest.Text("Nothing to do."))
	r := f.build()
	var info syncBuffer
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := r.Run(ctx, strings.NewReader(""), f.out, &info, nil); err != nil {
		t.Fatalf("run: %v\n%s", err, info.String())
	}
	// --json: stdout is JSON lines only, the first one the prompt.
	lines := strings.Split(strings.TrimSpace(f.out.String()), "\n")
	var first events.Event
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil || first.Type != events.TypeMessageUser ||
		!strings.Contains(string(first.Payload), "Resolve https://github.com/acme/web/issues/42") {
		t.Fatalf("first line = %q (%v)", lines[0], err)
	}
	for _, l := range lines {
		if !json.Valid([]byte(l)) {
			t.Errorf("not JSON: %q", l)
		}
	}
	if !strings.Contains(info.String(), "=== Session completed ===") {
		t.Errorf("summary not on the info writer:\n%s", info.String())
	}
	if got := string(r.ms.Session(r.sess.ID).Trigger); got != `{"kind":"cli","issue_url":"https://github.com/acme/web/issues/42"}` {
		t.Errorf("trigger = %s", got)
	}
	f.gh.mu.Lock()
	defer f.gh.mu.Unlock()
	if len(f.gh.paths) == 0 || f.gh.paths[0] != "GET /repos/acme/web/issues/42" {
		t.Errorf("GitHub calls = %v", f.gh.paths)
	}
}
