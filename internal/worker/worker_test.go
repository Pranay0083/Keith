package worker

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/store"
)

func writeHello(id string) *llm.Response {
	return toolCall(id, "write_file", map[string]any{"path": "/workspace/web/hello.txt", "content": "hello\n"})
}

func TestNewSessionRunsSetupLoopVerifyAndShips(t *testing.T) {
	e := newEnv(t)
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci", TestCmd: "npm test"})
	s := e.newSession()
	e.llm.Push(writeHello("c1"))
	e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		e.setChanges(true, 0)
		return llmtest.Text("Added hello.txt."), nil
	})

	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "events", e.types(s.ID), []string{
		events.TypeMessageUser,
		events.TypeSessionStatus,                                                                                    // provisioning
		events.TypeSessionStatus,                                                                                    // setting_up
		events.TypeToolCallStarted, events.TypeShellCreated, events.TypeToolCallOutput, events.TypeToolCallFinished, // setup
		events.TypeSessionStatus,                                                        // running
		events.TypeToolCallStarted, events.TypeFileChanged, events.TypeToolCallFinished, // write_file
		events.TypeMessageAssistant,
		events.TypeToolCallStarted, events.TypeShellCreated, events.TypeToolCallOutput, events.TypeToolCallFinished, // run_tests
		events.TypePROpened,
		events.TypeSessionStatus, // completed
	})
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "completed"})
	first := payload[events.MessageUserPayload](t, e.evs.All(s.ID)[0])
	if first.Text != "Fix the login bug" || first.UserID != testUser || first.MessageID == "" {
		t.Errorf("first message.user = %+v", first)
	}
	setup := payload[events.ToolCallStartedPayload](t, e.events(s.ID, events.TypeToolCallStarted)[0])
	if setup.Tool != "setup" || !strings.Contains(string(setup.Input), "npm ci") {
		t.Errorf("setup call = %+v", setup)
	}
	out := payload[events.ToolCallOutputPayload](t, e.events(s.ID, events.TypeToolCallOutput)[0])
	if out.CallID != setup.CallID || out.ShellID != "setup" {
		t.Errorf("setup output = %+v, want call %s", out, setup.CallID)
	}

	sess := e.st.Session(s.ID)
	if sess.Status != events.StatusCompleted || sess.FinishedAt == nil || sess.OpenQuestionID != "" {
		t.Errorf("session = %+v", sess)
	}
	if r := sess.Repos[0]; r.PRNumber != 100 || r.Branch != "agent/"+s.ID {
		t.Errorf("session repo = %+v", r)
	}
	prs := e.gh.host.PRRequests()
	if len(prs) != 1 || prs[0].Head != "agent/"+s.ID || prs[0].Base != "main" || prs[0].Title != "Fix the login bug" ||
		!strings.Contains(prs[0].Body, "Added hello.txt.") {
		t.Errorf("PR requests = %+v", prs)
	}
	if d := e.st.Diffs(s.ID); len(d) != 1 || !strings.Contains(d[0].Diff, "+hello") {
		t.Errorf("diffs = %+v", d)
	}

	sb := e.sandbox(s.ID)
	if !sb.Paused() {
		t.Error("sandbox not paused after completion")
	}
	if got := writes(sb, "/run/keith/git-token.tmp"); len(got) == 0 || got[0] != firstToken+"\n" {
		t.Errorf("token writes = %q", got)
	}
	if len(e.execs(sb, "rm -f /run/keith/git-token")) == 0 {
		t.Error("token file not deleted at park")
	}
	if !e.gh.lastSandbox().isRevoked() || !e.gh.lastWorker().isRevoked() {
		t.Error("tokens not revoked")
	}
	if e.leases.Held("session:" + s.ID) {
		t.Error("lease still held")
	}
	// Credential helper: root install, then git config for the remote,
	// then clone with a plain URL.
	if r := e.execs(sb, "base64 -d"); len(r) != 1 || r[0].User != "root" || !strings.Contains(r[0].Cmd, "install -d -m 0700 -o 1000 -g 1000 /run/keith") {
		t.Errorf("helper install = %+v", r)
	}
	if r := e.execs(sb, "credential.https://github.com.helper"); len(r) != 1 {
		t.Errorf("git config = %+v", r)
	}
	clone := e.execs(sb, "git clone")
	if len(clone) != 1 || !strings.Contains(clone[0].Cmd, "url='https://github.com/acme/web.git'") ||
		strings.Contains(clone[0].Cmd, firstToken) || clone[0].User != "" {
		t.Errorf("clone = %+v", clone)
	}
	// The snapshot of this first successful setup.
	prof, _ := e.st.GetRepoProfile(context.Background(), testRepo)
	if prof.Snapshot.ID == "" || prof.Snapshot.Provider != memsandbox.Name || !strings.HasPrefix(prof.Snapshot.LockHash, "v1:") {
		t.Errorf("snapshot = %+v", prof.Snapshot)
	}
	if sp := e.sbx.Specs(); len(sp) != 1 || sp[0].Labels["keith.session"] != s.ID || sp[0].SnapshotID != "" {
		t.Errorf("specs = %+v", sp)
	}
	// The system prompt carries the task and the profile.
	reqs := e.llm.Requests()
	if len(reqs) != 2 || !strings.Contains(reqs[0].System, "Fix the login bug") || !strings.Contains(reqs[0].System, "npm test") {
		t.Errorf("requests = %d, system %q", len(reqs), reqs[0].System)
	}
}

func TestIssueOnlySessionReadsIssue(t *testing.T) {
	e := newEnv(t)
	s := e.newSession(withPrompt(""), withIssue("https://github.com/acme/web/issues/7"))
	e.llm.Push(llmtest.Text("Nothing to do."))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	first := payload[events.MessageUserPayload](t, e.evs.All(s.ID)[0])
	if first.Text != "Resolve https://github.com/acme/web/issues/7" {
		t.Errorf("first message = %q", first.Text)
	}
	if sys := e.llm.Requests()[0].System; !strings.Contains(sys, "Login is broken") {
		t.Errorf("system prompt lacks the issue: %q", sys)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "no_changes"})
	if s := e.st.Session(s.ID); s.StatusReason != ReasonNoChanges {
		t.Errorf("reason = %q", s.StatusReason)
	}
	if len(e.gh.host.PRRequests()) != 0 {
		t.Error("a PR was opened without changes")
	}
}

func TestAskUserWaitsThenResumesWithAnswer(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "Which date library?", "choices": []string{"dayjs", "luxon"}}))

	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "waiting_for_user"})
	types := e.types(s.ID)
	if n := len(types); types[n-2] != events.TypeQuestionAsked || types[n-1] != events.TypeSessionStatus {
		t.Fatalf("tail of events = %v", types[n-3:])
	}
	sess := e.st.Session(s.ID)
	if sess.OpenQuestionID != "q1" || sess.Status != events.StatusWaitingForUser {
		t.Fatalf("session = %+v", sess)
	}
	sb := e.sandbox(s.ID)
	if !sb.Paused() || !e.gh.lastSandbox().isRevoked() {
		t.Fatal("waiting session not parked")
	}

	// A wakeup with nothing to do leaves it waiting.
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if got := e.statuses(s.ID); got[len(got)-1] != "waiting_for_user" {
		t.Fatalf("spurious wakeup changed status: %v", got)
	}

	ans, _ := e.st.AddInput(s.ID, store.InputAnswer, "dayjs", "q1", testUser)
	e.setChanges(true, 0)
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		res := agenttestToolResults(req)
		if !strings.Contains(res, "dayjs") {
			t.Errorf("answer not delivered as the tool result: %q", res)
		}
		return llmtest.Text("Using dayjs."), nil
	})
	before := len(e.types(s.ID))
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	after := e.types(s.ID)[before:]
	wantSeq(t, "resume events", after, []string{
		events.TypeMessageUser, events.TypeSessionStatus, events.TypeMessageAssistant,
		events.TypePROpened, events.TypeSessionStatus,
	})
	all := e.evs.All(s.ID)
	mu := payload[events.MessageUserPayload](t, all[before])
	if mu.MessageID != ans.ID || mu.QuestionID != "q1" || mu.Text != "dayjs" {
		t.Errorf("answer message.user = %+v", mu)
	}
	if st := payload[events.SessionStatusPayload](t, all[before+1]); st.Status != events.StatusRunning {
		t.Errorf("after the answer: %+v", st)
	}
	if pending, _ := e.st.ListPendingInputs(context.Background(), s.ID); len(pending) != 0 {
		t.Errorf("answer still pending: %+v", pending)
	}
	if sb.Paused() == false || len(e.sbx.Specs()) != 1 {
		t.Errorf("resume did not reuse and re-park the sandbox: specs %d", len(e.sbx.Specs()))
	}
	if !callsHave(sb, memsandbox.OpResume) {
		t.Error("sandbox not resumed")
	}
	// Fresh token written on resume, from a fresh source.
	if got := writes(sb, "/run/keith/git-token.tmp"); len(got) != 2 || got[1] != "ghs_sandbox2\n" {
		t.Errorf("token writes = %q", got)
	}
	if sh := e.lastShells(); sh.reattached != 1 {
		t.Errorf("reattached = %d", sh.reattached)
	}
	if e.st.Session(s.ID).Status != events.StatusCompleted {
		t.Errorf("final status = %s", e.st.Session(s.ID).Status)
	}
}

func agenttestToolResults(req llm.Request) string {
	var b strings.Builder
	last := req.Messages[len(req.Messages)-1]
	for _, bl := range last.Content {
		if bl.Type == llm.BlockToolResult {
			for _, c := range bl.Content {
				b.WriteString(c.Text)
			}
		}
	}
	return b.String()
}

func callsHave(sb *memsandbox.Sandbox, op memsandbox.Op) bool {
	for _, c := range sb.Calls() {
		if c.Op == op {
			return true
		}
	}
	return false
}

func TestCancelMidCommandInterruptsShells(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	started := make(chan struct{})
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if !strings.Contains(cmd, "sleep 600") {
			return sandbox.ShellResult{}, nil
		}
		close(started)
		sh := e.lastShells()
		select {
		case <-sh.interrupts:
			return sandbox.ShellResult{OutputTail: "^C", ExitCode: 130}, nil
		case <-time.After(10 * time.Second):
			return sandbox.ShellResult{}, errors.New("not interrupted")
		}
	}
	e.llm.Push(toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "sleep 600"}))
	go func() {
		<-started
		_ = e.st.RequestCancel(s.ID)
		_ = e.ctl.Publish(context.Background(), s.ID, "cancel")
	}()
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "cancelled"})
	sess := e.st.Session(s.ID)
	if sess.Status != events.StatusCancelled || sess.CancelRequestedAt != nil || sess.StatusReason != ReasonCancelled {
		t.Errorf("session = %+v", sess)
	}
	if fin := e.events(s.ID, events.TypeToolCallFinished); len(fin) != 1 || payload[events.ToolCallFinishedPayload](t, fin[0]).ExitCode != 130 {
		t.Errorf("finished = %v", fin)
	}
	if !e.sandbox(s.ID).Paused() {
		t.Error("not parked")
	}
	if e.llm.Remaining() != 0 || len(e.llm.Requests()) != 1 {
		t.Errorf("model called after cancel: %d requests", len(e.llm.Requests()))
	}

	// A follow-up message to the cancelled session resumes it (the flag
	// was cleared, so it is not cancelled at once).
	e.st.AddInput(s.ID, store.InputMessage, "carry on", "", testUser)
	e.llm.Push(llmtest.Text("Done."))
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	got := e.statuses(s.ID)
	wantSeq(t, "statuses after follow-up", got[len(got)-2:], []string{"running", "no_changes"})
}

func TestCancelBeforePickupAndWhileWaiting(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	_ = e.st.RequestCancel(s.ID)
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"cancelled"})
	if len(e.sbx.Specs()) != 0 {
		t.Error("a sandbox was created for a cancelled session")
	}

	s2 := e.newSession()
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "?"}))
	if err := e.run(s2.ID); err != nil {
		t.Fatal(err)
	}
	_ = e.st.RequestCancel(s2.ID)
	if err := e.resume(s2.ID); err != nil {
		t.Fatal(err)
	}
	got := e.statuses(s2.ID)
	if got[len(got)-1] != "cancelled" || e.st.Session(s2.ID).OpenQuestionID != "" {
		t.Errorf("statuses = %v, open question %q", got, e.st.Session(s2.ID).OpenQuestionID)
	}
}

// hookStore runs a func after a status write, before the worker
// re-checks: input or a Stop arriving while the worker finishes.
type hookStore struct {
	Store
	after func(status events.SessionStatus)
}

func (h hookStore) SetSessionStatus(ctx context.Context, id string, st events.SessionStatus, reason string) error {
	err := h.Store.SetSessionStatus(ctx, id, st, reason)
	if err == nil && h.after != nil {
		h.after(st)
	}
	return err
}

func TestReCheckAfterTerminalStatusPicksUpLateInput(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	once := false
	e.store = hookStore{Store: e.st, after: func(st events.SessionStatus) {
		if st == events.StatusNoChanges && !once {
			once = true
			e.st.AddInput(s.ID, store.InputMessage, "one more thing", "", testUser)
		}
	}}
	e.llm.Push(llmtest.Text("First."), llmtest.Text("Second."))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "no_changes", "running", "no_changes"})
	if mu := e.events(s.ID, events.TypeMessageUser); len(mu) != 2 || payload[events.MessageUserPayload](t, mu[1]).Text != "one more thing" {
		t.Errorf("message.user events = %d", len(mu))
	}
	if len(e.sbx.Specs()) != 1 {
		t.Errorf("sandboxes = %d", len(e.sbx.Specs()))
	}
}

func TestReCheckAfterWaitingSeesStop(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.store = hookStore{Store: e.st, after: func(st events.SessionStatus) {
		if st == events.StatusWaitingForUser {
			_ = e.st.RequestCancel(s.ID)
		}
	}}
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "?"}))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "waiting_for_user", "cancelled"})
}

func TestReCheckAfterWaitingSeesAnswer(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.store = hookStore{Store: e.st, after: func(st events.SessionStatus) {
		if st == events.StatusWaitingForUser {
			e.st.AddInput(s.ID, store.InputAnswer, "yes", "q1", testUser)
		}
	}}
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "?"}), llmtest.Text("ok"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "waiting_for_user", "running", "no_changes"})
}

func TestFailureOutcomes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		opts   func(*Options)
		script func(*llmtest.Provider)
		reason string
	}{
		{"step cap", func(o *Options) { o.MaxSteps = 1 }, func(p *llmtest.Provider) {
			p.Push(toolCall("c1", "message_user", map[string]any{"text": "working"}))
		}, ReasonStepCap},
		{"budget", func(o *Options) { o.MaxSessionTokens = 10 }, func(p *llmtest.Provider) {
			p.Push(toolCall("c1", "message_user", map[string]any{"text": "working"}))
		}, ReasonBudgetExceeded},
		{"llm error", nil, func(p *llmtest.Provider) {
			p.PushError(&llm.Error{Kind: llm.ErrInvalidRequest, Message: "bad request"})
		}, agent.ReasonLLMError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t)
			if tc.opts != nil {
				tc.opts(&e.opts)
			}
			s := e.newSession()
			tc.script(e.llm)
			if err := e.run(s.ID); err != nil {
				t.Fatal(err)
			}
			got := e.statuses(s.ID)
			if got[len(got)-1] != "failed" || e.st.Session(s.ID).StatusReason != tc.reason {
				t.Fatalf("statuses %v reason %q", got, e.st.Session(s.ID).StatusReason)
			}
			errs := e.events(s.ID, events.TypeError)
			if len(errs) != 1 || payload[events.ErrorPayload](t, errs[0]).Code != tc.reason {
				t.Errorf("error events = %v", errs)
			}
			if !e.sandbox(s.ID).Paused() {
				t.Error("not parked")
			}
		})
	}
}

func TestProvisionFailures(t *testing.T) {
	t.Run("repos on two installations", func(t *testing.T) {
		e := newEnv(t)
		other := store.SessionRepo{RepoID: testRepo2, FullName: "zeta/api", InstallationID: 2, DefaultBranch: "main"}
		s := e.newSession(withRepos(webRepo(), other))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonMultiInstallation {
			t.Errorf("session = %s %s", x.Status, x.StatusReason)
		}
		if len(e.sbx.Specs()) != 0 {
			t.Error("sandbox created")
		}
	})
	t.Run("clone fails", func(t *testing.T) {
		e := newEnv(t)
		e.sbRules = append(e.sbRules, memsandbox.On(memsandbox.Contains("git clone"), memsandbox.Exit(128, "", "fatal: Authentication failed for "+firstToken)))
		s := e.newSession()
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonCloneFailed {
			t.Errorf("session = %s %s", x.Status, x.StatusReason)
		}
		e.noSecrets(s.ID, firstToken)
	})
	t.Run("provider unavailable", func(t *testing.T) {
		e := newEnv(t)
		s := e.newSession(func(s *store.Session) { s.Provider = "missing" })
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if x := e.st.Session(s.ID); x.StatusReason != ReasonLLMUnavailable {
			t.Errorf("reason = %s", x.StatusReason)
		}
	})
}

func TestBusyAndLostLease(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	l, err := e.leases.Acquire(context.Background(), lease.SessionKey(s.ID), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.run(s.ID); !errors.Is(err, queue.ErrBusy) {
		t.Fatalf("held lease: err = %v", err)
	}
	if len(e.types(s.ID)) != 0 {
		t.Error("events written without the lease")
	}
	_ = l.Release(context.Background())

	// Losing the lease mid-run stops the run without a status, and the
	// task is retried (ErrBusy).
	e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		e.leases.Lose(lease.SessionKey(s.ID))
		time.Sleep(50 * time.Millisecond)
		return toolCall("c1", "message_user", map[string]any{"text": "hi"}), nil
	})
	err = e.run(s.ID)
	if !errors.Is(err, queue.ErrBusy) {
		t.Fatalf("lost lease: err = %v", err)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusRunning {
		t.Errorf("status after a lost lease = %s", st)
	}
	for _, ev := range e.evs.All(s.ID) {
		if ev.Type == events.TypeMessageAssistant || ev.Type == events.TypeToolCallStarted {
			t.Errorf("event after losing the lease: %s", ev.Type)
		}
	}
	if e.sandbox(s.ID).Paused() {
		t.Error("the worker touched the sandbox after losing the lease")
	}

	// The retry resumes the session from its saved state.
	e.llm.Push(llmtest.Text("Recovered."))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("after retry: %s", st)
	}
	if len(e.sbx.Specs()) != 1 {
		t.Errorf("retry provisioned again: %d sandboxes", len(e.sbx.Specs()))
	}
	if n := len(e.events(s.ID, events.TypeMessageUser)); n != 1 {
		t.Errorf("prompt emitted %d times", n)
	}
}

func TestSecretsAreInjectedAndRedacted(t *testing.T) {
	e := newEnv(t)
	e.addSecret("STRIPE_KEY", "org-value-ignored", "")
	e.addSecret("STRIPE_KEY", testSecret, testRepo)
	e.addSecret("OTHER", "another-secret-xyz", "")
	e.diffText = "diff --git a/.env b/.env\n+KEY=" + testSecret + "\n"
	s := e.newSession()
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if strings.Contains(cmd, "printenv") {
			sh := e.lastShells()
			// The secret arrives split across two chunks.
			sh.onOutput(name, []byte("key is SUPERSEC"))
			sh.onOutput(name, []byte("RETVALUE42 and done\n"))
			return sandbox.ShellResult{OutputTail: "key is " + testSecret}, nil
		}
		return sandbox.ShellResult{}, nil
	}
	e.llm.Push(
		toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "printenv STRIPE_KEY # " + testSecret}),
		writeHello("c2"),
		llmtest.Text("The key is "+testSecret),
	)
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if env := e.sbx.Specs()[0].Env; env["STRIPE_KEY"] != testSecret || env["OTHER"] != "another-secret-xyz" {
		t.Errorf("sandbox env = %v", env)
	}
	e.noSecrets(s.ID, testSecret, firstToken, "SUPERSEC")
	var output string
	for _, ev := range e.events(s.ID, events.TypeToolCallOutput) {
		output += payload[events.ToolCallOutputPayload](t, ev).Chunk
	}
	if !strings.Contains(output, "key is [redacted] and done") {
		t.Errorf("streamed output = %q", output)
	}
	if d := e.st.Diffs(s.ID)[0].Diff; strings.Contains(d, testSecret) || !strings.Contains(d, "[redacted]") {
		t.Errorf("diff = %q", d)
	}
}

func TestTokenRotationRewritesFileAndRedacts(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if strings.Contains(cmd, "rotate") {
			e.gh.lastSandbox().rotate("ghs_rotated99")
			sb := e.sandbox(s.ID)
			deadline := time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				if w := writes(sb, "/run/keith/git-token.tmp"); len(w) == 2 {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			e.lastShells().onOutput(name, []byte("token ghs_rotated99\n"))
		}
		return sandbox.ShellResult{}, nil
	}
	e.llm.Push(toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "rotate"}), llmtest.Text("ok"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	sb := e.sandbox(s.ID)
	if w := writes(sb, "/run/keith/git-token.tmp"); len(w) != 2 || w[1] != "ghs_rotated99\n" {
		t.Errorf("token writes = %q", w)
	}
	if n := len(e.execs(sb, "mv -f /run/keith/git-token.tmp /run/keith/git-token")); n != 2 {
		t.Errorf("atomic renames = %d", n)
	}
	e.noSecrets(s.ID, "ghs_rotated99", firstToken)
}

func TestSnapshotDecision(t *testing.T) {
	e := newEnv(t)
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	installs := func() int {
		n := 0
		for _, sh := range e.shells {
			for _, c := range sh.Calls() {
				if c.Op == "exec" && strings.Contains(c.Cmd, "npm ci") {
					n++
				}
			}
		}
		return n
	}

	s1 := e.newSession()
	e.llm.Push(llmtest.Text("one"))
	if err := e.run(s1.ID); err != nil {
		t.Fatal(err)
	}
	sb1 := e.sandbox(s1.ID)
	var order []string
	for _, c := range sb1.Calls() {
		switch {
		case c.Op == memsandbox.OpSnapshot:
			order = append(order, "snapshot")
		case c.Op == memsandbox.OpExec && strings.Contains(c.Exec.Cmd, "rm -rf /tmp/keith-shells"):
			order = append(order, "cleanup")
		case c.Op == memsandbox.OpWrite && c.Path == "/run/keith/git-token.tmp":
			order = append(order, "token")
		}
	}
	wantSeq(t, "snapshot order", order, []string{"token", "cleanup", "snapshot", "token"})
	prof, _ := e.st.GetRepoProfile(context.Background(), testRepo)
	snap := prof.Snapshot
	if snap.ID == "" || installs() != 1 {
		t.Fatalf("snapshot %+v, installs %d", snap, installs())
	}
	if sh := e.shells[0]; len(sh.closed) != 1 || sh.closed[0] != "setup" {
		t.Errorf("closed shells = %v", sh.closed)
	}

	// Same lockfiles: start from the snapshot and skip the install.
	s2 := e.newSession()
	e.llm.Push(llmtest.Text("two"))
	if err := e.run(s2.ID); err != nil {
		t.Fatal(err)
	}
	if sp := e.sbx.Specs(); sp[1].SnapshotID != snap.ID {
		t.Errorf("second session spec = %+v", sp[1])
	}
	if installs() != 1 {
		t.Errorf("install re-ran with unchanged lockfiles: %d", installs())
	}

	// Changed lockfiles: install again and refresh the snapshot.
	e.lockOut = "100644 cafebabe 0\tpackage-lock.json\n"
	s3 := e.newSession()
	e.llm.Push(llmtest.Text("three"))
	if err := e.run(s3.ID); err != nil {
		t.Fatal(err)
	}
	prof, _ = e.st.GetRepoProfile(context.Background(), testRepo)
	if installs() != 2 || prof.Snapshot.ID == snap.ID || prof.Snapshot.LockHash == snap.LockHash {
		t.Errorf("installs %d, snapshot %+v (was %+v)", installs(), prof.Snapshot, snap)
	}

	// Two repos: no snapshot.
	e2 := newEnv(t)
	e2.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	other := store.SessionRepo{RepoID: testRepo2, FullName: "acme/api", InstallationID: 1, DefaultBranch: "main"}
	s4 := e2.newSession(withRepos(webRepo(), other))
	e2.llm.Push(llmtest.Text("four"))
	if err := e2.run(s4.ID); err != nil {
		t.Fatal(err)
	}
	if callsHave(e2.sandbox(s4.ID), memsandbox.OpSnapshot) {
		t.Error("multi-repo session was snapshotted")
	}
	if p, _ := e2.st.GetRepoProfile(context.Background(), testRepo); p.Snapshot.ID != "" {
		t.Errorf("snapshot = %+v", p.Snapshot)
	}

	// A failed install is reported to the agent and not snapshotted.
	e3 := newEnv(t)
	e3.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	e3.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		return sandbox.ShellResult{OutputTail: "npm ERR! missing lockfile", ExitCode: 1}, nil
	}
	s5 := e3.newSession()
	e3.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]
		if !strings.Contains(last.Text(), "npm ERR! missing lockfile") {
			t.Errorf("setup failure not reported to the model: %q", last.Text())
		}
		return llmtest.Text("noted"), nil
	})
	if err := e3.run(s5.ID); err != nil {
		t.Fatal(err)
	}
	if callsHave(e3.sandbox(s5.ID), memsandbox.OpSnapshot) {
		t.Error("failed setup was snapshotted")
	}
}

func TestVerifyCycles(t *testing.T) {
	t.Run("failure fed back, then pass", func(t *testing.T) {
		e := newEnv(t)
		e.profile(store.RepoProfile{RepoID: testRepo, TestCmd: "npm test"})
		runs := 0
		e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
			if strings.Contains(cmd, "npm test") {
				runs++
				if runs == 1 {
					return sandbox.ShellResult{OutputTail: "FAIL date.test.js", ExitCode: 1}, nil
				}
			}
			return sandbox.ShellResult{OutputTail: "ok"}, nil
		}
		s := e.newSession()
		e.setChanges(true, 0)
		e.llm.Push(llmtest.Text("done?"))
		e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
			if last := req.Messages[len(req.Messages)-1]; !strings.Contains(last.Text(), "FAIL date.test.js") {
				t.Errorf("feedback = %q", last.Text())
			}
			return llmtest.Text("fixed"), nil
		})
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if runs != 2 || e.st.Session(s.ID).Status != events.StatusCompleted {
			t.Errorf("test runs %d, status %s", runs, e.st.Session(s.ID).Status)
		}
		mu := e.events(s.ID, events.TypeMessageUser)
		if len(mu) != 2 || !strings.HasPrefix(payload[events.MessageUserPayload](t, mu[1]).MessageID, syntheticInputPrefix) {
			t.Errorf("feedback message.user = %v", mu)
		}
	})
	t.Run("at most two cycles", func(t *testing.T) {
		e := newEnv(t)
		e.profile(store.RepoProfile{RepoID: testRepo, TestCmd: "npm test"})
		runs := 0
		e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
			runs++
			return sandbox.ShellResult{OutputTail: "FAIL", ExitCode: 1}, nil
		}
		s := e.newSession()
		e.setChanges(true, 0)
		e.llm.Push(llmtest.Text("1"), llmtest.Text("2"), llmtest.Text("3"))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if runs != 2 || len(e.llm.Requests()) != 3 || e.st.Session(s.ID).Status != events.StatusCompleted {
			t.Errorf("runs %d, model calls %d, status %s", runs, len(e.llm.Requests()), e.st.Session(s.ID).Status)
		}
	})
	t.Run("no changes, no tests", func(t *testing.T) {
		e := newEnv(t)
		e.profile(store.RepoProfile{RepoID: testRepo, TestCmd: "npm test"})
		s := e.newSession()
		e.llm.Push(llmtest.Text("nothing"))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		for _, ev := range e.events(s.ID, events.TypeToolCallStarted) {
			t.Errorf("unexpected call %s", ev.Payload)
		}
	})
}

func TestShipUsesTheAgentsPR(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.Push(writeHello("c1"))
	e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		e.setChanges(true, 0)
		return toolCall("c2", "open_pr", map[string]any{"title": "Add hello", "body": "Adds hello."}), nil
	})
	e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		// The agent edits once more after opening the PR.
		e.setChanges(true, 1)
		return llmtest.Text("PR opened."), nil
	})
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if prs := e.gh.host.PRRequests(); len(prs) != 1 || prs[0].Title != "Add hello" {
		t.Fatalf("PR requests = %+v", prs)
	}
	if n := len(e.events(s.ID, events.TypePROpened)); n != 1 {
		t.Errorf("pr.opened = %d", n)
	}
	sb := e.sandbox(s.ID)
	if n := len(e.execs(sb, "git push")); n < 2 {
		t.Errorf("the worker did not push the leftover changes: %d pushes", n)
	}
	if st := e.st.Session(s.ID); st.Status != events.StatusCompleted || st.Repos[0].PRNumber != 100 {
		t.Errorf("session = %s, repo %+v", st.Status, st.Repos[0])
	}
}

func TestNoPRMode(t *testing.T) {
	e := newEnv(t)
	e.opts.NoPR = true
	s := e.newSession()
	e.setChanges(true, 0)
	e.llm.Push(llmtest.Text("left changes"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	for _, spec := range e.llm.Requests()[0].Tools {
		if spec.Name == "open_pr" {
			t.Error("open_pr offered in NoPR mode")
		}
	}
	if len(e.gh.host.PRRequests()) != 0 || len(e.execs(e.sandbox(s.ID), "git push")) != 0 {
		t.Error("pushed or opened a PR in NoPR mode")
	}
	if e.st.Session(s.ID).Status != events.StatusCompleted {
		t.Errorf("status = %s", e.st.Session(s.ID).Status)
	}
}

func TestResumeReprovisionsAReapedSandbox(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "?"}))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	old := e.sandbox(s.ID)
	if err := old.Delete(context.Background()); err != nil {
		t.Fatal(err)
	}
	e.cloneOut = "default=main\nstart=origin/agent/" + s.ID + "\n"
	e.st.AddInput(s.ID, store.InputAnswer, "go", "q1", testUser)
	e.llm.Push(llmtest.Text("ok"))
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if len(e.sbx.Specs()) != 2 {
		t.Fatalf("sandboxes = %d", len(e.sbx.Specs()))
	}
	sb := e.sandbox(s.ID)
	if sb.ID() == old.ID() {
		t.Fatal("still on the deleted sandbox")
	}
	if c := e.execs(sb, "git clone"); len(c) != 1 || !strings.Contains(c[0].Cmd, "branch='agent/"+s.ID+"'") {
		t.Errorf("re-clone = %+v", c)
	}
	got := e.statuses(s.ID)
	wantSeq(t, "statuses", got[4:], []string{"provisioning", "setting_up", "running", "no_changes"})
}

func TestReaper(t *testing.T) {
	e := newEnv(t)
	e.opts.IdleTimeout = time.Minute
	e.opts.OrphanLabels = map[string]string{"keith.test": "w"}
	e.opts.SandboxLabels = map[string]string{"keith.test": "w"}
	s := e.newSession()
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	w := e.worker()
	ctx := context.Background()
	if n, err := w.Reap(ctx); err != nil || n != 0 {
		t.Fatalf("fresh session reaped: %d %v", n, err)
	}
	sb := e.sandbox(s.ID)
	e.st.Backdate(s.ID, 2*time.Minute)
	if n, err := w.Reap(ctx); err != nil || n != 1 {
		t.Fatalf("reap = %d %v", n, err)
	}
	if !sb.Deleted() || e.st.Session(s.ID).SandboxID != "" {
		t.Error("idle sandbox not deleted")
	}

	// Orphans: memsandbox cannot list, so give the worker a lister.
	lp := &listingProvider{Provider: e.sbx}
	orphan, _ := e.sbx.Create(ctx, sandbox.Spec{Labels: map[string]string{"keith.test": "w"}})
	w.d.Sandboxes = lp
	lp.ids = []string{orphan.ID()}
	now := time.Now()
	w.d.Now = func() time.Time { return now }
	if n, _ := w.Reap(ctx); n != 0 {
		t.Fatal("orphan reaped before its grace period")
	}
	now = now.Add(w.o.OrphanGrace + time.Second)
	if n, err := w.Reap(ctx); err != nil || n != 1 {
		t.Fatalf("orphan reap = %d %v", n, err)
	}
	if ms, _ := e.sbx.Sandbox(orphan.ID()); !ms.Deleted() {
		t.Error("orphan not deleted")
	}
}

type listingProvider struct {
	*memsandbox.Provider
	ids []string
}

func (l *listingProvider) List(context.Context, map[string]string) ([]string, error) {
	return l.ids, nil
}

func TestStaticTokenSource(t *testing.T) {
	ts := NewStaticTokenSource("ghp_x")
	var got []string
	ts.OnRefresh(func(t github.Token) { got = append(got, t.Value) })
	ctx := context.Background()
	for range 2 {
		if tok, err := ts.Token(ctx); err != nil || tok.Value != "ghp_x" {
			t.Fatal(tok, err)
		}
	}
	if len(got) != 1 {
		t.Errorf("hooks = %v", got)
	}
	done := make(chan error)
	go func() { done <- ts.Run(ctx) }()
	_ = ts.Revoke(ctx)
	if err := <-done; err != nil {
		t.Errorf("Run after Revoke = %v", err)
	}
	if _, err := ts.Token(ctx); !errors.Is(err, github.ErrTokenSourceClosed) {
		t.Errorf("Token after Revoke = %v", err)
	}
}

func TestFailureWithPendingInputDoesNotLoop(t *testing.T) {
	e := newEnv(t)
	e.sbRules = append(e.sbRules, memsandbox.On(memsandbox.Contains("git clone"), memsandbox.Exit(128, "", "fatal: repository not found")))
	s := e.newSession()
	e.st.AddInput(s.ID, store.InputMessage, "hurry", "", testUser)
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if got := e.statuses(s.ID); len(got) != 2 || got[1] != "failed" {
		t.Errorf("statuses = %v", got)
	}
	if n := len(e.sbx.Specs()); n != 1 {
		t.Errorf("provision attempts = %d", n)
	}
}

func TestTimeLimit(t *testing.T) {
	e := newEnv(t)
	e.opts.MaxSessionDuration = 200 * time.Millisecond
	s := e.newSession()
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		select {
		case <-e.lastShells().interrupts:
			return sandbox.ShellResult{ExitCode: 130}, nil
		case <-time.After(10 * time.Second):
			return sandbox.ShellResult{}, errors.New("not interrupted")
		}
	}
	e.llm.Push(toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "sleep 600"}))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonTimeLimit {
		t.Errorf("session = %s %s", x.Status, x.StatusReason)
	}
}
