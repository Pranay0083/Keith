package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

// fakeTool is a tool with a scripted Execute.
type fakeTool struct {
	name string
	fn   func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error)
}

func (f fakeTool) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: f.name, Description: f.name, Strict: true,
		InputSchema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}},"required":[],"additionalProperties":false}`)}
}

func (f fakeTool) Execute(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
	return f.fn(ctx, run, input)
}

// echoTool returns its input and the call ID.
var echoTool = fakeTool{name: "echo", fn: func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
	return agent.Result{Content: []llm.Block{llm.TextBlock("echo " + string(input))}, Summary: "echoed " + agent.CallID(ctx)}, nil
}}

// failTool always fails the same way.
var failTool = fakeTool{name: "fail", fn: func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
	return agent.Result{IsError: true, Content: []llm.Block{llm.TextBlock(fmt.Sprintf("boom after %dms", time.Now().Nanosecond()%1000))}}, nil
}}

type harness struct {
	t      *testing.T
	prov   *llmtest.Provider
	rec    *agenttest.Recording
	events *agenttest.Recorder
	queue  *agenttest.Queue
	sb     *memsandbox.Sandbox
	shells *agenttest.Shells
	host   *agenttest.Host
	run    *agent.Run
	st     *agent.State
	cfg    agent.Config
	sleeps func() []time.Duration
	stop   bool
	saves  int
}

func newHarness(t *testing.T, tools ...agent.Tool) *harness {
	t.Helper()
	h := &harness{
		t:      t,
		prov:   llmtest.New(),
		events: &agenttest.Recorder{},
		queue:  &agenttest.Queue{},
		sb:     memsandbox.New(),
		shells: &agenttest.Shells{},
		host:   &agenttest.Host{},
		st:     agent.NewState("You are a test agent."),
	}
	h.rec = agenttest.Record(h.prov)
	h.run = &agent.Run{
		SessionID: "s1",
		Sandbox:   h.sb,
		Shells:    h.shells,
		Emit:      h.events.Emit,
		Host:      h.host,
		Repos: []agent.RepoContext{{
			FullName: "acme/api", Path: "/workspace/api", Branch: "agent/s1", DefaultBranch: "main",
			Profile: agent.RepoProfile{TestCmd: "go test ./..."},
		}},
	}
	sleep, sleeps := agenttest.Sleeps()
	h.sleeps = sleeps
	h.cfg = agent.Config{
		Provider:   h.rec,
		Model:      "test-model",
		Tools:      tools,
		Effort:     llm.EffortHigh,
		Inputs:     h.queue.Poll,
		ShouldStop: func(context.Context) bool { return h.stop },
		Checkpoint: func(context.Context, *agent.State) error { h.saves++; return nil },
		Sleep:      sleep,
		NewID:      agenttest.IDs(),
	}
	t.Cleanup(func() { h.events.Check(t) })
	return h
}

func (h *harness) runOnce() agent.Outcome {
	h.t.Helper()
	r, err := agent.NewRunner(h.cfg)
	if err != nil {
		h.t.Fatal(err)
	}
	return r.Run(context.Background(), h.run, h.st)
}

// reload saves and loads the state the way the worker does between runs.
func (h *harness) reload() {
	h.t.Helper()
	msgs, aux, err := h.st.MarshalColumns()
	if err != nil {
		h.t.Fatal(err)
	}
	st, err := agent.UnmarshalState(msgs, aux)
	if err != nil {
		h.t.Fatal(err)
	}
	h.st = st
	h.run = &agent.Run{SessionID: h.run.SessionID, Sandbox: h.run.Sandbox, Shells: h.run.Shells,
		Emit: h.run.Emit, Host: h.run.Host, Repos: h.run.Repos}
}

func call(id, name string, input any) llm.Block {
	b, err := json.Marshal(input)
	if err != nil {
		panic(err)
	}
	return llm.ToolUseBlock(id, name, b)
}

func wantKind(t *testing.T, out agent.Outcome, kind agent.OutcomeKind) {
	t.Helper()
	if out.Kind != kind {
		t.Fatalf("outcome = %s (reason %q, err %v), want %s", out.Kind, out.Reason, out.Err, kind)
	}
}

func TestLoopRunsToolsInOrderAndEndsTurn(t *testing.T) {
	h := newHarness(t, echoTool)
	h.prov.Push(
		llmtest.ToolCalls(llm.TextBlock("Looking."), call("c1", "echo", map[string]string{"x": "a"}), call("c2", "echo", map[string]string{"x": "b"})),
		llmtest.ToolCall("c3", "echo", map[string]string{"x": "c"}),
		llmtest.Text("All done."),
	)
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Text != "All done." || out.Steps != 3 || out.Usage.InputTokens != 30 {
		t.Fatalf("outcome = %+v", out)
	}

	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	if len(calls) != 3 {
		t.Fatalf("%d calls", len(calls))
	}
	req := calls[0].Req
	if req.Model != "test-model" || req.Effort != llm.EffortHigh || req.System != "You are a test agent." || len(req.Tools) != 1 {
		t.Fatalf("request = %+v", req)
	}
	if got := req.Messages[0].Text(); got != agent.DefaultKickoff {
		t.Fatalf("kickoff = %q", got)
	}
	res := agenttest.ToolResults(calls[1].Req)
	if len(res) != 2 || res[0].ToolUseID != "c1" || res[1].ToolUseID != "c2" ||
		agenttest.BlockText(res[0]) != `echo {"x":"a"}` {
		t.Fatalf("results = %+v", res)
	}

	want := []string{
		events.TypeMessageAssistant,
		events.TypeToolCallStarted, events.TypeToolCallFinished,
		events.TypeToolCallStarted, events.TypeToolCallFinished,
		events.TypeToolCallStarted, events.TypeToolCallFinished,
		events.TypeMessageAssistant,
	}
	if got := h.events.Stored(); !slices.Equal(got, want) {
		t.Fatalf("events = %v\nwant %v", got, want)
	}
	fin := agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished)
	if fin[0].CallID != "c1" || fin[0].Summary != "echoed c1" || fin[0].IsError {
		t.Fatalf("finished = %+v", fin[0])
	}
	started := agenttest.Payloads[events.ToolCallStartedPayload](h.events, events.TypeToolCallStarted)
	if string(started[1].Input) != `{"x":"b"}` || started[1].Tool != "echo" {
		t.Fatalf("started = %+v", started[1])
	}
	if h.saves < 4 {
		t.Fatalf("checkpoints = %d", h.saves)
	}
	if len(h.st.Messages) != 6 || h.st.Aux.Steps != 3 || h.st.Aux.Pending != nil {
		t.Fatalf("state: %d messages, %d steps, pending %+v", len(h.st.Messages), h.st.Aux.Steps, h.st.Aux.Pending)
	}

	// Run again with nothing new: the turn is over.
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Steps != 0 || out.Text != "All done." {
		t.Fatalf("idle run = %+v", out)
	}
}

func TestDeltasShareTheAssistantMessageID(t *testing.T) {
	h := newHarness(t)
	h.prov.Push(llmtest.ToolCalls(llm.TextBlock("one"), llm.TextBlock("two")))
	h.prov.Push(llmtest.Text("three"))
	// ToolCalls has StopToolUse but no tool_use blocks: it is treated as
	// an end of turn.
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	deltas := agenttest.Payloads[events.MessageDeltaPayload](h.events, events.TypeMessageDelta)
	msgs := agenttest.Payloads[events.MessageAssistantPayload](h.events, events.TypeMessageAssistant)
	if len(deltas) != 2 || len(msgs) != 1 {
		t.Fatalf("deltas %+v msgs %+v", deltas, msgs)
	}
	for _, d := range deltas {
		if d.MessageID != msgs[0].MessageID {
			t.Fatalf("delta %+v vs message %+v", d, msgs[0])
		}
	}
	if msgs[0].Text != "one\ntwo" {
		t.Fatalf("text = %q", msgs[0].Text)
	}
	if h.prov.Remaining() != 1 {
		t.Fatal("loop should have stopped after the turn")
	}
}

func TestAskUserPausesAndResumesWithSiblingResults(t *testing.T) {
	h := newHarness(t, append(agent.DefaultTools(agent.ToolOptions{}), echoTool)...)
	h.prov.Push(llmtest.ToolCalls(
		call("q1", "ask_user", map[string]any{"question": "Which DB?", "choices": []string{"pg", "mysql"}}),
		call("e1", "echo", map[string]string{"x": "sib"}),
	))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if out.Question == nil || out.Question.ID != "q1" || out.Question.Text != "Which DB?" || !slices.Equal(out.Question.Choices, []string{"pg", "mysql"}) {
		t.Fatalf("question = %+v", out.Question)
	}
	// The sibling ran; question.asked comes after the whole batch.
	want := []string{
		events.TypeToolCallStarted, events.TypeToolCallFinished,
		events.TypeToolCallStarted, events.TypeToolCallFinished,
		events.TypeQuestionAsked,
	}
	if got := h.events.Stored(); !slices.Equal(got, want) {
		t.Fatalf("events = %v", got)
	}
	qa := agenttest.Payloads[events.QuestionAskedPayload](h.events, events.TypeQuestionAsked)[0]
	if qa.QuestionID != "q1" || qa.Question != "Which DB?" || len(qa.Choices) != 2 {
		t.Fatalf("question.asked = %+v", qa)
	}
	if q := h.st.OpenQuestion(); q == nil || q.ToolUseID != "q1" {
		t.Fatalf("open question = %+v", q)
	}
	if n := len(h.st.Aux.Pending.Results); n != 2 {
		t.Fatalf("held results = %d", n)
	}

	h.reload()
	// Resuming without an answer stays paused.
	out = h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if out.Steps != 0 {
		t.Fatal("paused run called the model")
	}

	// An unrelated message and the answer arrive together; the answer is
	// matched by question ID.
	h.events.Reset()
	h.queue.Push(
		agent.Input{ID: "m1", UserID: "u1", Text: "also: keep it small"},
		agent.Input{ID: "m2", UserID: "u1", Text: "pg", QuestionID: "q1"},
	)
	h.prov.Push(llmtest.Text("Using pg."))
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)

	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	last := calls[len(calls)-1].Req.Messages
	um := last[len(last)-1]
	if len(um.Content) != 3 ||
		um.Content[0].ToolUseID != "q1" || agenttest.BlockText(um.Content[0]) != "pg" ||
		um.Content[1].ToolUseID != "e1" ||
		um.Content[2].Type != llm.BlockText || um.Content[2].Text != "also: keep it small" {
		t.Fatalf("resumed user message = %s", mustJSON(um))
	}
	mu := agenttest.Payloads[events.MessageUserPayload](h.events, events.TypeMessageUser)
	if len(mu) != 2 || mu[0].MessageID != "m1" || mu[0].QuestionID != "" || mu[1].MessageID != "m2" || mu[1].QuestionID != "q1" || mu[1].UserID != "u1" {
		t.Fatalf("message.user = %+v", mu)
	}
	if h.st.OpenQuestion() != nil {
		t.Fatal("question still open")
	}
}

func TestPlainMessageAnswersOpenQuestion(t *testing.T) {
	h := newHarness(t, agent.DefaultTools(agent.ToolOptions{})...)
	h.prov.Push(llmtest.ToolCall("q1", "ask_user", map[string]any{"question": "Proceed?"}))
	wantKind(t, h.runOnce(), agent.OutcomePaused)

	var delivered []agent.Input
	h.cfg.Delivered = func(ctx context.Context, in []agent.Input) error { delivered = in; return nil }
	h.queue.Push(agent.Input{ID: "m1", Text: "yes"})
	h.prov.Push(llmtest.Text("ok"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	res := agenttest.ToolResults(h.rec.Calls()[1].Req)
	if len(res) != 1 || agenttest.BlockText(res[0]) != "yes" {
		t.Fatalf("results = %+v", res)
	}
	mu := agenttest.Payloads[events.MessageUserPayload](h.events, events.TypeMessageUser)
	if mu[0].QuestionID != "q1" || len(delivered) != 1 || delivered[0].QuestionID != "q1" {
		t.Fatalf("message.user = %+v delivered = %+v", mu, delivered)
	}
}

func TestSecondQuestionInOneTurnIsRefused(t *testing.T) {
	h := newHarness(t, agent.DefaultTools(agent.ToolOptions{})...)
	h.prov.Push(llmtest.ToolCalls(
		call("q1", "ask_user", map[string]any{"question": "A?"}),
		call("q2", "ask_user", map[string]any{"question": "B?"}),
	))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if out.Question.ID != "q1" {
		t.Fatalf("question = %+v", out.Question)
	}
	if n := len(agenttest.Payloads[events.QuestionAskedPayload](h.events, events.TypeQuestionAsked)); n != 1 {
		t.Fatalf("%d question.asked", n)
	}
	r := h.st.Aux.Pending.Results[1]
	if !r.IsError || !strings.Contains(agenttest.BlockText(r), "already waiting") {
		t.Fatalf("second ask = %+v", r)
	}
}

func TestMidRunInputJoinsToolResults(t *testing.T) {
	h := newHarness(t, echoTool)
	h.prov.Push(llmtest.ToolCall("c1", "echo", map[string]string{}))
	h.prov.PushFunc(func(req llm.Request) (*llm.Response, error) { return llmtest.Text("noted"), nil })
	h.queue.Push(agent.Input{ID: "first", Text: "Fix the login bug"})
	h.rec.Before = func(i int) {
		if i == 0 {
			h.queue.Push(agent.Input{ID: "m1", Text: "use the v2 API"})
			h.queue.Push(agent.Input{ID: "first", Text: "duplicate delivery"})
		}
	}
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	if got := calls[0].Req.Messages[0].Text(); got != "Fix the login bug" {
		t.Fatalf("first message = %q (no kickoff when there is input)", got)
	}
	um := calls[1].Req.Messages[2]
	if len(um.Content) != 2 || um.Content[0].Type != llm.BlockToolResult || um.Content[1].Text != "use the v2 API" {
		t.Fatalf("user turn = %s", mustJSON(um))
	}
	mu := agenttest.Payloads[events.MessageUserPayload](h.events, events.TypeMessageUser)
	if len(mu) != 2 || mu[0].MessageID != "first" || mu[1].MessageID != "m1" {
		t.Fatalf("message.user = %+v", mu)
	}
}

func TestInputAfterEndTurnContinues(t *testing.T) {
	h := newHarness(t)
	h.prov.Push(llmtest.Text("first answer"), llmtest.Text("second answer"))
	h.rec.Before = func(i int) {
		if i == 0 {
			h.queue.Push(agent.Input{ID: "m1", Text: "one more thing"})
		}
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Text != "second answer" || out.Steps != 2 {
		t.Fatalf("outcome = %+v", out)
	}
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestCancelBetweenSteps(t *testing.T) {
	h := newHarness(t, echoTool)
	ran := 0
	counting := fakeTool{name: "count", fn: func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
		ran++
		h.stop = true // the user presses Stop while the first call runs
		return agent.Result{Content: []llm.Block{llm.TextBlock("ok")}}, nil
	}}
	h.cfg.Tools = []agent.Tool{counting}
	h.prov.Push(llmtest.ToolCalls(call("c1", "count", map[string]string{}), call("c2", "count", map[string]string{})))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeCancelled)
	if ran != 1 {
		t.Fatalf("ran %d tools", ran)
	}
	res := h.st.Aux.Pending.Results
	if len(res) != 2 || res[0].IsError || !res[1].IsError || !strings.Contains(agenttest.BlockText(res[1]), "stopped") {
		t.Fatalf("results = %s", mustJSON(res))
	}
	if got := h.events.Stored(); !slices.Equal(got, []string{events.TypeToolCallStarted, events.TypeToolCallFinished}) {
		t.Fatalf("events = %v", got)
	}

	// Stop is checked before the first step too.
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeCancelled)

	// A later follow-up resumes with the held results.
	h.stop = false
	h.queue.Push(agent.Input{ID: "m1", Text: "carry on"})
	h.prov.Push(llmtest.Text("done"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestStepCap(t *testing.T) {
	h := newHarness(t, echoTool)
	h.cfg.MaxSteps = 2
	for i := 0; i < 5; i++ {
		h.prov.Push(llmtest.ToolCall(fmt.Sprintf("c%d", i), "echo", map[string]string{}))
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeStepCap)
	if out.Steps != 2 || len(h.rec.Calls()) != 2 {
		t.Fatalf("steps = %d, calls = %d", out.Steps, len(h.rec.Calls()))
	}
	if h.st.Aux.Pending == nil || len(h.st.Aux.Pending.Results) != 1 {
		t.Fatalf("pending = %+v", h.st.Aux.Pending)
	}
	// Each run gets a fresh allowance.
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeStepCap)
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestTokenBudget(t *testing.T) {
	h := newHarness(t, echoTool)
	h.cfg.TokenBudget = 30 // each scripted response uses 15
	for i := 0; i < 5; i++ {
		h.prov.Push(llmtest.ToolCall(fmt.Sprintf("c%d", i), "echo", map[string]string{}))
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeBudgetExceeded)
	if out.Steps != 2 || h.st.Aux.Usage.InputTokens != 20 || h.st.Aux.Usage.OutputTokens != 10 {
		t.Fatalf("outcome = %+v usage = %+v", out, h.st.Aux.Usage)
	}
	// The budget is per session: the next run stops at once.
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeBudgetExceeded)
	if out.Steps != 0 {
		t.Fatal("over-budget run called the model")
	}
	// A custom cost that counts output only (10 so far) leaves room.
	h.cfg.Cost = func(u llm.Usage) int { return u.OutputTokens }
	h.prov.Push(llmtest.Text("done"))
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Text != "done" || out.Steps != 4 {
		t.Fatalf("outcome = %+v", out)
	}
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestRetryHonoursRetryAfter(t *testing.T) {
	h := newHarness(t)
	h.prov.PushError(&llm.Error{Provider: "fake", Kind: llm.ErrRateLimited, RetryAfter: 7 * time.Second})
	h.prov.PushError(&llm.Error{Provider: "fake", Kind: llm.ErrOverloaded})
	h.prov.Push(llmtest.Text("hi"))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if got := h.sleeps(); !slices.Equal(got, []time.Duration{7 * time.Second, 4 * time.Second}) {
		t.Fatalf("sleeps = %v", got)
	}
	calls := h.rec.Calls()
	if len(calls) != 3 || out.Steps != 1 {
		t.Fatalf("calls = %d steps = %d", len(calls), out.Steps)
	}
	agenttest.CheckAppendOnly(t, calls)
	if !jsonEq(calls[0].Req, calls[2].Req) {
		t.Fatal("retry sent a different request")
	}
}

func TestRetryGivesUpAfterThreeAttempts(t *testing.T) {
	h := newHarness(t)
	for i := 0; i < 3; i++ {
		h.prov.PushError(&llm.Error{Provider: "fake", Kind: llm.ErrServer, RetryAfter: time.Hour})
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonLLMError || !llm.IsRetryable(out.Err) {
		t.Fatalf("outcome = %+v", out)
	}
	// RetryAfter is capped.
	if got := h.sleeps(); !slices.Equal(got, []time.Duration{2 * time.Minute, 2 * time.Minute}) {
		t.Fatalf("sleeps = %v", got)
	}
	// The kickoff is still pending, so a later run retries the same turn.
	if h.st.Aux.Pending == nil || len(h.st.Messages) != 0 {
		t.Fatalf("state = %+v", h.st)
	}
	h.prov.Push(llmtest.Text("back"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestNonRetryableErrors(t *testing.T) {
	cases := []struct {
		err    error
		reason string
	}{
		{&llm.Error{Provider: "fake", Kind: llm.ErrContextTooLong}, agent.ReasonContextExhausted},
		{&llm.Error{Provider: "fake", Kind: llm.ErrInvalidRequest}, agent.ReasonLLMError},
		{errors.New("plain"), agent.ReasonLLMError},
	}
	for _, c := range cases {
		h := newHarness(t)
		h.prov.PushError(c.err)
		out := h.runOnce()
		wantKind(t, out, agent.OutcomeFailed)
		if out.Reason != c.reason || !errors.Is(out.Err, c.err) || len(h.sleeps()) != 0 || len(h.rec.Calls()) != 1 {
			t.Fatalf("%v: outcome = %+v", c.err, out)
		}
	}
}

func TestEmptyResponseIsRetried(t *testing.T) {
	h := newHarness(t)
	h.prov.Push(&llm.Response{Message: llm.Message{Role: llm.RoleAssistant}, StopReason: llm.StopEndTurn})
	h.prov.Push(llmtest.Text("there"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	agenttest.CheckAppendOnly(t, h.rec.Calls())
}

func TestCancelledContext(t *testing.T) {
	h := newHarness(t)
	r, err := agent.NewRunner(h.cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := r.Run(ctx, h.run, h.st)
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonInterrupted {
		t.Fatalf("reason = %s", out.Reason)
	}
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func jsonEq(a, b any) bool { return mustJSON(a) == mustJSON(b) }
