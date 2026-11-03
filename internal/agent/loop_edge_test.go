package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
)

// TestAppendOnlyAcrossPauseResumeAndInputs drives a long session through
// tool batches, a mid-run message, an ask_user pause, a save/load of the
// state and a resume, then checks that every request extended the last.
func TestAppendOnlyAcrossPauseResumeAndInputs(t *testing.T) {
	h := newHarness(t, append(agent.DefaultTools(agent.ToolOptions{}), echoTool)...)
	h.queue.Push(agent.Input{ID: "p", Text: "do the thing"})
	h.prov.Push(
		llmtest.ToolCalls(llm.TextBlock("step 1"), call("a1", "echo", map[string]string{"x": "1"})),
		&llm.Response{ // thinking blocks and provider data must be replayed untouched
			Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{
				{Type: llm.BlockThinking, Text: "", Signature: "sig-2", ProviderRaw: json.RawMessage(`{"r":2}`)},
				call("a2", "echo", map[string]string{"x": "2"}),
				call("a3", "echo", map[string]string{"x": "3"}),
			}},
			StopReason: llm.StopToolUse,
			Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
			Raw:        json.RawMessage(`{"id":"resp-2"}`),
		},
		llmtest.ToolCalls(call("q1", "ask_user", map[string]any{"question": "pg or mysql?"}), call("a4", "echo", map[string]string{})),
	)
	h.rec.Before = func(i int) {
		if i == 1 {
			h.queue.Push(agent.Input{ID: "m1", Text: "mid-run note"})
		}
	}
	wantKind(t, h.runOnce(), agent.OutcomePaused)
	if len(h.st.Aux.Raw) != 1 || string(h.st.Aux.Raw[0].Raw) != `{"id":"resp-2"}` || h.st.Aux.Raw[0].Step != 2 {
		t.Fatalf("raw = %+v", h.st.Aux.Raw)
	}

	h.reload()
	h.queue.Push(agent.Input{ID: "m2", Text: "pg", QuestionID: "q1"})
	h.prov.Push(
		llmtest.ToolCall("a5", "echo", map[string]string{"x": "5"}),
		&llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{llm.TextBlock("partial")}}, StopReason: llm.StopPauseTurn},
		llmtest.ToolCall("a6", "echo", map[string]string{"x": "6"}),
		llmtest.Text("finished"),
	)
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Text != "finished" {
		t.Fatalf("text = %q", out.Text)
	}

	calls := h.rec.Calls()
	if len(calls) != 7 {
		t.Fatalf("%d calls", len(calls))
	}
	agenttest.CheckAppendOnly(t, calls)
	// The pause_turn continuation is sent unchanged.
	if !jsonEq(calls[5].Req.Messages, append(calls[4].Req.Messages, calls[4].Resp.Message)) {
		t.Fatal("pause_turn continuation added a message")
	}
	// The mid-run note rode along with a2/a3's results.
	um := calls[2].Req.Messages[len(calls[2].Req.Messages)-1]
	if len(um.Content) != 3 || um.Content[2].Text != "mid-run note" {
		t.Fatalf("user turn = %s", mustJSON(um))
	}
	if h.st.Aux.Steps != 7 || len(h.st.Messages) != 13 { // pause_turn adds no user message
		t.Fatalf("steps %d messages %d", h.st.Aux.Steps, len(h.st.Messages))
	}
}

func TestMaxTokensAsksToContinue(t *testing.T) {
	h := newHarness(t)
	cut := llmtest.Text("cut o")
	cut.StopReason = llm.StopMaxTokens
	h.prov.Push(cut, llmtest.Text("ff here"))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	last := calls[1].Req.Messages
	if !strings.Contains(last[len(last)-1].Text(), "output token limit") {
		t.Fatalf("continue note = %q", last[len(last)-1].Text())
	}
}

func TestRefusalFails(t *testing.T) {
	h := newHarness(t)
	r := llmtest.Text("I can't help with that.")
	r.StopReason = llm.StopRefusal
	h.prov.Push(r)
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonRefusal || out.Text != "I can't help with that." {
		t.Fatalf("outcome = %+v", out)
	}
}

func TestUnknownToolAndPanicAreToolErrors(t *testing.T) {
	panicky := fakeTool{name: "panicky", fn: func(context.Context, *agent.Run, json.RawMessage) (agent.Result, error) {
		panic("kaboom")
	}}
	h := newHarness(t, panicky)
	h.prov.Push(
		llmtest.ToolCalls(call("c1", "nope", map[string]string{}), call("c2", "panicky", map[string]string{})),
		llmtest.Text("ok"),
	)
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	res := agenttest.ToolResults(h.rec.Calls()[1].Req)
	if !res[0].IsError || !strings.Contains(agenttest.BlockText(res[0]), `Unknown tool "nope"`) {
		t.Fatalf("unknown = %+v", res[0])
	}
	if !res[1].IsError || !strings.Contains(agenttest.BlockText(res[1]), "kaboom") {
		t.Fatalf("panic = %+v", res[1])
	}
	fin := agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished)
	if len(fin) != 2 || !fin[0].IsError || !fin[1].IsError {
		t.Fatalf("finished = %+v", fin)
	}
}

func TestToolErrorStopsTheRun(t *testing.T) {
	sandboxGone := errors.New("sandbox gone")
	broken := fakeTool{name: "broken", fn: func(context.Context, *agent.Run, json.RawMessage) (agent.Result, error) {
		return agent.Result{}, sandboxGone
	}}
	h := newHarness(t, broken, echoTool)
	h.prov.Push(llmtest.ToolCalls(call("c1", "echo", map[string]string{}), call("c2", "broken", map[string]string{}), call("c3", "echo", map[string]string{})))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonToolError || !errors.Is(out.Err, sandboxGone) {
		t.Fatalf("outcome = %+v", out)
	}
	// Every tool_use still has a result, and c2 has its finished event.
	res := h.st.Aux.Pending.Results
	if len(res) != 3 || res[0].IsError || !res[1].IsError || !res[2].IsError {
		t.Fatalf("results = %s", mustJSON(res))
	}
	fin := agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished)
	if len(fin) != 2 || fin[1].CallID != "c2" || !fin[1].IsError {
		t.Fatalf("finished = %+v", fin)
	}
}

func TestEmitFailureStopsTheRun(t *testing.T) {
	h := newHarness(t, echoTool)
	h.events.Fail = func(typ string) error {
		if typ == events.TypeToolCallStarted {
			return errors.New("store down")
		}
		return nil
	}
	h.prov.Push(llmtest.ToolCall("c1", "echo", map[string]string{}))
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonEmitFailed {
		t.Fatalf("outcome = %+v", out)
	}

	// A publish failure after the event was stored is not fatal.
	h2 := newHarness(t, echoTool)
	h2.events.Fail = func(typ string) error { return events.ErrPublish }
	h2.prov.Push(llmtest.ToolCall("c1", "echo", map[string]string{}), llmtest.Text("ok"))
	wantKind(t, h2.runOnce(), agent.OutcomeEndTurn)
}

func TestInputsAndCheckpointErrors(t *testing.T) {
	h := newHarness(t)
	h.queue.SetErr(errors.New("redis down"))
	out := h.runOnce()
	if out.Kind != agent.OutcomeFailed || out.Reason != agent.ReasonInputsFailed {
		t.Fatalf("outcome = %+v", out)
	}

	h = newHarness(t)
	h.cfg.Checkpoint = func(context.Context, *agent.State) error { return errors.New("db down") }
	h.prov.Push(llmtest.Text("hi"))
	out = h.runOnce()
	if out.Kind != agent.OutcomeFailed || out.Reason != agent.ReasonCheckpointFailed {
		t.Fatalf("outcome = %+v", out)
	}
	// The response is in the state even though saving failed.
	if len(h.st.Messages) != 2 {
		t.Fatalf("messages = %d", len(h.st.Messages))
	}
}

func TestNewRunnerValidates(t *testing.T) {
	p := llmtest.New()
	cases := map[string]agent.Config{
		"no provider": {Model: "m"},
		"no model":    {Provider: p},
		"dup tool":    {Provider: p, Model: "m", Tools: []agent.Tool{echoTool, echoTool}},
		"empty name":  {Provider: p, Model: "m", Tools: []agent.Tool{fakeTool{}}},
	}
	for name, cfg := range cases {
		if _, err := agent.NewRunner(cfg); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

// TestStateHasNoNUL checks that a failing call (which repeat detection
// records) and output with NUL bytes leave a state Postgres jsonb can
// store: no \u0000 anywhere.
func TestStateHasNoNUL(t *testing.T) {
	nul := fakeTool{name: "nul", fn: func(context.Context, *agent.Run, json.RawMessage) (agent.Result, error) {
		return agent.Result{IsError: true, Content: []llm.Block{llm.TextBlock("bin\x00ary")}, Summary: "failed"}, nil
	}}
	h := newHarness(t, nul)
	h.queue.Push(agent.Input{ID: "p", Text: "go\x00"})
	h.prov.Push(llmtest.ToolCall("n1", "nul", map[string]string{}), llmtest.Text("done"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	msgs, aux, err := h.st.MarshalColumns()
	if err != nil {
		t.Fatal(err)
	}
	for _, col := range [][]byte{msgs, aux} {
		if strings.Contains(string(col), `\u0000`) {
			t.Fatalf("state holds \\u0000: %s", col)
		}
	}
	if !strings.Contains(string(msgs), "bin�ary") {
		t.Errorf("tool output not kept: %s", msgs)
	}
}
