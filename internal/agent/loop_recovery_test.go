package agent_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
)

// openCallsState is the state a worker saves after a model call asked for
// tools, before any of them ran (the checkpoint after generate). A worker
// that crashes or loses its lease while a tool runs leaves exactly this.
func openCallsState(h *harness) {
	h.st.Messages = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock("Fix the build")}},
		{Role: llm.RoleAssistant, Content: []llm.Block{
			llm.TextBlock("Running both."),
			call("c1", "echo", map[string]string{"x": "1"}),
			call("c2", "echo", map[string]string{"x": "2"}),
		}},
	}
	h.st.Aux.Steps = 1
}

func TestRunAnswersToolCallsAStoppedWorkerLeftOpen(t *testing.T) {
	h := newHarness(t, echoTool)
	openCallsState(h)
	h.prov.Push(llmtest.Text("Checked the state; carrying on."))

	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if out.Steps != 1 || out.Text != "Checked the state; carrying on." {
		t.Fatalf("outcome = %+v", out)
	}
	calls := h.rec.Calls()
	if len(calls) != 1 {
		t.Fatalf("%d model calls, want 1", len(calls))
	}
	// No tool_use goes out without its tool_result.
	agenttest.CheckAppendOnly(t, calls)
	res := agenttest.ToolResults(calls[0].Req)
	if len(res) != 2 || res[0].ToolUseID != "c1" || res[1].ToolUseID != "c2" {
		t.Fatalf("results = %s", mustJSON(res))
	}
	for _, r := range res {
		if !r.IsError || !strings.HasPrefix(agenttest.BlockText(r), "Interrupted: the worker stopped while this tool call was running") {
			t.Fatalf("placeholder = %s", mustJSON(r))
		}
	}

	want := []string{events.TypeToolCallFinished, events.TypeToolCallFinished, events.TypeMessageAssistant}
	if got := h.events.Stored(); !slices.Equal(got, want) {
		t.Fatalf("events = %v\nwant %v", got, want)
	}
	fin := agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished)
	if fin[0] != (events.ToolCallFinishedPayload{CallID: "c1", Tool: "echo", IsError: true, Summary: "interrupted"}) ||
		fin[1].CallID != "c2" {
		t.Fatalf("finished = %+v", fin)
	}
	// The placeholders were saved before the model call.
	if h.saves < 2 {
		t.Fatalf("checkpoints = %d", h.saves)
	}
	if len(h.st.Messages) != 4 || h.st.Aux.Pending != nil {
		t.Fatalf("state: %d messages, pending %+v", len(h.st.Messages), h.st.Aux.Pending)
	}
}

// A result already held for one call is kept; only the missing one gets a
// placeholder, and the results follow the tool_use order. Pending texts
// stay as they were.
func TestRunKeepsHeldResultsWhenAnsweringOpenCalls(t *testing.T) {
	h := newHarness(t, echoTool)
	openCallsState(h)
	h.st.Aux.Pending = &agent.PendingTurn{
		Results: []llm.Block{llm.ToolResultBlock("c2", false, llm.TextBlock("c2 ran"))},
		Texts:   []llm.Block{llm.TextBlock("a note")},
	}
	h.prov.Push(llmtest.Text("ok"))

	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	last := calls[0].Req.Messages[len(calls[0].Req.Messages)-1]
	if len(last.Content) != 3 ||
		last.Content[0].ToolUseID != "c1" || !last.Content[0].IsError ||
		last.Content[1].ToolUseID != "c2" || agenttest.BlockText(last.Content[1]) != "c2 ran" ||
		last.Content[2].Type != llm.BlockText || last.Content[2].Text != "a note" {
		t.Fatalf("user turn = %s", mustJSON(last))
	}
	fin := agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished)
	if len(fin) != 1 || fin[0].CallID != "c1" {
		t.Fatalf("finished = %+v", fin)
	}
}

// A session paused on ask_user holds a result (the question's placeholder)
// for every call: resuming it without an answer changes nothing.
func TestRunLeavesAPausedTurnAlone(t *testing.T) {
	h := newHarness(t, append(agent.DefaultTools(agent.ToolOptions{}), echoTool)...)
	h.prov.Push(llmtest.ToolCalls(
		call("q1", "ask_user", map[string]any{"question": "Which DB?"}),
		call("e1", "echo", map[string]string{"x": "sib"}),
	))
	wantKind(t, h.runOnce(), agent.OutcomePaused)
	h.reload()
	before := mustJSON(h.st)
	saves := h.saves
	h.events.Reset()

	out := h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if out.Steps != 0 || len(h.events.Types()) != 0 || h.saves != saves {
		t.Fatalf("steps %d, events %v, saves %d -> %d", out.Steps, h.events.Types(), saves, h.saves)
	}
	if got := mustJSON(h.st); got != before {
		t.Fatalf("state changed:\n got %s\nwant %s", got, before)
	}
}

func TestAnsweringOpenCallsReportsEmitAndCheckpointFailures(t *testing.T) {
	h := newHarness(t, echoTool)
	openCallsState(h)
	h.events.Fail = func(typ string) error {
		if typ == events.TypeToolCallFinished {
			return errors.New("store down")
		}
		return nil
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonEmitFailed || len(h.rec.Calls()) != 0 {
		t.Fatalf("outcome = %+v, %d model calls", out, len(h.rec.Calls()))
	}

	h = newHarness(t, echoTool)
	openCallsState(h)
	h.cfg.Checkpoint = func(context.Context, *agent.State) error { return errors.New("db down") }
	out = h.runOnce()
	wantKind(t, out, agent.OutcomeFailed)
	if out.Reason != agent.ReasonCheckpointFailed || len(h.rec.Calls()) != 0 {
		t.Fatalf("outcome = %+v, %d model calls", out, len(h.rec.Calls()))
	}
}
