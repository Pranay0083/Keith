package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
)

const nudge = "[keith] This exact call has now failed"

func TestRepeatNudgesThenPauses(t *testing.T) {
	h := newHarness(t, failTool, echoTool)
	for i := 1; i <= 5; i++ {
		h.prov.Push(llmtest.ToolCalls(
			call(fmt.Sprintf("f%d", i), "fail", map[string]string{"x": "same"}),
			// A different call in between does not reset the count.
			call(fmt.Sprintf("e%d", i), "echo", map[string]string{}),
		))
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if out.Question == nil || out.Question.ID != "stuck_f5" || !out.Question.Auto || out.Question.ToolUseID != "" ||
		!strings.Contains(out.Question.Text, "failed 5 times in a row") || !strings.Contains(out.Question.Text, "boom after") {
		t.Fatalf("question = %+v", out.Question)
	}
	calls := h.rec.Calls()
	for i, c := range calls[1:] {
		res := agenttest.ToolResults(c.Req)[0]
		got := strings.Contains(agenttest.BlockText(res), nudge)
		if want := i+1 >= 3; got != want {
			t.Errorf("failure %d: nudge = %v, want %v", i+1, got, want)
		}
	}
	held := h.st.Aux.Pending.Results
	if len(held) != 2 || !strings.Contains(agenttest.BlockText(held[0]), "session is paused") {
		t.Fatalf("held = %s", mustJSON(held))
	}
	qa := agenttest.Payloads[events.QuestionAskedPayload](h.events, events.TypeQuestionAsked)
	if len(qa) != 1 || qa[0].QuestionID != "stuck_f5" {
		t.Fatalf("question.asked = %+v", qa)
	}
	if got := h.events.Stored(); got[len(got)-1] != events.TypeQuestionAsked {
		t.Fatalf("question.asked is not last: %v", got)
	}

	// The answer goes in as text after the held results, and the count
	// starts over.
	h.reload()
	h.queue.Push(agent.Input{ID: "m1", Text: "the API is down, skip it", QuestionID: "stuck_f5"})
	h.prov.Push(llmtest.ToolCall("f6", "fail", map[string]string{"x": "same"}), llmtest.Text("skipping"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	calls = h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	um := calls[5].Req.Messages[len(calls[5].Req.Messages)-1]
	if len(um.Content) != 3 || um.Content[2].Text != "the API is down, skip it" {
		t.Fatalf("resume turn = %s", mustJSON(um))
	}
	if strings.Contains(agenttest.BlockText(agenttest.ToolResults(calls[6].Req)...), nudge) {
		t.Fatal("count did not start over after the pause")
	}
}

func TestRepeatResetsOnSuccessOrNewError(t *testing.T) {
	n := 0
	flaky := fakeTool{name: "flaky", fn: func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
		n++
		switch n {
		case 3: // success resets
			return agent.Result{Content: []llm.Block{llm.TextBlock("ok")}}, nil
		case 6: // a different error resets
			return agent.Result{IsError: true, Content: []llm.Block{llm.TextBlock("other error")}}, nil
		}
		return agent.Result{IsError: true, Content: []llm.Block{llm.TextBlock("same error")}}, nil
	}}
	h := newHarness(t, flaky)
	for i := 1; i <= 9; i++ {
		h.prov.Push(llmtest.ToolCall(fmt.Sprintf("c%d", i), "flaky", map[string]string{}))
	}
	h.prov.Push(llmtest.Text("done"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	calls := h.rec.Calls()
	// same same | ok | same same | other | same same same: counts
	// 1 2 - 1 2 1 1 2 3, so only the ninth call is nudged.
	wantNudge := map[int]bool{9: true}
	for i := 1; i <= 9; i++ {
		got := strings.Contains(agenttest.BlockText(agenttest.ToolResults(calls[i].Req)...), nudge)
		if got != wantNudge[i] {
			t.Errorf("call %d: nudge = %v", i, got)
		}
	}
}

func TestRepeatedTestCommandFailures(t *testing.T) {
	n := 0
	shell := fakeTool{name: "shell_exec", fn: func(ctx context.Context, run *agent.Run, input json.RawMessage) (agent.Result, error) {
		n++
		var in struct {
			Command string `json:"command"`
		}
		_ = json.Unmarshal(input, &in)
		if strings.Contains(strings.Join(strings.Fields(in.Command), " "), "go test") && n != 3 {
			// Each run fails differently: the identical-call rule alone
			// would never fire.
			return agent.Result{IsError: true, ExitCode: 1, Content: []llm.Block{llm.TextBlock(fmt.Sprintf("FAIL TestThing%c", 'A'+n))}}, nil
		}
		return agent.Result{Content: []llm.Block{llm.TextBlock("PASS")}}, nil
	}}
	h := newHarness(t, shell, echoTool)
	cmds := []string{
		"cd /workspace/api && go test ./...",                   // fail 1
		"cd /workspace/api && go test ./...",                   // fail 2
		"cd /workspace/api && go test ./...",                   // pass: reset
		"cd /workspace/api &&  go  test ./... 2>&1 | tail -50", // fail 1
		"cd /workspace/api && go test ./...",                   // fail 2
		"go vet ./...",                                         // not a test run
		"cd /workspace/api && go test ./...",                   // fail 3: nudge
		"cd /workspace/api && go test ./...",                   // fail 4: nudge
		"cd /workspace/api && go test ./...",                   // fail 5: pause
	}
	for i, c := range cmds {
		h.prov.Push(llmtest.ToolCall(fmt.Sprintf("s%d", i), "shell_exec", map[string]string{"command": c}))
	}
	out := h.runOnce()
	wantKind(t, out, agent.OutcomePaused)
	if !strings.Contains(out.Question.Text, "test command") || !strings.Contains(out.Question.Text, "5 runs in a row") {
		t.Fatalf("question = %q", out.Question.Text)
	}
	calls := h.rec.Calls()
	for i := 1; i < len(calls); i++ {
		got := strings.Contains(agenttest.BlockText(agenttest.ToolResults(calls[i].Req)...), "test command has now failed")
		if want := i == 7 || i == 8; got != want {
			t.Errorf("result %d: nudge = %v, want %v", i, got, want)
		}
	}
}

func TestRepeatDetectionCanBeDisabled(t *testing.T) {
	h := newHarness(t, failTool)
	h.cfg.Repeat = agent.RepeatPolicy{NudgeAfter: -1, PauseAfter: -1}
	for i := 1; i <= 6; i++ {
		h.prov.Push(llmtest.ToolCall(fmt.Sprintf("f%d", i), "fail", map[string]string{}))
	}
	h.prov.Push(llmtest.Text("gave up"))
	wantKind(t, h.runOnce(), agent.OutcomeEndTurn)
	for _, c := range h.rec.Calls()[1:] {
		if strings.Contains(agenttest.BlockText(agenttest.ToolResults(c.Req)...), "[keith]") {
			t.Fatal("nudged while disabled")
		}
	}
}
