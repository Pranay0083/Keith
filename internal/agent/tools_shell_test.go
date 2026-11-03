package agent_test

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// newToolHarness is a harness with the default tools.
func newToolHarness(t *testing.T) *harness {
	return newHarness(t, agent.DefaultTools(agent.ToolOptions{NewID: agenttest.IDs()})...)
}

// tool runs one call through the loop and returns its tool_result.
func (h *harness) tool(id, name string, input any) llm.Block {
	h.t.Helper()
	if len(h.st.Messages) > 0 {
		// A follow-up message starts the next turn.
		h.queue.Push(agent.Input{ID: "go-" + id, Text: "next"})
	}
	h.prov.Push(llmtest.ToolCall(id, name, input), llmtest.Text("ok"))
	out := h.runOnce()
	if out.Kind != agent.OutcomeEndTurn {
		h.t.Fatalf("%s: outcome = %+v", name, out)
	}
	calls := h.rec.Calls()
	res := agenttest.ToolResults(calls[len(calls)-1].Req)
	if len(res) != 1 || res[0].ToolUseID != id {
		h.t.Fatalf("%s: results = %+v", name, res)
	}
	return res[0]
}

// finished returns the tool_call.finished payload for a call.
func (h *harness) finished(id string) events.ToolCallFinishedPayload {
	h.t.Helper()
	for _, p := range agenttest.Payloads[events.ToolCallFinishedPayload](h.events, events.TypeToolCallFinished) {
		if p.CallID == id {
			return p
		}
	}
	h.t.Fatalf("no tool_call.finished for %s", id)
	return events.ToolCallFinishedPayload{}
}

func text(b llm.Block) string { return agenttest.BlockText(b) }

func wantErrorResult(t *testing.T, b llm.Block, sub string) {
	t.Helper()
	if !b.IsError || !strings.Contains(text(b), sub) {
		t.Fatalf("want error containing %q, got is_error=%v %q", sub, b.IsError, text(b))
	}
}

func TestCreateShell(t *testing.T) {
	h := newToolHarness(t)
	r := h.tool("c1", "create_shell", map[string]string{"name": "main"})
	if r.IsError || !strings.Contains(text(r), `Created shell "main"`) {
		t.Fatalf("result = %q", text(r))
	}
	sc := agenttest.Payloads[events.ShellCreatedPayload](h.events, events.TypeShellCreated)
	if len(sc) != 1 || sc[0].ShellID != "main" || sc[0].CallID != "c1" {
		t.Fatalf("shell.created = %+v", sc)
	}
	wantErrorResult(t, h.tool("c2", "create_shell", map[string]string{"name": "main"}), "already exists")
	wantErrorResult(t, h.tool("c3", "create_shell", map[string]string{"name": " "}), "empty")
	wantErrorResult(t, h.tool("c4", "create_shell", map[string]any{"name": "x", "extra": 1}), "invalid input")
}

func TestShellExec(t *testing.T) {
	h := newToolHarness(t)
	var outputs []string
	h.shells.OnOutput = func(shell string, chunk []byte) {
		// What the worker's tmuxshell.WithOutput callback sees.
		outputs = append(outputs, shell+"|"+h.run.CurrentCall(shell)+"|"+string(chunk))
	}
	h.shells.OnExec = func(name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
		switch cmd {
		case "make":
			return sandbox.ShellResult{OutputTail: "build ok\n", ExitCode: 0}, nil
		case "make test":
			return sandbox.ShellResult{OutputTail: "FAIL\n", ExitCode: 2}, nil
		case "npm run dev":
			return sandbox.ShellResult{OutputTail: "listening on 3000\n", ExitCode: -1, TimedOut: true}, nil
		case "busy":
			return sandbox.ShellResult{}, sandbox.ErrShellBusy
		}
		return sandbox.ShellResult{}, nil
	}

	// An unknown shell is created on first use; shell.created comes
	// before its output.
	r := h.tool("e1", "shell_exec", map[string]string{"shell": "main", "command": "make"})
	if r.IsError || !strings.Contains(text(r), "Exit code: 0") || !strings.Contains(text(r), "build ok") || !strings.Contains(text(r), `Created shell "main"`) {
		t.Fatalf("result = %q", text(r))
	}
	if got := h.events.Stored(); !slices.Equal(got[len(got)-4:len(got)-1], []string{events.TypeToolCallStarted, events.TypeShellCreated, events.TypeToolCallFinished}) {
		t.Fatalf("events = %v", got)
	}
	if !slices.Equal(outputs, []string{"main|e1|build ok\n"}) {
		t.Fatalf("outputs = %q", outputs)
	}
	if h.run.CurrentCall("main") != "" {
		t.Fatal("shell still bound after the call")
	}

	r = h.tool("e2", "shell_exec", map[string]string{"shell": "main", "command": "make test"})
	if !r.IsError || !strings.Contains(text(r), "Exit code: 2") {
		t.Fatalf("result = %q", text(r))
	}
	if f := h.finished("e2"); !f.IsError || f.ExitCode != 2 || f.Summary != "exit 2: make test" {
		t.Fatalf("finished = %+v", f)
	}

	r = h.tool("e3", "shell_exec", map[string]any{"shell": "main", "command": "npm run dev", "timeout_seconds": 5})
	if r.IsError || !strings.Contains(text(r), "still running") || !strings.Contains(text(r), "shell_read") || !strings.Contains(text(r), "listening on 3000") {
		t.Fatalf("result = %q", text(r))
	}
	if f := h.finished("e3"); !f.TimedOut || f.ExitCode != 0 || f.IsError {
		t.Fatalf("finished = %+v", f)
	}

	wantErrorResult(t, h.tool("e4", "shell_exec", map[string]string{"shell": "main", "command": "busy"}), "still running an earlier command")
	wantErrorResult(t, h.tool("e5", "shell_exec", map[string]string{"shell": "main", "command": " "}), "required")

	// Timeouts: default, requested, capped.
	h.tool("e6", "shell_exec", map[string]any{"shell": "main", "command": "x", "timeout_seconds": 100000})
	var timeouts []time.Duration
	for _, c := range h.shells.Calls() {
		if c.Op == "exec" && c.Cmd != "busy" {
			timeouts = append(timeouts, c.Timeout)
		}
	}
	want := []time.Duration{60 * time.Second, 60 * time.Second, 60 * time.Second, 5 * time.Second, 10 * time.Minute}
	if !slices.Equal(timeouts, want) {
		t.Fatalf("timeouts = %v, want %v", timeouts, want)
	}
}

func TestShellOutputIsCapped(t *testing.T) {
	h := newHarness(t, agent.DefaultTools(agent.ToolOptions{MaxOutputBytes: 100})...)
	h.shells.OnExec = func(name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
		return sandbox.ShellResult{OutputTail: strings.Repeat("a", 500) + "THE END"}, nil
	}
	r := h.tool("e1", "shell_exec", map[string]string{"shell": "main", "command": "x"})
	if got := text(r); len(got) > 200 || !strings.HasSuffix(got, "THE END") {
		t.Fatalf("result (%d bytes) = %q", len(got), got)
	}
}

func TestShellReadAndInterrupt(t *testing.T) {
	h := newToolHarness(t)
	var bound []string
	h.shells.OnOutput = func(shell string, chunk []byte) { bound = append(bound, h.run.CurrentCall(shell)) }
	wantErrorResult(t, h.tool("r0", "shell_read", map[string]string{"shell": "dev"}), `no shell named "dev"`)
	h.tool("c1", "create_shell", map[string]string{"name": "dev"})

	r := h.tool("r1", "shell_read", map[string]string{"shell": "dev"})
	if r.IsError || text(r) != "(no new output)" {
		t.Fatalf("empty read = %q", text(r))
	}
	h.shells.QueueRead("dev", "GET / 200\n")
	if r := h.tool("r2", "shell_read", map[string]string{"shell": "dev"}); text(r) != "GET / 200\n" {
		t.Fatalf("read = %q", text(r))
	}
	h.shells.QueueRead("dev", "^C\n")
	r = h.tool("i1", "shell_interrupt", map[string]string{"shell": "dev"})
	if r.IsError || !strings.Contains(text(r), "Sent Ctrl-C") || !strings.Contains(text(r), "^C") {
		t.Fatalf("interrupt = %q", text(r))
	}
	if !slices.Equal(bound, []string{"r2", "i1"}) {
		t.Fatalf("output bound to %v", bound)
	}
	var ops []string
	for _, c := range h.shells.Calls() {
		ops = append(ops, c.Op)
	}
	if !slices.Equal(ops, []string{"read", "create", "read", "read", "interrupt", "read"}) {
		t.Fatalf("ops = %v", ops)
	}
}

func TestShellToolsWithoutShells(t *testing.T) {
	h := newToolHarness(t)
	h.run.Shells = nil
	wantErrorResult(t, h.tool("c1", "create_shell", map[string]string{"name": "a"}), "not available")
	wantErrorResult(t, h.tool("c2", "shell_exec", map[string]string{"shell": "a", "command": "ls"}), "not available")
	wantErrorResult(t, h.tool("c3", "shell_read", map[string]string{"shell": "a"}), "not available")
	wantErrorResult(t, h.tool("c4", "shell_interrupt", map[string]string{"shell": "a"}), "not available")
}
