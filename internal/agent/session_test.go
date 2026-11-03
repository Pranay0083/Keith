package agent_test

import (
	"encoding/json"
	"regexp"
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
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

var v1Tools = []string{
	"create_shell", "shell_exec", "shell_read", "shell_interrupt",
	"read_file", "write_file", "edit_file", "delete_file", "undo_edit",
	"search", "read_issue", "message_user", "ask_user", "open_pr",
}

// unsupported are JSON Schema keywords strict tool use rejects.
var unsupported = []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
	"minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "$ref", "default"}

func TestToolSpecsAreStrictSchemas(t *testing.T) {
	tools := agent.DefaultTools(agent.ToolOptions{})
	var names []string
	for _, tool := range tools {
		s := tool.Spec()
		names = append(names, s.Name)
		if !s.Strict || s.Description == "" || !regexp.MustCompile(`^[a-z][a-z_]*$`).MatchString(s.Name) {
			t.Errorf("%s: strict=%v description=%q", s.Name, s.Strict, s.Description)
		}
		var root any
		if err := json.Unmarshal(s.InputSchema, &root); err != nil {
			t.Errorf("%s: schema is not JSON: %v", s.Name, err)
			continue
		}
		checkSchema(t, s.Name, "$", root)
		if root.(map[string]any)["type"] != "object" {
			t.Errorf("%s: root is not an object", s.Name)
		}
	}
	if !slices.Equal(names, v1Tools) {
		t.Fatalf("tools = %v\nwant %v", names, v1Tools)
	}
	// Specs are byte-stable: they are part of the cached prefix.
	again := agent.DefaultTools(agent.ToolOptions{})
	for i := range tools {
		if mustJSON(tools[i].Spec()) != mustJSON(again[i].Spec()) {
			t.Errorf("%s: spec is not deterministic", names[i])
		}
	}
	// And they work as a Runner's tool list.
	r, err := agent.NewRunner(agent.Config{Provider: llmtest.New(), Model: "m", Tools: tools})
	if err != nil || len(r.ToolSpecs()) != len(v1Tools) {
		t.Fatalf("NewRunner: %v", err)
	}
}

func checkSchema(t *testing.T, tool, at string, node any) {
	t.Helper()
	m, ok := node.(map[string]any)
	if !ok {
		t.Errorf("%s %s: schema node is %T", tool, at, node)
		return
	}
	for _, k := range unsupported {
		if _, has := m[k]; has {
			t.Errorf("%s %s: unsupported keyword %q", tool, at, k)
		}
	}
	switch typ := m["type"]; typ {
	case "object":
		if m["additionalProperties"] != false {
			t.Errorf("%s %s: additionalProperties is %v, want false", tool, at, m["additionalProperties"])
		}
		props, ok := m["properties"].(map[string]any)
		if !ok {
			t.Errorf("%s %s: no properties", tool, at)
			return
		}
		req, ok := m["required"].([]any)
		if !ok {
			t.Errorf("%s %s: required is %T, want an array", tool, at, m["required"])
		}
		for _, r := range req {
			if _, ok := props[r.(string)]; !ok {
				t.Errorf("%s %s: required %q is not a property", tool, at, r)
			}
		}
		for name, p := range props {
			if d, _ := p.(map[string]any)["description"].(string); d == "" {
				t.Errorf("%s %s.%s: no description", tool, at, name)
			}
			checkSchema(t, tool, at+"."+name, p)
		}
	case "array":
		checkSchema(t, tool, at+"[]", m["items"])
	case "string", "integer", "number", "boolean":
	default:
		t.Errorf("%s %s: type %v", tool, at, typ)
	}
}

// TestFullSession drives the default tools through a realistic session:
// explore, run the tests, edit, re-run, open a PR and finish. It checks
// the event log the UI renders from and the append-only conversation.
func TestFullSession(t *testing.T) {
	h := newToolHarness(t)
	h.st = agent.NewState(agent.BuildSystemPrompt(agent.PromptInput{Task: "Fix the pager", Repos: h.run.Repos}))
	h.seed(map[string]string{"api/pager.go": "package api\n\nfunc last(n int) int { return n }\n"})
	tests := 0
	h.shells.OnExec = func(name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
		if strings.Contains(cmd, "go test") {
			tests++
			if tests == 1 {
				return sandbox.ShellResult{OutputTail: "--- FAIL: TestLast\n", ExitCode: 1}, nil
			}
			return sandbox.ShellResult{OutputTail: "ok  acme/api 0.01s\n"}, nil
		}
		return sandbox.ShellResult{}, nil
	}
	h.sb.SetHandler(memsandbox.Route(
		memsandbox.On(memsandbox.Prefix("{ rg "), rgOut("api/pager.go:3:func last(n int) int { return n }\n", 0)),
		anyRule(gitRepo{head: "agent/s1", status: " M pager.go\n", ahead: "0"}.handler()),
	))
	var outputs []string
	h.shells.OnOutput = func(shell string, chunk []byte) { outputs = append(outputs, h.run.CurrentCall(shell)) }
	h.queue.Push(agent.Input{ID: "msg-prompt", UserID: "u1", Text: "Fix the pager"})

	h.prov.Push(
		llmtest.ToolCalls(llm.TextBlock("Let me look."), call("t1", "search", map[string]string{"pattern": "func last"}), call("t2", "read_file", map[string]string{"path": "api/pager.go"})),
		llmtest.ToolCalls(call("t3", "create_shell", map[string]string{"name": "main"}), call("t4", "shell_exec", map[string]string{"shell": "main", "command": "cd /workspace/api && go test ./..."})),
		llmtest.ToolCall("t5", "edit_file", map[string]string{"path": "api/pager.go", "old_string": "return n }", "new_string": "return n - 1 }"}),
		llmtest.ToolCalls(call("t6", "message_user", map[string]string{"text": "Found it: off by one."}), call("t7", "shell_exec", map[string]string{"shell": "main", "command": "cd /workspace/api && go test ./..."})),
		llmtest.ToolCall("t8", "open_pr", map[string]string{"title": "Fix pager off-by-one", "body": "Tested with go test."}),
		llmtest.Text("Fixed the off-by-one in `last` and opened https://github.com/acme/api/pull/100."),
	)
	out := h.runOnce()
	wantKind(t, out, agent.OutcomeEndTurn)
	if !strings.Contains(out.Text, "pull/100") || out.Steps != 6 {
		t.Fatalf("outcome = %+v", out)
	}
	if got, _ := h.file("api/pager.go"); !strings.Contains(got, "return n - 1 }") {
		t.Fatalf("file = %q", got)
	}
	calls := h.rec.Calls()
	agenttest.CheckAppendOnly(t, calls)
	for _, c := range calls {
		if c.Req.System != h.st.System || !strings.Contains(c.Req.System, "Fix the pager") {
			t.Fatal("system prompt changed or missing")
		}
	}
	for i, r := range agenttest.ToolResults(calls[2].Req) {
		if i == 1 && (!r.IsError || !strings.Contains(text(r), "Exit code: 1")) {
			t.Fatalf("first test run = %q", text(r))
		}
	}

	want := []string{
		"message.user",
		"message.assistant",
		"tool_call.started", "tool_call.finished", // search
		"tool_call.started", "tool_call.finished", // read_file
		"tool_call.started", "shell.created", "tool_call.finished", // create_shell
		"tool_call.started", "tool_call.finished", // go test (fails)
		"tool_call.started", "file.changed", "tool_call.finished", // edit_file
		"tool_call.started", "message.assistant", "tool_call.finished", // message_user
		"tool_call.started", "tool_call.finished", // go test (passes)
		"tool_call.started", "pr.opened", "tool_call.finished", // open_pr
		"message.assistant",
	}
	if got := h.events.Stored(); !slices.Equal(got, want) {
		t.Fatalf("events =\n%v\nwant\n%v", got, want)
	}
	if !slices.Equal(outputs, []string{"t4", "t7"}) {
		t.Fatalf("shell output bound to %v", outputs)
	}
	if n := len(agenttest.Payloads[events.MessageDeltaPayload](h.events, events.TypeMessageDelta)); n != 2 {
		t.Fatalf("%d deltas", n)
	}
}

func anyRule(h memsandbox.Handler) memsandbox.RouteRule { return memsandbox.On(memsandbox.Any(), h) }

// TestEveryPayloadType checks that the loop and tools emit each event type
// they own with its events.*Payload struct (the Recorder checks types and
// JSON round trips on every Emit).
func TestEveryPayloadType(t *testing.T) {
	h := newToolHarness(t)
	h.sb.SetHandler(gitRepo{head: "agent/s1", status: " M a\n", ahead: "0"}.handler())
	h.queue.Push(agent.Input{ID: "m0", Text: "go"})
	h.prov.Push(
		llmtest.ToolCalls(llm.TextBlock("hi"),
			call("a", "create_shell", map[string]string{"name": "main"}),
			call("b", "write_file", map[string]string{"path": "api/a", "content": "x"}),
			call("c", "message_user", map[string]string{"text": "note"}),
			call("d", "open_pr", prInput),
			call("e", "ask_user", map[string]string{"question": "ok?"})),
	)
	wantKind(t, h.runOnce(), agent.OutcomePaused)
	seen := map[string]bool{}
	for _, typ := range h.events.Types() {
		seen[typ] = true
	}
	for _, typ := range []string{
		events.TypeMessageUser, events.TypeMessageAssistant, events.TypeMessageDelta,
		events.TypeToolCallStarted, events.TypeToolCallFinished, events.TypeShellCreated,
		events.TypeFileChanged, events.TypeQuestionAsked, events.TypePROpened,
	} {
		if !seen[typ] {
			t.Errorf("%s never emitted", typ)
		}
	}
	// The worker owns the rest.
	for _, typ := range []string{events.TypeSessionStatus, events.TypeToolCallOutput, events.TypeError} {
		if seen[typ] {
			t.Errorf("agent emitted %s", typ)
		}
	}
}
