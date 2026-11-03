package agent_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

func (h *harness) seed(files map[string]string) {
	for p, c := range files {
		if err := h.sb.WriteFile(context.Background(), p, []byte(c), 0o644); err != nil {
			h.t.Fatal(err)
		}
	}
	h.sb.ResetCalls()
}

func (h *harness) file(p string) (string, bool) {
	b, ok := h.sb.File(p)
	return string(b), ok
}

func TestPathTraversalIsRejected(t *testing.T) {
	h := newToolHarness(t)
	h.seed(map[string]string{"/etc/passwd": "root"})
	bad := []string{"../etc/passwd", "/etc/passwd", "/workspace/../etc/passwd", "api/../../etc/passwd", "/workspacex/a", ""}
	for i, p := range bad {
		for _, tc := range []struct {
			tool  string
			input map[string]string
		}{
			{"read_file", map[string]string{"path": p}},
			{"write_file", map[string]string{"path": p, "content": "x"}},
			{"edit_file", map[string]string{"path": p, "old_string": "root", "new_string": "x"}},
			{"delete_file", map[string]string{"path": p}},
			{"undo_edit", map[string]string{"path": p}},
		} {
			r := h.tool(fmt.Sprintf("%s-%d", tc.tool, i), tc.tool, tc.input)
			if !r.IsError {
				t.Errorf("%s %q: not rejected: %q", tc.tool, p, text(r))
			}
		}
	}
	if got, _ := h.file("/etc/passwd"); got != "root" {
		t.Fatal("file outside the workspace changed")
	}
	for _, c := range h.sb.Calls() {
		if !strings.HasPrefix(c.Path, "/workspace/") && c.Op != memsandbox.OpExec {
			t.Errorf("sandbox call outside the workspace: %+v", c)
		}
	}
	// Writes into .git are refused too.
	wantErrorResult(t, h.tool("g", "write_file", map[string]string{"path": "api/.git/config", "content": "x"}), ".git")
}

func TestReadFile(t *testing.T) {
	h := newToolHarness(t)
	var long []string
	for i := 1; i <= 2500; i++ {
		long = append(long, fmt.Sprintf("line %d", i))
	}
	h.seed(map[string]string{
		"api/main.go": "package main\n\nfunc main() {}\n",
		"api/big.txt": strings.Join(long, "\n") + "\n",
		"api/bin":     "\x00\x01\x02",
		"api/empty":   "",
		"api/wide":    strings.Repeat("x", 5000),
	})

	r := h.tool("r1", "read_file", map[string]string{"path": "api/main.go"})
	if r.IsError || text(r) != "     1\tpackage main\n     2\t\n     3\tfunc main() {}\n" {
		t.Fatalf("whole file = %q", text(r))
	}
	r = h.tool("r2", "read_file", map[string]any{"path": "/workspace/api/main.go", "start_line": 3, "end_line": 9})
	if text(r) != "     3\tfunc main() {}\n\n(Showing lines 3-3 of 3.)" {
		t.Fatalf("range = %q", text(r))
	}
	r = h.tool("r3", "read_file", map[string]string{"path": "api/big.txt"})
	if !strings.Contains(text(r), "  2000\tline 2000\n") || strings.Contains(text(r), "line 2001") ||
		!strings.Contains(text(r), "Showing lines 1-2000 of 2500. Use start_line=2001 to read more.") {
		t.Fatalf("big = ...%q", text(r)[len(text(r))-200:])
	}
	r = h.tool("r4", "read_file", map[string]any{"path": "api/big.txt", "start_line": 2499})
	if !strings.HasPrefix(text(r), "  2499\tline 2499\n  2500\tline 2500\n") {
		t.Fatalf("tail = %q", text(r))
	}
	if r := h.tool("r5", "read_file", map[string]string{"path": "api/bin"}); !strings.Contains(text(r), "binary file (3 bytes)") {
		t.Fatalf("binary = %q", text(r))
	}
	if r := h.tool("r6", "read_file", map[string]string{"path": "api/empty"}); r.IsError || !strings.Contains(text(r), "is empty") {
		t.Fatalf("empty = %q", text(r))
	}
	if r := h.tool("r7", "read_file", map[string]string{"path": "api/wide"}); !strings.Contains(text(r), "(line truncated)") || len(text(r)) > 2100 {
		t.Fatalf("wide = %d bytes", len(text(r)))
	}
	wantErrorResult(t, h.tool("r8", "read_file", map[string]string{"path": "api/nope.go"}), "No such file: /workspace/api/nope.go")
	wantErrorResult(t, h.tool("r9", "read_file", map[string]any{"path": "api/main.go", "start_line": 10}), "past the end")
	wantErrorResult(t, h.tool("r10", "read_file", map[string]any{"path": "api/main.go", "start_line": 3, "end_line": 2}), "before start_line")
}

func TestReadFileOutputCap(t *testing.T) {
	h := newHarness(t, agent.DefaultTools(agent.ToolOptions{MaxReadBytes: 100})...)
	h.seed(map[string]string{"a.txt": strings.Repeat("0123456789\n", 50)})
	r := h.tool("r1", "read_file", map[string]string{"path": "a.txt"})
	if len(text(r)) > 200 || !strings.Contains(text(r), "Use start_line=") {
		t.Fatalf("capped read = %q", text(r))
	}
}

func TestWriteFile(t *testing.T) {
	h := newToolHarness(t)
	var changes []agent.FileChange
	var storedAtHook []int
	h.run.AfterFileChange = func(ctx context.Context, c agent.FileChange) error {
		changes = append(changes, c)
		storedAtHook = append(storedAtHook, len(agenttest.Payloads[events.FileChangedPayload](h.events, events.TypeFileChanged)))
		return nil
	}
	r := h.tool("w1", "write_file", map[string]string{"path": "api/new/x.go", "content": "a\nb\n"})
	if r.IsError || text(r) != "Created /workspace/api/new/x.go (2 lines)." {
		t.Fatalf("create = %q", text(r))
	}
	if got, _ := h.file("/workspace/api/new/x.go"); got != "a\nb\n" {
		t.Fatalf("content = %q", got)
	}
	h.tool("w2", "write_file", map[string]string{"path": "api/new/x.go", "content": "c\n"})
	r = h.tool("w3", "write_file", map[string]string{"path": "api/new/x.go", "content": "c\n"})
	if !strings.Contains(text(r), "nothing changed") {
		t.Fatalf("unchanged = %q", text(r))
	}
	h.tool("w4", "write_file", map[string]string{"path": "notes.txt", "content": "outside repos"})

	fc := agenttest.Payloads[events.FileChangedPayload](h.events, events.TypeFileChanged)
	want := []events.FileChangedPayload{
		{CallID: "w1", Repo: "acme/api", Path: "/workspace/api/new/x.go", Action: events.FileCreated},
		{CallID: "w2", Repo: "acme/api", Path: "/workspace/api/new/x.go", Action: events.FileModified},
		{CallID: "w4", Path: "/workspace/notes.txt", Action: events.FileCreated},
	}
	if !slices.Equal(fc, want) {
		t.Fatalf("file.changed = %+v", fc)
	}
	// The diff hook runs before each file.changed.
	if !slices.Equal(storedAtHook, []int{0, 1, 2}) || changes[0].Tool != "write_file" || changes[0].Repo.FullName != "acme/api" || changes[2].Repo.FullName != "" {
		t.Fatalf("hook: at %v changes %+v", storedAtHook, changes)
	}
}

func TestFileHookErrorStopsTheRun(t *testing.T) {
	h := newToolHarness(t)
	h.run.AfterFileChange = func(context.Context, agent.FileChange) error { return errors.New("diff store down") }
	h.prov.Push(llmtest.ToolCall("w1", "write_file", map[string]string{"path": "a", "content": "x"}))
	out := h.runOnce()
	if out.Kind != agent.OutcomeFailed || out.Reason != agent.ReasonToolError {
		t.Fatalf("outcome = %+v", out)
	}
	if n := len(agenttest.Payloads[events.FileChangedPayload](h.events, events.TypeFileChanged)); n != 0 {
		t.Fatal("file.changed emitted although the diff was not saved")
	}
}

func TestWritePreservesMode(t *testing.T) {
	h := newToolHarness(t)
	h.seed(map[string]string{"api/run.sh": "#!/bin/sh\necho hi\n"})
	h.sb.SetHandler(memsandbox.Route(memsandbox.On(memsandbox.Prefix("stat -c %a -- '/workspace/api/run.sh'"), memsandbox.OK("755\n"))))
	h.tool("e1", "edit_file", map[string]string{"path": "api/run.sh", "old_string": "hi", "new_string": "hello"})
	if m, _ := h.sb.FileMode("api/run.sh"); m != 0o755 {
		t.Fatalf("mode = %v", m)
	}
	h.tool("d1", "delete_file", map[string]string{"path": "api/run.sh"})
	h.tool("u1", "undo_edit", map[string]string{"path": "api/run.sh"})
	if m, ok := h.sb.FileMode("api/run.sh"); !ok || m != 0o755 {
		t.Fatalf("restored mode = %v %v", m, ok)
	}
}

func TestEditFileMatches(t *testing.T) {
	h := newToolHarness(t)
	h.seed(map[string]string{"api/a.go": "func a() {\n\treturn 1\n}\n\nfunc b() {\n\treturn 1\n}\n"})

	// Zero matches, with a whitespace hint.
	r := h.tool("e0", "edit_file", map[string]string{"path": "api/a.go", "old_string": "func a() {\n    return 1", "new_string": "x"})
	wantErrorResult(t, r, "0 matches")
	if !strings.Contains(text(r), "whitespace is ignored") {
		t.Fatalf("no whitespace hint: %q", text(r))
	}
	wantErrorResult(t, h.tool("e0b", "edit_file", map[string]string{"path": "api/a.go", "old_string": "nothing like this", "new_string": "x"}), "0 matches")

	// Two matches, with their lines.
	r = h.tool("e2", "edit_file", map[string]string{"path": "api/a.go", "old_string": "\treturn 1", "new_string": "\treturn 2"})
	wantErrorResult(t, r, "matches 2 times")
	if !strings.Contains(text(r), "at lines 2, 6") {
		t.Fatalf("lines: %q", text(r))
	}
	if got, _ := h.file("api/a.go"); strings.Contains(got, "return 2") {
		t.Fatal("ambiguous edit changed the file")
	}

	// One match.
	r = h.tool("e1", "edit_file", map[string]string{"path": "api/a.go", "old_string": "func b() {\n\treturn 1", "new_string": "func b() {\n\treturn 2"})
	if r.IsError || !strings.Contains(text(r), "replaced 1 match at line 5") || !strings.Contains(text(r), "     6\t\treturn 2\n") {
		t.Fatalf("edit = %q", text(r))
	}
	if got, _ := h.file("api/a.go"); got != "func a() {\n\treturn 1\n}\n\nfunc b() {\n\treturn 2\n}\n" {
		t.Fatalf("content = %q", got)
	}
	if f := h.finished("e1"); f.Summary != "edited /workspace/api/a.go:5" {
		t.Fatalf("summary = %q", f.Summary)
	}

	wantErrorResult(t, h.tool("e3", "edit_file", map[string]string{"path": "api/missing.go", "old_string": "a", "new_string": "b"}), "Use write_file to create it")
	wantErrorResult(t, h.tool("e4", "edit_file", map[string]string{"path": "api/a.go", "old_string": "", "new_string": "b"}), "old_string is empty")
	wantErrorResult(t, h.tool("e5", "edit_file", map[string]string{"path": "api/a.go", "old_string": "a", "new_string": "a"}), "the same")
}

func TestUndoEdit(t *testing.T) {
	h := newToolHarness(t)
	h.seed(map[string]string{"api/a.txt": "v1\n"})
	wantErrorResult(t, h.tool("u0", "undo_edit", map[string]string{"path": "api/a.txt"}), "no edit")

	h.tool("e1", "edit_file", map[string]string{"path": "api/a.txt", "old_string": "v1", "new_string": "v2"})
	h.tool("w1", "write_file", map[string]string{"path": "api/a.txt", "content": "v3\n"})
	h.tool("d1", "delete_file", map[string]string{"path": "api/a.txt"})
	if _, ok := h.file("api/a.txt"); ok {
		t.Fatal("not deleted")
	}

	steps := []struct{ id, want, msg string }{
		{"u1", "v3\n", "restored"},
		{"u2", "v2\n", "1 edit earlier left"},
		{"u3", "v1\n", "Reverted the last edit_file"},
	}
	for _, s := range steps {
		r := h.tool(s.id, "undo_edit", map[string]string{"path": "/workspace/api/a.txt"})
		if got, _ := h.file("api/a.txt"); r.IsError || got != s.want || !strings.Contains(text(r), s.msg) {
			t.Fatalf("%s: %q, file %q", s.id, text(r), got)
		}
	}
	wantErrorResult(t, h.tool("u4", "undo_edit", map[string]string{"path": "api/a.txt"}), "no edit")

	// Undoing a creation removes the file.
	h.tool("w2", "write_file", map[string]string{"path": "api/b.txt", "content": "new"})
	r := h.tool("u5", "undo_edit", map[string]string{"path": "api/b.txt"})
	if _, ok := h.file("api/b.txt"); ok || !strings.Contains(text(r), "did not exist before") {
		t.Fatalf("undo create: %q", text(r))
	}
	fc := agenttest.Payloads[events.FileChangedPayload](h.events, events.TypeFileChanged)
	var actions []events.FileAction
	for _, p := range fc {
		actions = append(actions, p.Action)
	}
	want := []events.FileAction{"modified", "modified", "deleted", "created", "modified", "modified", "created", "deleted"}
	if !slices.Equal(actions, want) {
		t.Fatalf("actions = %v", actions)
	}

	// A change made outside the file tools blocks undo.
	h.tool("w3", "write_file", map[string]string{"path": "api/c.txt", "content": "mine"})
	if err := h.sb.WriteFile(context.Background(), "api/c.txt", []byte("shell edit"), fs.FileMode(0o644)); err != nil {
		t.Fatal(err)
	}
	wantErrorResult(t, h.tool("u6", "undo_edit", map[string]string{"path": "api/c.txt"}), "changed after your last write_file")
	if got, _ := h.file("api/c.txt"); got != "shell edit" {
		t.Fatal("undo clobbered an outside change")
	}
}

func TestUndoSurvivesPauseAndReload(t *testing.T) {
	h := newToolHarness(t)
	h.seed(map[string]string{"api/a.txt": "old"})
	h.prov.Push(llmtest.ToolCalls(
		call("e1", "write_file", map[string]string{"path": "api/a.txt", "content": "new"}),
		call("q1", "ask_user", map[string]string{"question": "keep it?"}),
	))
	if out := h.runOnce(); out.Kind != agent.OutcomePaused {
		t.Fatalf("outcome = %+v", out)
	}
	h.reload()
	h.queue.Push(agent.Input{ID: "a", Text: "no", QuestionID: "q1"})
	h.prov.Push(llmtest.ToolCall("u1", "undo_edit", map[string]string{"path": "api/a.txt"}), llmtest.Text("reverted"))
	if out := h.runOnce(); out.Kind != agent.OutcomeEndTurn {
		t.Fatalf("outcome = %+v", out)
	}
	if got, _ := h.file("api/a.txt"); got != "old" {
		t.Fatalf("content = %q", got)
	}
}
