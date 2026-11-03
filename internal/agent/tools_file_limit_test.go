package agent_test

import (
	"context"
	"io/fs"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

// specialFiles is a memsandbox in which some paths are not regular files
// (a /dev/zero symlink, a FIFO): reads of them fail the way the real
// providers fail them.
type specialFiles struct {
	*memsandbox.Sandbox
	special map[string]bool
}

func (s specialFiles) ReadFile(ctx context.Context, p string) ([]byte, error) {
	return s.ReadFileLimit(ctx, p, sandbox.MaxReadFile)
}

func (s specialFiles) ReadFileLimit(ctx context.Context, p string, limit int64) ([]byte, error) {
	if s.special[p] {
		return nil, &fs.PathError{Op: "read", Path: p, Err: sandbox.ErrNotRegular}
	}
	return s.Sandbox.ReadFileLimit(ctx, p, limit)
}

// Regression: loadFile read the whole file and checked its size after, so
// read_file on a symlink to /dev/zero (or any huge file) could exhaust the
// worker's memory. It now asks the sandbox for at most 8 MiB, and every
// file tool turns a refusal into a model-visible error.
func TestFileToolsRefuseSpecialAndOversizedFiles(t *testing.T) {
	h := newToolHarness(t)
	huge := strings.Repeat("x", 8<<20+1)
	h.seed(map[string]string{"api/huge.txt": huge, "api/ok.txt": "fine\n"})
	h.run.Sandbox = specialFiles{Sandbox: h.sb, special: map[string]bool{"/workspace/api/zero": true}}

	for i, tc := range []struct {
		tool  string
		input map[string]string
	}{
		{"read_file", map[string]string{}},
		{"write_file", map[string]string{"content": "x"}},
		{"edit_file", map[string]string{"old_string": "x", "new_string": "y"}},
		{"delete_file", map[string]string{}},
	} {
		for _, f := range []struct{ path, want string }{
			{"api/huge.txt", "is too large for this tool (over 8 MiB)"},
			{"api/zero", "is not a regular file"},
		} {
			in := map[string]string{"path": f.path}
			for k, v := range tc.input {
				in[k] = v
			}
			wantErrorResult(t, h.tool(tc.tool+"-"+f.path[4:]+"-"+string(rune('a'+i)), tc.tool, in), f.want)
		}
	}
	if got, ok := h.file("/workspace/api/huge.txt"); !ok || len(got) != len(huge) {
		t.Fatalf("huge.txt changed: %d bytes, exists %v", len(got), ok)
	}
	if r := h.tool("ok", "read_file", map[string]string{"path": "api/ok.txt"}); r.IsError || !strings.Contains(text(r), "fine") {
		t.Fatalf("read ok.txt = %q", text(r))
	}

	reads := 0
	for _, c := range h.sb.Calls() {
		if c.Op != memsandbox.OpRead {
			continue
		}
		reads++
		if c.Limit != 8<<20 {
			t.Errorf("read of %s asked for %d bytes; want the 8 MiB tool limit", c.Path, c.Limit)
		}
	}
	if reads == 0 {
		t.Fatal("no reads recorded")
	}
}
