package agent

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path"
	"strings"
	"unicode/utf8"

	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// Tool names.
const (
	toolCreateShell    = "create_shell"
	toolShellExec      = "shell_exec"
	toolShellRead      = "shell_read"
	toolShellInterrupt = "shell_interrupt"
	toolReadFile       = "read_file"
	toolWriteFile      = "write_file"
	toolEditFile       = "edit_file"
	toolDeleteFile     = "delete_file"
	toolUndoEdit       = "undo_edit"
	toolSearch         = "search"
	toolMessageUser    = "message_user"
	toolAskUser        = "ask_user"
	toolReadIssue      = "read_issue"
	toolOpenPR         = "open_pr"
)

// errorResult is a tool result the model sees as an error.
func errorResult(msg string) Result {
	return Result{IsError: true, Content: []llm.Block{llm.TextBlock(msg)}, Summary: oneLine(msg, 120)}
}

// textResult is a successful tool result.
func textResult(summary, text string) Result {
	return Result{Content: []llm.Block{llm.TextBlock(text)}, Summary: summary}
}

// decodeInput unmarshals a tool's input strictly: unknown fields and
// trailing data are errors the model sees.
func decodeInput(input json.RawMessage, v any) error {
	dec := json.NewDecoder(bytes.NewReader(input))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid input: %v", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return errors.New("invalid input: trailing data after the JSON object")
	}
	return nil
}

// badInput is the result for input that failed decodeInput or a check.
func badInput(err error) Result {
	return errorResult(err.Error())
}

// ---- JSON schemas ----

// schema builds a strict tool input schema: an object with the given
// properties, the given required ones and no others. Keys are emitted in
// sorted order, so the spec bytes are stable (they are part of the cached
// prompt prefix).
func schema(props map[string]any, required ...string) json.RawMessage {
	if required == nil {
		required = []string{}
	}
	b, err := json.Marshal(map[string]any{
		"type":                 "object",
		"properties":           props,
		"required":             required,
		"additionalProperties": false,
	})
	if err != nil {
		panic(fmt.Sprintf("agent: tool schema: %v", err))
	}
	return b
}

func strProp(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}

func intProp(desc string) map[string]any {
	return map[string]any{"type": "integer", "description": desc}
}

func boolProp(desc string) map[string]any {
	return map[string]any{"type": "boolean", "description": desc}
}

func strArrayProp(desc string) map[string]any {
	return map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": desc}
}

func spec(name, desc string, props map[string]any, required ...string) llm.ToolSpec {
	return llm.ToolSpec{Name: name, Description: desc, InputSchema: schema(props, required...), Strict: true}
}

// ---- paths ----

// resolvePath turns a model-supplied path into a clean absolute path under
// sandbox.WorkspaceDir. Relative paths are relative to WorkspaceDir. Paths
// that leave it (by "..", or by being absolute elsewhere) are rejected.
//
// The check is lexical: a symlink inside the workspace that points out of
// it is followed by the sandbox. The sandbox is the trust boundary, so
// this keeps the agent on track rather than enforcing security.
func resolvePath(p string) (string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", errors.New("path is empty")
	}
	if strings.ContainsRune(p, 0) {
		return "", errors.New("path contains a NUL byte")
	}
	if !path.IsAbs(p) {
		p = path.Join(sandbox.WorkspaceDir, p)
	}
	c := path.Clean(p)
	if c != sandbox.WorkspaceDir && !strings.HasPrefix(c, sandbox.WorkspaceDir+"/") {
		return "", fmt.Errorf("path %q is outside %s; every path must be inside the workspace", p, sandbox.WorkspaceDir)
	}
	return c, nil
}

// resolveFilePath is resolvePath for a file the agent changes: not the
// workspace itself and not inside a .git directory.
func resolveFilePath(p string) (string, error) {
	c, err := resolvePath(p)
	if err != nil {
		return "", err
	}
	if c == sandbox.WorkspaceDir {
		return "", fmt.Errorf("%s is a directory, not a file", c)
	}
	for _, part := range strings.Split(c, "/") {
		if part == ".git" {
			return "", fmt.Errorf("%s is inside a .git directory; use git commands through shell_exec instead", c)
		}
	}
	return c, nil
}

// repoFor returns the repo containing the absolute path p.
func repoFor(run *Run, p string) (RepoContext, bool) {
	best, ok := RepoContext{}, false
	for _, r := range run.Repos {
		root := strings.TrimRight(r.Path, "/")
		if root == "" {
			continue
		}
		if (p == root || strings.HasPrefix(p, root+"/")) && len(root) > len(best.Path) {
			best, ok = r, true
		}
	}
	return best, ok
}

// shellQuote returns s as one POSIX shell word.
func shellQuote(s string) string {
	if s == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// ---- text ----

// oneLine flattens s to one line of at most max bytes.
func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	return headString(s, max)
}

// headString cuts s to at most max bytes at a rune boundary, adding "…".
func headString(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

// tailString keeps the last max bytes of s at a rune boundary, adding "…".
func tailString(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := len(s) - max
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return "…" + s[cut:]
}

// validUTF8 replaces invalid UTF-8 so text survives JSON.
func validUTF8(s string) string {
	return strings.ToValidUTF8(s, "�")
}

// plural returns "1 thing" or "n things".
func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}
