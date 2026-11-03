package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"strconv"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// read_file limits.
const (
	// maxFileBytes is the largest file the file tools load.
	maxFileBytes = 8 << 20
	// maxReadLines is the most lines read_file returns at once.
	maxReadLines = 2000
	// maxLineBytes cuts longer lines in read_file's output.
	maxLineBytes = 2000
	// snippetContext is the lines of context edit_file shows around a
	// change.
	snippetContext = 3
)

const pathDesc = "The file's path: absolute under /workspace, or relative to /workspace (for example \"api/src/main.go\")."

// loadFile reads a file for a file tool. A missing file is (nil, false,
// nil). A file over maxFileBytes, or a path that is not a regular file (a
// directory, a device such as /dev/zero, a FIFO, or a symlink to one), is
// refused by the sandbox before anything is read into memory. Failures
// become a model-visible result in res, except a cancelled ctx, which is
// returned as err.
func loadFile(ctx context.Context, run *Run, p string) (data []byte, exists bool, res *Result, err error) {
	data, err = run.Sandbox.ReadFileLimit(ctx, p, maxFileBytes)
	switch {
	case err == nil:
		return data, true, nil, nil
	case errors.Is(err, fs.ErrNotExist):
		return nil, false, nil, nil
	case ctx.Err() != nil:
		return nil, false, nil, ctx.Err()
	case errors.Is(err, sandbox.ErrFileTooLarge):
		r := errorResult(fmt.Sprintf("%s is too large for this tool (over %d MiB); use shell commands (for example sed -n '1,200p' or head) instead.", p, maxFileBytes>>20))
		return nil, true, &r, nil
	case errors.Is(err, sandbox.ErrNotRegular):
		r := errorResult(fmt.Sprintf("%s is not a regular file (it is a directory, a device, a FIFO or a socket, or a symlink to one), so the file tools can't open it. Use shell commands to inspect it.", p))
		return nil, true, &r, nil
	}
	r := errorResult(fmt.Sprintf("Could not read %s: %v", p, err))
	return nil, false, &r, nil
}

// fileMode returns a file's permission bits, or 0 (the provider default)
// if they can't be read, so rewriting a script keeps its executable bit.
func fileMode(ctx context.Context, run *Run, p string) fs.FileMode {
	res, err := run.Sandbox.Exec(ctx, sandbox.ExecRequest{Cmd: "stat -c %a -- " + shellQuote(p), Timeout: 10 * time.Second}, nil)
	if err != nil || res.ExitCode != 0 {
		return 0
	}
	m, err := strconv.ParseUint(strings.TrimSpace(res.Stdout), 8, 32)
	if err != nil {
		return 0
	}
	return fs.FileMode(m).Perm()
}

// fileChanged runs Run.AfterFileChange (the worker saves the diff) and
// then emits file.changed.
func fileChanged(ctx context.Context, run *Run, tool, p string, action events.FileAction) error {
	repo, _ := repoFor(run, p)
	if run.AfterFileChange != nil {
		if err := run.AfterFileChange(ctx, FileChange{CallID: CallID(ctx), Tool: tool, Repo: repo, Path: p, Action: action}); err != nil {
			return fmt.Errorf("agent: AfterFileChange: %w", err)
		}
	}
	return run.emit(ctx, events.TypeFileChanged, events.FileChangedPayload{
		CallID: CallID(ctx), Repo: repo.FullName, Path: p, Action: action,
	})
}

// recordEdit adds an edit to the session's edit history.
func recordEdit(ctx context.Context, run *Run, tool, p string, existed bool, prev []byte, mode fs.FileMode, after []byte, deleted bool) {
	if run.state == nil {
		return
	}
	rec := EditRecord{Tool: tool, CallID: CallID(ctx), Existed: existed, Mode: mode}
	if existed {
		rec.Prev = bytes.Clone(prev)
	}
	if !deleted {
		rec.After = contentHash(after)
	}
	run.state.Aux.Edits.push(p, rec)
}

func countLines(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	n := bytes.Count(b, []byte("\n"))
	if b[len(b)-1] != '\n' {
		n++
	}
	return n
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

// numbered formats lines[from-1:to] with line numbers like cat -n.
func numbered(lines []string, from, to int) string {
	var b strings.Builder
	for i := from; i <= to && i <= len(lines); i++ {
		line := lines[i-1]
		if len(line) > maxLineBytes {
			line = headString(line, maxLineBytes) + " (line truncated)"
		}
		fmt.Fprintf(&b, "%6d\t%s\n", i, line)
	}
	return b.String()
}

func isBinary(b []byte) bool {
	return bytes.IndexByte(b[:min(len(b), 8000)], 0) >= 0
}

// ---- read_file ----

type readFileTool struct{ maxOutput int }

func (readFileTool) Spec() llm.ToolSpec {
	return spec(toolReadFile,
		fmt.Sprintf("Read a text file, with line numbers (up to %d lines at a time). Use start_line and end_line to read part of a large file. Line numbers are for reference only; they are not part of the file.", maxReadLines),
		map[string]any{
			"path":       strProp(pathDesc),
			"start_line": intProp("First line to show, 1-based (default 1)."),
			"end_line":   intProp("Last line to show, inclusive (default: the end of the file)."),
		},
		"path")
}

func (t readFileTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Path      string `json:"path"`
		StartLine *int   `json:"start_line"`
		EndLine   *int   `json:"end_line"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	p, err := resolvePath(in.Path)
	if err != nil {
		return badInput(err), nil
	}
	data, exists, res, err := loadFile(ctx, run, p)
	if err != nil || res != nil {
		return deref(res), err
	}
	if !exists {
		return errorResult(fmt.Sprintf("No such file: %s", p)), nil
	}
	if isBinary(data) {
		return textResult("binary file "+p, fmt.Sprintf("%s is a binary file (%d bytes); not shown.", p, len(data))), nil
	}
	lines := splitLines(validUTF8(string(data)))
	total := len(lines)
	if total == 0 {
		return textResult("read "+p, fmt.Sprintf("%s is empty.", p)), nil
	}
	start, end := 1, total
	if in.StartLine != nil && *in.StartLine > 0 {
		start = *in.StartLine
	}
	if in.EndLine != nil && *in.EndLine > 0 {
		end = min(*in.EndLine, total)
	}
	if start > total {
		return errorResult(fmt.Sprintf("start_line %d is past the end of %s, which has %d lines.", start, p, total)), nil
	}
	if end < start {
		return errorResult(fmt.Sprintf("end_line %d is before start_line %d.", end, start)), nil
	}

	var b strings.Builder
	shown := start - 1
	for i := start; i <= end && i-start < maxReadLines; i++ {
		line := numbered(lines, i, i)
		if b.Len()+len(line) > t.maxOutput && b.Len() > 0 {
			break
		}
		b.WriteString(line)
		shown = i
	}
	if start > 1 || shown < total {
		fmt.Fprintf(&b, "\n(Showing lines %d-%d of %d.", start, shown, total)
		if shown < total {
			fmt.Fprintf(&b, " Use start_line=%d to read more.", shown+1)
		}
		b.WriteString(")")
	}
	return textResult(fmt.Sprintf("read %s (%d-%d of %d)", p, start, shown, total), b.String()), nil
}

func deref(r *Result) Result {
	if r == nil {
		return Result{}
	}
	return *r
}

// ---- write_file ----

type writeFileTool struct{}

func (writeFileTool) Spec() llm.ToolSpec {
	return spec(toolWriteFile,
		"Create a file, or replace a file's whole content. Parent directories are created. Prefer edit_file for changing part of an existing file. undo_edit reverts it.",
		map[string]any{
			"path":    strProp(pathDesc),
			"content": strProp("The complete new content of the file."),
		},
		"path", "content")
}

func (writeFileTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	p, err := resolveFilePath(in.Path)
	if err != nil {
		return badInput(err), nil
	}
	old, existed, res, err := loadFile(ctx, run, p)
	if err != nil || res != nil {
		return deref(res), err
	}
	content := []byte(in.Content)
	if existed && bytes.Equal(old, content) {
		return textResult("unchanged "+p, fmt.Sprintf("%s already has this content; nothing changed.", p)), nil
	}
	var mode fs.FileMode
	if existed {
		mode = fileMode(ctx, run, p)
	}
	if err := run.Sandbox.WriteFile(ctx, p, content, mode); err != nil {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return errorResult(fmt.Sprintf("Could not write %s: %v", p, err)), nil
	}
	recordEdit(ctx, run, toolWriteFile, p, existed, old, mode, content, false)
	action, verb := events.FileCreated, "Created"
	if existed {
		action, verb = events.FileModified, "Wrote"
	}
	if err := fileChanged(ctx, run, toolWriteFile, p, action); err != nil {
		return Result{}, err
	}
	lines := plural(countLines(content), "line", "lines")
	return textResult(strings.ToLower(verb)+" "+p, fmt.Sprintf("%s %s (%s).", verb, p, lines)), nil
}

// ---- edit_file ----

type editFileTool struct{}

func (editFileTool) Spec() llm.ToolSpec {
	return spec(toolEditFile,
		"Replace one exact piece of text in a file. old_string must match the file exactly once, including whitespace and indentation; if it matches zero or several times nothing changes and you are told the match count, so add surrounding lines to make it unique. Read the file first. undo_edit reverts the edit.",
		map[string]any{
			"path":       strProp(pathDesc),
			"old_string": strProp("The exact text to replace, unique in the file. Do not include line numbers from read_file."),
			"new_string": strProp("The text to put in its place (may be empty to delete old_string)."),
		},
		"path", "old_string", "new_string")
}

func (editFileTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Path      string `json:"path"`
		OldString string `json:"old_string"`
		NewString string `json:"new_string"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	p, err := resolveFilePath(in.Path)
	if err != nil {
		return badInput(err), nil
	}
	if in.OldString == "" {
		return errorResult("old_string is empty. Use write_file to create a file or replace all of it."), nil
	}
	if in.OldString == in.NewString {
		return errorResult("old_string and new_string are the same; nothing to change."), nil
	}
	data, exists, res, err := loadFile(ctx, run, p)
	if err != nil || res != nil {
		return deref(res), err
	}
	if !exists {
		return errorResult(fmt.Sprintf("No such file: %s. Use write_file to create it.", p)), nil
	}
	content := string(data)
	switch n := strings.Count(content, in.OldString); {
	case n == 0:
		msg := fmt.Sprintf("old_string was not found in %s (0 matches). Read the file again and copy the text exactly, including indentation.", p)
		if strings.Contains(collapseSpace(content), collapseSpace(in.OldString)) {
			msg += " A match exists if whitespace is ignored: check indentation (tabs vs spaces) and line breaks."
		}
		return errorResult(msg), nil
	case n > 1:
		return errorResult(fmt.Sprintf("old_string matches %d times in %s (at lines %s). Include more surrounding lines so it matches exactly once.",
			n, p, matchLines(content, in.OldString, 10))), nil
	}

	at := strings.Index(content, in.OldString)
	updated := content[:at] + in.NewString + content[at+len(in.OldString):]
	mode := fileMode(ctx, run, p)
	if err := run.Sandbox.WriteFile(ctx, p, []byte(updated), mode); err != nil {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return errorResult(fmt.Sprintf("Could not write %s: %v", p, err)), nil
	}
	recordEdit(ctx, run, toolEditFile, p, true, data, mode, []byte(updated), false)
	if err := fileChanged(ctx, run, toolEditFile, p, events.FileModified); err != nil {
		return Result{}, err
	}

	line := strings.Count(content[:at], "\n") + 1
	lines := splitLines(validUTF8(updated))
	from := max(1, line-snippetContext)
	to := min(len(lines), line+strings.Count(in.NewString, "\n")+snippetContext)
	return textResult(fmt.Sprintf("edited %s:%d", p, line),
		fmt.Sprintf("Edited %s: replaced 1 match at line %d. The file now reads:\n%s", p, line, numbered(lines, from, to))), nil
}

func collapseSpace(s string) string { return strings.Join(strings.Fields(s), " ") }

// matchLines lists the line numbers where sub starts, at most max of them.
func matchLines(content, sub string, max int) string {
	var nums []string
	off := 0
	for len(nums) < max {
		i := strings.Index(content[off:], sub)
		if i < 0 {
			break
		}
		nums = append(nums, strconv.Itoa(strings.Count(content[:off+i], "\n")+1))
		off += i + len(sub)
	}
	return strings.Join(nums, ", ")
}

// ---- delete_file ----

type deleteFileTool struct{}

func (deleteFileTool) Spec() llm.ToolSpec {
	return spec(toolDeleteFile,
		"Delete a file. undo_edit restores it.",
		map[string]any{"path": strProp(pathDesc)},
		"path")
}

func (deleteFileTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Path string `json:"path"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	p, err := resolveFilePath(in.Path)
	if err != nil {
		return badInput(err), nil
	}
	old, exists, res, err := loadFile(ctx, run, p)
	if err != nil || res != nil {
		return deref(res), err
	}
	if !exists {
		return errorResult(fmt.Sprintf("No such file: %s", p)), nil
	}
	mode := fileMode(ctx, run, p)
	if err := run.Sandbox.RemoveFile(ctx, p); err != nil {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return errorResult(fmt.Sprintf("Could not delete %s: %v", p, err)), nil
	}
	recordEdit(ctx, run, toolDeleteFile, p, true, old, mode, nil, true)
	if err := fileChanged(ctx, run, toolDeleteFile, p, events.FileDeleted); err != nil {
		return Result{}, err
	}
	return textResult("deleted "+p, fmt.Sprintf("Deleted %s.", p)), nil
}

// ---- undo_edit ----

type undoEditTool struct{}

func (undoEditTool) Spec() llm.ToolSpec {
	return spec(toolUndoEdit,
		"Revert the most recent write_file, edit_file or delete_file on a file, restoring its previous content. Call it again to step further back. It refuses if the file changed since that edit by other means (for example a shell command).",
		map[string]any{"path": strProp(pathDesc)},
		"path")
}

func (undoEditTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Path string `json:"path"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	p, err := resolveFilePath(in.Path)
	if err != nil {
		return badInput(err), nil
	}
	if run.state == nil {
		return errorResult("Undo is not available outside a session run."), nil
	}
	hist := &run.state.Aux.Edits
	rec, ok := hist.last(p)
	if !ok {
		return errorResult(fmt.Sprintf("There is no edit to %s to undo. Only changes made with write_file, edit_file and delete_file in this session can be undone.", p)), nil
	}
	if rec.Dropped {
		return errorResult(fmt.Sprintf("The content of %s before the last %s was too large to keep, so it can't be undone.", p, rec.Tool)), nil
	}
	cur, exists, res, err := loadFile(ctx, run, p)
	if err != nil || res != nil {
		return deref(res), err
	}
	curHash := ""
	if exists {
		curHash = contentHash(cur)
	}
	if curHash != rec.After {
		return errorResult(fmt.Sprintf("%s changed after your last %s (for example through a shell command), so undoing would discard that change. Use edit_file or write_file instead.", p, rec.Tool)), nil
	}

	var action events.FileAction
	if rec.Existed {
		if err := run.Sandbox.WriteFile(ctx, p, rec.Prev, rec.Mode); err != nil {
			if ctx.Err() != nil {
				return Result{}, ctx.Err()
			}
			return errorResult(fmt.Sprintf("Could not restore %s: %v", p, err)), nil
		}
		action = events.FileModified
		if !exists {
			action = events.FileCreated
		}
	} else {
		if err := run.Sandbox.RemoveFile(ctx, p); err != nil {
			if ctx.Err() != nil {
				return Result{}, ctx.Err()
			}
			return errorResult(fmt.Sprintf("Could not remove %s: %v", p, err)), nil
		}
		action = events.FileDeleted
	}
	hist.pop(p)
	if err := fileChanged(ctx, run, toolUndoEdit, p, action); err != nil {
		return Result{}, err
	}
	msg := fmt.Sprintf("Reverted the last %s to %s.", rec.Tool, p)
	switch {
	case !rec.Existed:
		msg = fmt.Sprintf("Reverted the last %s: %s did not exist before, so it was removed.", rec.Tool, p)
	case action == events.FileCreated:
		msg = fmt.Sprintf("Reverted the last %s: restored %s.", rec.Tool, p)
	}
	if left := len(hist.Files[p]); left > 0 {
		msg += fmt.Sprintf(" %s earlier left to undo.", plural(left, "edit", "edits"))
	}
	return textResult("undid "+rec.Tool+" "+p, msg), nil
}
