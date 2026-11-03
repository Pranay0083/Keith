package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// search limits.
const (
	maxSearchLines   = 200
	maxSearchColumns = 300
	searchTimeout    = 60 * time.Second
)

// rgExitRE finds the exit status marker the search command prints.
var rgExitRE = regexp.MustCompile(`__keith_rg_exit=(\d+)\n?`)

type searchTool struct{ maxOutput int }

func (searchTool) Spec() llm.ToolSpec {
	return spec(toolSearch,
		fmt.Sprintf("Search the workspace with ripgrep. With a pattern, returns matching lines as path:line:text; without one, lists the files matching glob. Results are relative to /workspace, respect .gitignore, skip .git and stop after %d lines, so narrow the search with path and glob when it is too broad.", maxSearchLines),
		map[string]any{
			"pattern":     strProp("A regular expression (Rust regex syntax, like grep -E) to search file contents for. Omit it to list files instead."),
			"glob":        strProp("Only search files matching this glob, for example \"*.go\" or \"src/**/*.test.ts\". Prefix with ! to exclude."),
			"path":        strProp("A directory or file to search in, under /workspace (default: all of /workspace)."),
			"ignore_case": boolProp("Match case-insensitively."),
			"literal":     boolProp("Treat pattern as a literal string, not a regular expression."),
		})
}

func (t searchTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Pattern    string `json:"pattern"`
		Glob       string `json:"glob"`
		Path       string `json:"path"`
		IgnoreCase bool   `json:"ignore_case"`
		Literal    bool   `json:"literal"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if in.Pattern == "" && strings.TrimSpace(in.Glob) == "" {
		return errorResult("Give a pattern to search for, or a glob to list files."), nil
	}
	root := sandbox.WorkspaceDir
	if strings.TrimSpace(in.Path) != "" {
		p, err := resolvePath(in.Path)
		if err != nil {
			return badInput(err), nil
		}
		root = p
	}

	args := []string{"rg", "--color=never", "--hidden", "--glob", shellQuote("!.git"), "--sort=path"}
	if in.Pattern != "" {
		args = append(args, "--line-number", "--no-heading", "--with-filename",
			"--max-columns="+strconv.Itoa(maxSearchColumns), "--max-columns-preview")
		if in.IgnoreCase {
			args = append(args, "--ignore-case")
		}
		if in.Literal {
			args = append(args, "--fixed-strings")
		}
	} else {
		args = append(args, "--files")
	}
	if g := strings.TrimSpace(in.Glob); g != "" {
		args = append(args, "--glob", shellQuote(g))
	}
	if in.Pattern != "" {
		args = append(args, "--regexp", shellQuote(in.Pattern))
	}
	if root != sandbox.WorkspaceDir {
		args = append(args, "--", shellQuote(strings.TrimPrefix(root, sandbox.WorkspaceDir+"/")))
	}
	// head keeps the first lines (the sandbox would keep the last MiB of a
	// huge result); the marker carries rg's own exit status through the
	// pipe.
	cmd := fmt.Sprintf("{ %s; echo \"__keith_rg_exit=$?\" >&2; } | head -n %d", strings.Join(args, " "), maxSearchLines+1)
	res, err := run.Sandbox.Exec(ctx, sandbox.ExecRequest{Cmd: cmd, Dir: sandbox.WorkspaceDir, Timeout: searchTimeout}, nil)
	if err != nil {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return errorResult(fmt.Sprintf("search failed: %v", err)), nil
	}
	if res.TimedOut {
		return errorResult(fmt.Sprintf("search timed out after %s; narrow it with path or glob.", searchTimeout)), nil
	}

	code := -1
	if m := rgExitRE.FindStringSubmatch(res.Stderr); m != nil {
		code, _ = strconv.Atoi(m[1])
	}
	stderr := strings.TrimSpace(rgExitRE.ReplaceAllString(res.Stderr, ""))
	lines := splitLines(validUTF8(res.Stdout))
	what := "files matching " + strconv.Quote(in.Glob)
	if in.Pattern != "" {
		what = "matches for " + strconv.Quote(in.Pattern)
	}
	switch {
	case code == 2 && len(lines) == 0, code > 2 && len(lines) == 0, code < 0 && res.ExitCode != 0:
		msg := stderr
		if msg == "" {
			msg = fmt.Sprintf("exit status %d", res.ExitCode)
		}
		return errorResult("search failed: " + headString(msg, 1000)), nil
	case len(lines) == 0:
		return textResult("no "+what, "No "+what+"."), nil
	}

	more := len(lines) > maxSearchLines
	if more {
		lines = lines[:maxSearchLines]
	}
	var b strings.Builder
	shown := 0
	for _, l := range lines {
		if b.Len()+len(l)+1 > t.maxOutput && shown > 0 {
			more = true
			break
		}
		b.WriteString(l)
		b.WriteByte('\n')
		shown++
	}
	if more {
		fmt.Fprintf(&b, "\n(Showing the first %d lines; there are more. Narrow the search with path, glob or a more specific pattern.)", shown)
	}
	if stderr != "" {
		fmt.Fprintf(&b, "\n(Warnings: %s)", headString(stderr, 500))
	}
	summary := fmt.Sprintf("%s: %d", what, shown)
	if more {
		summary += "+"
	}
	return textResult(oneLine(summary, 120), b.String()), nil
}
