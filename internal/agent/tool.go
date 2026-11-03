// Package agent is the coding agent: the tool_use loop over an
// llm.Provider, its tools and its prompt.
//
// Package boundaries: agent imports llm, sandbox and events only. It never
// imports github, store, api or a concrete provider; the worker wires
// those in through Run.
package agent

import (
	"context"
	"encoding/json"
	"time"

	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// Tool is one tool the model can call.
type Tool interface {
	// Spec describes the tool to the model. Its InputSchema must be a
	// JSON Schema object that works with Strict set.
	Spec() llm.ToolSpec

	// Execute runs the tool with the model's arguments.
	//
	// Problems the model should see and react to (invalid arguments, a
	// missing file, a failing command) are returned as a Result with
	// IsError set and an explanation in Content, and a nil error. A
	// non-nil error means the run cannot continue (the sandbox is gone,
	// ctx was cancelled) and stops the loop.
	Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error)
}

// Result is a tool's answer.
type Result struct {
	// Content becomes the tool_result block's content (text and images).
	Content []llm.Block
	// IsError marks the tool_result as an error for the model.
	IsError bool
	// Summary is a one-line description of the outcome for the UI
	// timeline (tool_call.finished).
	Summary string
	// Pause ends the loop after this tool result is recorded and moves
	// the session to waiting_for_user. ask_user sets it.
	Pause bool
}

// Run is everything a tool can reach during one session run. The worker
// builds it; tools must not keep it after Execute returns.
type Run struct {
	SessionID string
	// Sandbox is the session's sandbox (file and exec API).
	Sandbox sandbox.Sandbox
	// Shells is the named-shell manager inside Sandbox.
	Shells sandbox.Shells
	// Emit appends an event to the session log (store, then publish).
	// typ is an events.Type* constant and payload the matching
	// events.*Payload struct.
	Emit func(ctx context.Context, typ string, payload any) error
	// Repos are the repos cloned for this session.
	Repos []RepoContext
	// Host reaches the code host (GitHub) from the control plane. Its
	// credentials never enter the sandbox.
	Host RepoHost
}

// RepoContext is one repo cloned into the sandbox.
type RepoContext struct {
	// FullName is owner/name.
	FullName string
	// Path is the clone's absolute path in the sandbox,
	// sandbox.WorkspaceDir/<name>.
	Path string
	// Branch is the session's working branch, agent/<session-id>.
	Branch string
	// DefaultBranch is the base branch PRs target.
	DefaultBranch string
}

// RepoHost is the code-host API the agent's read_issue and
// create_pull_request tools use. internal/github implements it; agent
// must not import github.
type RepoHost interface {
	// ReadIssue fetches an issue (or pull request) with its comments.
	// repo is owner/name.
	ReadIssue(ctx context.Context, repo string, number int) (Issue, error)
	// OpenPR opens a pull request from an already pushed branch. If one
	// is already open for the same head and base, it returns that one.
	OpenPR(ctx context.Context, req PRRequest) (PR, error)
}

// Issue is an issue or pull request as the model sees it.
type Issue struct {
	Repo     string
	Number   int
	Title    string
	Body     string
	State    string // "open" or "closed"
	Author   string
	Labels   []string
	URL      string
	Comments []IssueComment
}

// IssueComment is one comment on an Issue.
type IssueComment struct {
	Author    string
	Body      string
	CreatedAt time.Time
}

// PRRequest asks RepoHost to open a pull request.
type PRRequest struct {
	Repo  string // owner/name
	Head  string // branch with the changes, agent/<session-id>
	Base  string // target branch, usually the default branch
	Title string
	Body  string
	Draft bool
}

// PR is an opened pull request.
type PR struct {
	Repo   string
	Number int
	URL    string
	Head   string
	Base   string
}
