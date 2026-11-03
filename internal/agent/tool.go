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
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/events"
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
	// ExitCode is a shell command's exit status, reported in
	// tool_call.finished; 0 when not applicable.
	ExitCode int
	// TimedOut reports a shell command still running when the tool
	// returned (tool_call.finished timed_out).
	TimedOut bool

	// question is the open question behind a Pause (set by ask_user).
	question *Question
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

	// GitAuthor is the identity open_pr commits as (git -c user.name/
	// user.email). The zero value means DefaultGitAuthor.
	GitAuthor GitIdentity
	// AfterFileChange, if set, is called after write_file, edit_file,
	// delete_file or undo_edit changed a file and before file.changed is
	// emitted, so the worker can save the session's diff first (the web
	// app fetches the diff when it sees file.changed). A non-nil error
	// stops the run.
	AfterFileChange func(ctx context.Context, change FileChange) error
	// OnPROpened, if set, is called after open_pr opened a repo's pull
	// request and before pr.opened is emitted, so the worker can record it
	// in session_repos first. It runs once per repo per session; a later
	// open_pr only pushes new commits to the same PR. A non-nil error
	// stops the run.
	OnPROpened func(ctx context.Context, repo RepoContext, pr PR) error

	// mu guards shellCalls, which CurrentCall reads from the shell
	// manager's output goroutine.
	mu         sync.Mutex
	shellCalls map[string]string
	// state is the State of the Runner.Run in progress (undo_edit's edit
	// history). Only the loop goroutine touches it.
	state *State
}

// CurrentCall returns the ID of the tool call (shell_exec, shell_read or
// shell_interrupt) that is currently reading the named shell, or "" when
// none is. The worker's tmuxshell.WithOutput callback uses it to fill
// tool_call.output's call_id:
//
//	run := &agent.Run{...}
//	run.Shells = tmuxshell.New(sb, tmuxshell.WithOutput(func(shell string, chunk []byte) {
//		emit(ctx, events.TypeToolCallOutput, events.ToolCallOutputPayload{
//			CallID: run.CurrentCall(shell), ShellID: shell, Chunk: redact(chunk)})
//	}))
//
// It is safe to call from any goroutine.
func (r *Run) CurrentCall(shell string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.shellCalls[shell]
}

// bindShell records callID as the call reading shell until the returned
// func runs.
func (r *Run) bindShell(shell, callID string) func() {
	r.mu.Lock()
	if r.shellCalls == nil {
		r.shellCalls = map[string]string{}
	}
	prev, had := r.shellCalls[shell]
	r.shellCalls[shell] = callID
	r.mu.Unlock()
	return func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		if had {
			r.shellCalls[shell] = prev
		} else {
			delete(r.shellCalls, shell)
		}
	}
}

// GitIdentity is a git author and committer.
type GitIdentity struct {
	Name  string
	Email string
}

// DefaultGitAuthor is used when Run.GitAuthor is empty.
var DefaultGitAuthor = GitIdentity{Name: "Keith", Email: "keith-agent@users.noreply.github.com"}

// FileChange describes one file a tool changed (Run.AfterFileChange).
type FileChange struct {
	// CallID is the tool call that changed the file.
	CallID string
	// Tool is the tool's name.
	Tool string
	// Repo is the repo the file is in; the zero RepoContext when the
	// file is outside every repo.
	Repo RepoContext
	// Path is the file's absolute path in the sandbox.
	Path   string
	Action events.FileAction
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
	// Profile is the repo's onboarding profile (repo_profiles). The
	// prompt builder shows it to the model, and repeat detection watches
	// its TestCmd.
	Profile RepoProfile
}

// RepoProfile is a repo's onboarding answers (repo_profiles).
type RepoProfile struct {
	InstallCmd string
	BuildCmd   string
	TestCmd    string
	RunCmd     string
	// Knowledge is free-form notes from the team.
	Knowledge string
}

// RepoHost is the code-host API the agent's read_issue and
// open_pr tools use. internal/github implements it; agent
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
