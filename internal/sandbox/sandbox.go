// Package sandbox defines the isolated machine a session's agent works in,
// and the named-shell manager its shell tools use.
//
// Implementations live in subpackages (sandbox/docker, sandbox/daytona, a
// fake for tests). The worker talks to a sandbox through these interfaces
// only; nothing outside a provider package sees SDK types.
//
// Trust boundary: a sandbox runs code the agent and the user's repo chose,
// and users can open shells in it. Never put LLM API keys, the GitHub App
// private key, agent code or prompts inside one. The only credential a
// sandbox holds is a short-lived, repo-scoped, read-only (contents: read)
// installation token behind a git credential helper, plus the org's own
// secrets injected as env vars. Pushes happen in the worker.
package sandbox

import (
	"context"
	"errors"
	"io/fs"
	"strings"
	"time"
)

// WorkspaceDir is where repos are cloned (WorkspaceDir/<repo name>) and
// the default working directory for Exec and relative paths.
const WorkspaceDir = "/workspace"

// MaxExecOutput is the most bytes of each output stream an ExecResult or
// ShellResult keeps. Providers keep the tail when output is longer.
const MaxExecOutput = 1 << 20

// MaxReadFile is the most bytes Sandbox.ReadFile returns; larger files
// fail with ErrFileTooLarge. Sandbox.ReadFileLimit takes its own limit.
const MaxReadFile = 64 << 20

// SystemPath is the PATH providers give every command they run as root,
// and their own helper scripts, so nothing in a directory the sandbox
// user can write (such as ~/.local/bin) runs in their place.
const SystemPath = "/usr/sbin:/usr/bin:/sbin:/bin"

// IsRoot reports whether an ExecRequest.User names root ("root" or "0",
// optionally with a ":group").
func IsRoot(user string) bool {
	u, _, _ := strings.Cut(user, ":")
	return u == "root" || u == "0"
}

var (
	// ErrFileTooLarge: ReadFile or ReadFileLimit was given a file larger
	// than its limit. The file is not read into memory.
	ErrFileTooLarge = errors.New("sandbox: file too large")
	// ErrNotRegular: ReadFile or ReadFileLimit was given a path that is
	// not a regular file or a symlink to one, such as a directory, a
	// device (/dev/zero), a FIFO or a socket. Nothing is read.
	ErrNotRegular = errors.New("sandbox: not a regular file")
	// ErrNotFound: no sandbox, snapshot or shell with that ID or name.
	ErrNotFound = errors.New("sandbox: not found")
	// ErrNotSupported: the provider cannot do this operation.
	ErrNotSupported = errors.New("sandbox: operation not supported by provider")
	// ErrShellExists: Shells.Create was given a name already in use.
	ErrShellExists = errors.New("sandbox: shell already exists")
	// ErrShellBusy: Shells.Exec was called while the shell's previous
	// command is still running. Read or Interrupt it, or use another shell.
	ErrShellBusy = errors.New("sandbox: shell is busy")
)

// Spec describes a sandbox to create.
type Spec struct {
	// Template is the base image or provider template. Ignored when
	// SnapshotID is set.
	Template string
	// SnapshotID starts the sandbox from a snapshot returned by
	// Sandbox.Snapshot (a repo's ready-made template).
	SnapshotID string
	// CPUs and MemoryMB size the sandbox; 0 means the provider default
	// (about 2 vCPU and 4096 MB).
	CPUs     int
	MemoryMB int
	// Env is set for every process in the sandbox. It carries the org's
	// decrypted secrets, so implementations must not log it.
	Env map[string]string
	// Labels tag the sandbox at the provider (session ID, org ID) so
	// orphans can be found and reaped.
	Labels map[string]string
	// EgressAllowlist lists the hosts outbound traffic may reach (GitHub,
	// package registries, org hosts). Empty means the provider's default
	// policy. Providers that cannot enforce it document that.
	EgressAllowlist []string
}

// ExecRequest is one command to run to completion.
type ExecRequest struct {
	// Cmd is a shell command line, run with `sh -c`.
	Cmd string
	// Dir is the working directory; empty means WorkspaceDir.
	Dir string
	// Env is added to the sandbox's environment for this command.
	Env map[string]string
	// Timeout kills the command after this long; 0 means no timeout
	// beyond ctx.
	Timeout time.Duration
	// User runs the command as this user; empty means the sandbox's
	// default non-root user. A command run as root gets PATH=SystemPath,
	// whatever the sandbox's environment or Env says, so call binaries by
	// name only if they are in a system directory.
	User string
}

// ExecResult is the outcome of a finished command.
type ExecResult struct {
	// ExitCode is the process's exit status, or -1 if it did not exit
	// normally (killed or timed out).
	ExitCode int
	// Stdout and Stderr hold up to MaxExecOutput bytes each (the tail).
	Stdout string
	Stderr string
	// Duration is the wall time the command ran.
	Duration time.Duration
	// TimedOut is true when ExecRequest.Timeout killed the command.
	TimedOut bool
}

// Sandbox is one running (or paused) sandbox.
//
// File paths are absolute paths inside the sandbox; relative paths are
// resolved against WorkspaceDir. Errors for a missing file wrap
// fs.ErrNotExist.
type Sandbox interface {
	// ID is the provider's ID, stored in sessions.sandbox_id.
	ID() string

	// Exec runs a command and waits for it. A non-zero exit status is not
	// an error: err is non-nil only when the command could not be run or
	// the sandbox failed. If onOutput is non-nil it receives stdout and
	// stderr chunks, interleaved, as they arrive, from one goroutine.
	Exec(ctx context.Context, req ExecRequest, onOutput func(chunk []byte)) (*ExecResult, error)

	// ReadFile is ReadFileLimit with MaxReadFile.
	ReadFile(ctx context.Context, path string) ([]byte, error)
	// ReadFileLimit returns the contents of a regular file of at most
	// limit bytes, following symlinks; limit <= 0 means MaxReadFile. A
	// larger file fails with an error wrapping ErrFileTooLarge, and a path
	// that is not a regular file with one wrapping ErrNotRegular, before
	// anything is read. Implementations also stop reading after limit+1
	// bytes, so a file that changes after the check (or a symlink swapped
	// for /dev/zero) can't make one call hold more than that in memory.
	ReadFileLimit(ctx context.Context, path string, limit int64) ([]byte, error)
	// WriteFile creates or replaces a file, creating parent directories,
	// owned by the default user.
	WriteFile(ctx context.Context, path string, data []byte, mode fs.FileMode) error
	// RemoveFile deletes a file.
	RemoveFile(ctx context.Context, path string) error

	// PortURL returns a URL that reaches the given port inside the
	// sandbox. Callers must put it behind the API's auth proxy before
	// showing it to a user.
	PortURL(ctx context.Context, port int) (string, error)

	// Pause stops the sandbox's compute and keeps its disk. Resume
	// restarts it. Both are no-ops when already in the target state.
	Pause(ctx context.Context) error
	Resume(ctx context.Context) error

	// Snapshot saves the sandbox's filesystem and returns an ID usable as
	// Spec.SnapshotID. name is a human-readable label.
	Snapshot(ctx context.Context, name string) (string, error)

	// Delete destroys the sandbox. Deleting an already deleted sandbox
	// returns nil.
	Delete(ctx context.Context) error
}

// Provider creates and finds sandboxes.
type Provider interface {
	// Name is the provider's stable ID, for example "docker" or "daytona".
	Name() string
	// Create provisions a sandbox and returns it running.
	Create(ctx context.Context, spec Spec) (Sandbox, error)
	// Get returns an existing sandbox by ID, in whatever state it is in.
	// It returns an error wrapping ErrNotFound if there is none.
	Get(ctx context.Context, id string) (Sandbox, error)
}

// ShellResult is what Shells.Exec returns.
type ShellResult struct {
	// OutputTail is the end of the command's output (stdout and stderr as
	// the terminal showed them), trimmed for the model.
	OutputTail string
	// ExitCode is the command's exit status, or -1 if it is still running
	// (TimedOut).
	ExitCode int
	// TimedOut is true when the timeout passed before the command
	// finished. The command keeps running: call Read to see new output or
	// Interrupt to stop it.
	TimedOut bool
	// Duration is how long Exec waited.
	Duration time.Duration
}

// Shells manages named, persistent shells inside one sandbox (tmux
// windows). The agent's create_shell, shell_exec, shell_read and
// shell_interrupt tools map one-to-one onto these methods. Shell names are
// chosen by the model and are unique within the sandbox.
//
// Implementations stream live terminal output to the worker through a
// callback given at construction (not part of this interface), which the
// worker turns into tool_call.output events. Shells is safe for
// concurrent use; calls for one shell are serialized.
type Shells interface {
	// Create opens a new shell in WorkspaceDir. It returns ErrShellExists
	// if name is taken.
	Create(ctx context.Context, name string) error

	// Exec runs cmd in the named shell and waits up to timeout for it to
	// finish. A timeout is not an error (see ShellResult.TimedOut). It
	// returns an error wrapping ErrNotFound for an unknown shell and
	// ErrShellBusy if the shell's previous command is still running.
	Exec(ctx context.Context, name, cmd string, timeout time.Duration) (ShellResult, error)

	// Read returns output the shell printed since the last Exec or Read
	// returned, for following a long-running process such as a dev
	// server.
	Read(ctx context.Context, name string) (string, error)

	// Interrupt stops the shell's running command. It sends Ctrl-C and,
	// while a foreground program still holds the terminal (a pager such
	// as less ignores Ctrl-C), escalates: it types q, then terminates and
	// finally kills the shell's foreground process group, so the shell is
	// free again.
	Interrupt(ctx context.Context, name string) error

	// List returns the names of open shells in creation order.
	List() []string
}
