package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

const shellNameDesc = "The shell's name, as given to create_shell (letters, digits, '-' and '_')."

// shellError turns a Shells error into a result the model can act on. A
// cancelled context is returned as an error to stop the run.
func shellError(ctx context.Context, op, shell string, err error) (Result, error) {
	if ctx.Err() != nil {
		return Result{}, ctx.Err()
	}
	switch {
	case errors.Is(err, sandbox.ErrNotFound):
		return errorResult(fmt.Sprintf("There is no shell named %q. Create it with create_shell first.", shell)), nil
	case errors.Is(err, sandbox.ErrShellBusy):
		return errorResult(fmt.Sprintf("Shell %q is still running an earlier command. Use shell_read to see its output, shell_interrupt to stop it, or create another shell for this command.", shell)), nil
	case errors.Is(err, sandbox.ErrShellExists):
		return errorResult(fmt.Sprintf("A shell named %q already exists; use it, or pick another name.", shell)), nil
	}
	return errorResult(fmt.Sprintf("%s on shell %q failed: %v", op, shell, err)), nil
}

func noShells() Result {
	return errorResult("Shells are not available in this session.")
}

// createShell opens a shell and emits shell.created.
func createShell(ctx context.Context, run *Run, name string) error {
	if err := run.Shells.Create(ctx, name); err != nil {
		return err
	}
	return run.emit(ctx, events.TypeShellCreated, events.ShellCreatedPayload{ShellID: name, CallID: CallID(ctx)})
}

// ---- create_shell ----

type createShellTool struct{}

func (createShellTool) Spec() llm.ToolSpec {
	return spec(toolCreateShell,
		"Open a new named, persistent bash shell in /workspace. Shells keep their working directory, environment and running processes between commands, and the user can watch each one in its own terminal tab. Use separate shells for things that run at the same time, for example \"main\" for builds and tests and \"server\" for a dev server.",
		map[string]any{"name": strProp("A short name for the shell, for example \"main\" or \"server\" (letters, digits, '-' and '_').")},
		"name")
}

func (createShellTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Name string `json:"name"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if run.Shells == nil {
		return noShells(), nil
	}
	name := strings.TrimSpace(in.Name)
	if name == "" {
		return errorResult("name is empty."), nil
	}
	if err := createShell(ctx, run, name); err != nil {
		if errors.Is(err, errEmitFailed) {
			return Result{}, err
		}
		return shellError(ctx, "create_shell", name, err)
	}
	return textResult("created shell "+name, fmt.Sprintf("Created shell %q in %s.", name, sandbox.WorkspaceDir)), nil
}

// ---- shell_exec ----

type shellExecTool struct {
	timeout, maxTimeout time.Duration
	maxOutput           int
}

func (t shellExecTool) Spec() llm.ToolSpec {
	return spec(toolShellExec,
		fmt.Sprintf("Run a bash command in a named shell and wait for it to finish, then return the exit code and the end of its output. The shell is created if it does not exist yet. If the command is still running after the timeout (default %ds), you get the output so far and it keeps running: use shell_read to follow it or shell_interrupt to stop it. Long-running processes such as dev servers belong in their own shell. Commands run as a non-root user; prefer non-interactive flags (for example -y), since nobody can type into the shell for you.", int(t.timeout/time.Second)),
		map[string]any{
			"shell":           strProp(shellNameDesc),
			"command":         strProp("The bash command line to run."),
			"timeout_seconds": intProp(fmt.Sprintf("How long to wait for the command before returning, in seconds (default %d, at most %d). Raise it for slow builds or test suites.", int(t.timeout/time.Second), int(t.maxTimeout/time.Second))),
		},
		"shell", "command")
}

func (t shellExecTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Shell          string `json:"shell"`
		Command        string `json:"command"`
		TimeoutSeconds *int   `json:"timeout_seconds"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if run.Shells == nil {
		return noShells(), nil
	}
	shell := strings.TrimSpace(in.Shell)
	if shell == "" || strings.TrimSpace(in.Command) == "" {
		return errorResult("shell and command are required."), nil
	}
	timeout := t.timeout
	if in.TimeoutSeconds != nil && *in.TimeoutSeconds > 0 {
		timeout = min(time.Duration(*in.TimeoutSeconds)*time.Second, t.maxTimeout)
	}

	defer run.bindShell(shell, CallID(ctx))()
	res, err := run.Shells.Exec(ctx, shell, in.Command, timeout)
	created := false
	if errors.Is(err, sandbox.ErrNotFound) && ctx.Err() == nil {
		if cerr := createShell(ctx, run, shell); cerr != nil {
			if errors.Is(cerr, errEmitFailed) {
				return Result{}, cerr
			}
			return shellError(ctx, "create_shell", shell, cerr)
		}
		created = true
		res, err = run.Shells.Exec(ctx, shell, in.Command, timeout)
	}
	if err != nil {
		return shellError(ctx, "shell_exec", shell, err)
	}

	var b strings.Builder
	if created {
		fmt.Fprintf(&b, "(Created shell %q.)\n", shell)
	}
	out := validUTF8(tailString(res.OutputTail, t.maxOutput))
	cmd := oneLine(in.Command, 80)
	if res.TimedOut {
		fmt.Fprintf(&b, "The command is still running after %s. Use shell_read on shell %q to see new output, or shell_interrupt to stop it.\n\nOutput so far:\n%s",
			timeout.Round(time.Second), shell, orNone(out))
		return Result{Content: []llm.Block{llm.TextBlock(b.String())}, TimedOut: true, Summary: "still running: " + cmd}, nil
	}
	fmt.Fprintf(&b, "Exit code: %d\n\n%s", res.ExitCode, orNone(out))
	return Result{
		Content:  []llm.Block{llm.TextBlock(b.String())},
		IsError:  res.ExitCode != 0,
		ExitCode: res.ExitCode,
		Summary:  fmt.Sprintf("exit %d: %s", res.ExitCode, cmd),
	}, nil
}

func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(no output)"
	}
	return s
}

// ---- shell_read ----

type shellReadTool struct{ maxOutput int }

func (shellReadTool) Spec() llm.ToolSpec {
	return spec(toolShellRead,
		"Return the output a shell printed since the last shell_exec or shell_read on it, without waiting. Use it to follow a command that is still running, such as a dev server or a long build.",
		map[string]any{"shell": strProp(shellNameDesc)},
		"shell")
}

func (t shellReadTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Shell string `json:"shell"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if run.Shells == nil {
		return noShells(), nil
	}
	shell := strings.TrimSpace(in.Shell)
	defer run.bindShell(shell, CallID(ctx))()
	out, err := run.Shells.Read(ctx, shell)
	if err != nil {
		return shellError(ctx, "shell_read", shell, err)
	}
	if strings.TrimSpace(out) == "" {
		return textResult("no new output in "+shell, "(no new output)"), nil
	}
	return textResult("read "+shell, validUTF8(tailString(out, t.maxOutput))), nil
}

// ---- shell_interrupt ----

type shellInterruptTool struct{ maxOutput int }

func (shellInterruptTool) Spec() llm.ToolSpec {
	return spec(toolShellInterrupt,
		"Send Ctrl-C to the command running in a shell (for example a dev server or a stuck test run) and return its last output.",
		map[string]any{"shell": strProp(shellNameDesc)},
		"shell")
}

func (t shellInterruptTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Shell string `json:"shell"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if run.Shells == nil {
		return noShells(), nil
	}
	shell := strings.TrimSpace(in.Shell)
	defer run.bindShell(shell, CallID(ctx))()
	if err := run.Shells.Interrupt(ctx, shell); err != nil {
		return shellError(ctx, "shell_interrupt", shell, err)
	}
	out, err := run.Shells.Read(ctx, shell)
	if err != nil {
		return shellError(ctx, "shell_read", shell, err)
	}
	return textResult("interrupted "+shell,
		fmt.Sprintf("Sent Ctrl-C to shell %q.\n\nOutput:\n%s", shell, orNone(validUTF8(tailString(out, t.maxOutput))))), nil
}
