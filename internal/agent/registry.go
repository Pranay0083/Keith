package agent

import "time"

// ToolOptions configures DefaultTools. The zero value is usable.
type ToolOptions struct {
	// ShellTimeout is shell_exec's default wait before it returns with
	// the command still running (default 60s). The model may ask for
	// up to MaxShellTimeout.
	ShellTimeout time.Duration
	// MaxShellTimeout caps shell_exec's timeout_seconds (default 10m).
	MaxShellTimeout time.Duration
	// MaxOutputBytes caps the shell and search output a tool returns to
	// the model (default 16 KiB).
	MaxOutputBytes int
	// MaxReadBytes caps read_file's output (default 64 KiB).
	MaxReadBytes int
	// NewID makes message IDs for message_user; nil uses random hex. Use
	// the same function as Config.NewID.
	NewID func(prefix string) string
}

func (o ToolOptions) withDefaults() ToolOptions {
	if o.ShellTimeout <= 0 {
		o.ShellTimeout = 60 * time.Second
	}
	if o.MaxShellTimeout <= 0 {
		o.MaxShellTimeout = 10 * time.Minute
	}
	if o.MaxOutputBytes <= 0 {
		o.MaxOutputBytes = 16 << 10
	}
	if o.MaxReadBytes <= 0 {
		o.MaxReadBytes = 64 << 10
	}
	if o.NewID == nil {
		o.NewID = randomID
	}
	return o
}

// DefaultTools returns the V1 tool set in a fixed order. The order and
// specs are part of the cached prompt prefix, so they must not change
// within a session.
func DefaultTools(opts ToolOptions) []Tool {
	o := opts.withDefaults()
	return []Tool{
		createShellTool{},
		shellExecTool{timeout: o.ShellTimeout, maxTimeout: o.MaxShellTimeout, maxOutput: o.MaxOutputBytes},
		shellReadTool{maxOutput: o.MaxOutputBytes},
		shellInterruptTool{maxOutput: o.MaxOutputBytes},
		readFileTool{maxOutput: o.MaxReadBytes},
		writeFileTool{},
		editFileTool{},
		deleteFileTool{},
		undoEditTool{},
		messageUserTool{newID: o.NewID},
		askUserTool{},
	}
}
