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
	// MaxOutputBytes caps the output a tool returns to the model
	// (default 16 KiB).
	MaxOutputBytes int
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
		messageUserTool{newID: o.NewID},
		askUserTool{},
	}
}
