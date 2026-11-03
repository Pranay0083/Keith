package tmuxshell

import "time"

// Timeout defaults. See WithTimeouts.
const (
	DefaultTimeout = 60 * time.Second
	DefaultSilence = 15 * time.Second
	DefaultMaxWait = 10 * time.Minute
)

// deadline is the progressive timeout ported from Keith's original
// shell_exec tool: a command gets its timeout, but if it is still printing
// when the timeout passes, the wait is extended until it has been silent
// for the silence limit, and never beyond the hard maximum. When the
// deadline passes, Exec returns the output so far with TimedOut set; the
// command keeps running in its shell.
//
// Concretely, with start s, timeout T, silence limit S, hard max M and
// the time of the latest new output g, the wait ends at
//
//	min(max(s+T, g+S), s+M)
//
// (unless the command finishes first).
type deadline struct {
	start, end, hard time.Time
	silence          time.Duration
}

func newDeadline(start time.Time, timeout, silence, max time.Duration) *deadline {
	if timeout > max {
		timeout = max
	}
	return &deadline{
		start:   start,
		end:     start.Add(timeout),
		hard:    start.Add(max),
		silence: silence,
	}
}

// observe records the output state at now: grew is true if the command
// printed something since the last observation.
func (d *deadline) observe(now time.Time, grew bool) {
	if grew {
		if ext := now.Add(d.silence); ext.After(d.end) {
			d.end = ext
		}
	}
}

// at returns when the wait ends, given the observations so far.
func (d *deadline) at() time.Time {
	if d.hard.Before(d.end) {
		return d.hard
	}
	return d.end
}

// expired reports whether the wait is over at now.
func (d *deadline) expired(now time.Time) bool { return !now.Before(d.at()) }

// remaining is how long until the wait ends (0 if it has).
func (d *deadline) remaining(now time.Time) time.Duration {
	if r := d.at().Sub(now); r > 0 {
		return r
	}
	return 0
}
