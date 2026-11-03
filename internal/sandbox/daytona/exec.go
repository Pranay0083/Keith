package daytona

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

const (
	// pollFastStart and pollFastMax bound the backoff once the stream has
	// ended or failed.
	pollFastStart = 100 * time.Millisecond
	pollFastMax   = time.Second
	// maxPollFailures is how many consecutive failed status checks Exec
	// tolerates before giving up.
	maxPollFailures = 5
	// sessionCallTimeout bounds the session cleanup and final log fetch.
	sessionCallTimeout = 30 * time.Second
)

// Variables so tests can shorten them.
var (
	// pollSlow is how often Exec checks for the exit code while the log
	// stream is healthy (the stream ending is the usual signal).
	pollSlow = 2 * time.Second
	// streamGrace is how long Exec waits, after the exit code is known, for
	// the log stream to deliver the rest of the output.
	streamGrace = 3 * time.Second
)

// Exec runs req.Cmd in a fresh Daytona session. See the package doc for how
// the command is wrapped, streamed and stopped on timeout.
func (s *daytonaSandbox) Exec(ctx context.Context, req sandbox.ExecRequest, onOutput func(chunk []byte)) (*sandbox.ExecResult, error) {
	start := time.Now()
	command, err := s.buildCommand(req)
	if err != nil {
		return nil, err
	}

	proc := s.sb.Process
	sessionID := "keith-exec-" + randomHex(8)
	if err := proc.CreateSession(ctx, sessionID); err != nil {
		return nil, sbErr("exec in "+s.id+": create session", err)
	}
	defer s.deleteSession(ctx, sessionID)

	resp, err := proc.ExecuteSessionCommand(ctx, sessionID, command, true, false)
	if err != nil {
		return nil, sbErr("exec in "+s.id, err)
	}
	cmdID, _ := resp["id"].(string)
	if cmdID == "" {
		return nil, fmt.Errorf("daytona: exec in %s: no command id in response", s.id)
	}

	runCtx, cancelRun := ctx, context.CancelFunc(func() {})
	if req.Timeout > 0 {
		runCtx, cancelRun = context.WithTimeout(ctx, req.Timeout)
	}
	defer cancelRun()

	out := newExecOutput(onOutput)
	defer out.close()
	stream := s.streamLogs(runCtx, sessionID, cmdID, out)

	exitCode, err := s.waitExit(runCtx, sessionID, cmdID, stream.done)
	if err != nil {
		nOut, nErr := out.close()
		if ctx.Err() != nil {
			return nil, fmt.Errorf("daytona: exec in %s: %w", s.id, ctx.Err())
		}
		if req.Timeout > 0 && runCtx.Err() != nil {
			// The stream holds back a few bytes until it ends; the logs
			// have everything printed so far.
			s.fillOutput(ctx, sessionID, cmdID, out, nOut, nErr)
			return &sandbox.ExecResult{
				ExitCode: -1,
				Stdout:   out.stdout.String(),
				Stderr:   out.stderr.String(),
				Duration: time.Since(start),
				TimedOut: true,
			}, nil
		}
		return nil, sbErr("exec in "+s.id+": command status", err)
	}

	// The command has exited: give the stream a moment to flush the tail.
	timer := time.NewTimer(streamGrace)
	select {
	case <-stream.done:
	case <-timer.C:
	case <-ctx.Done():
	}
	timer.Stop()
	complete := false
	select {
	case <-stream.done:
		complete = stream.err == nil
	default:
	}
	nOut, nErr := out.close()
	if !complete && ctx.Err() == nil {
		s.fillOutput(ctx, sessionID, cmdID, out, nOut, nErr)
	}

	return &sandbox.ExecResult{
		ExitCode: exitCode,
		Stdout:   out.stdout.String(),
		Stderr:   out.stderr.String(),
		Duration: time.Since(start),
	}, nil
}

// buildCommand wraps req for a session shell:
//
//	/bin/sh -c 'cd <dir> && K=v ... /bin/sh -c <cmd>' < /dev/null
//
// so cd and the env never touch the session shell, and nothing waits on
// stdin. A user other than the default goes through /usr/bin/sudo, and a
// command run as root gets PATH=sandbox.SystemPath (after the env, so it
// wins). No shell or sudo is looked up by PATH.
func (s *daytonaSandbox) buildCommand(req sandbox.ExecRequest) (string, error) {
	dir := sandbox.WorkspaceDir
	if req.Dir != "" {
		dir = sbutil.ResolvePath(req.Dir)
	}
	var script strings.Builder
	script.WriteString("cd ")
	script.WriteString(sbutil.Quote(dir))
	script.WriteString(" && ")
	for _, k := range slices.Sorted(maps.Keys(req.Env)) {
		if !validEnvName(k) {
			return "", fmt.Errorf("daytona: exec: invalid environment variable name %q", k)
		}
		script.WriteString(k)
		script.WriteByte('=')
		script.WriteString(sbutil.Quote(req.Env[k]))
		script.WriteByte(' ')
	}
	if sandbox.IsRoot(req.User) {
		script.WriteString("PATH=" + sandbox.SystemPath + " ")
	}
	script.WriteString("/bin/sh -c ")
	script.WriteString(sbutil.Quote(req.Cmd))

	if u := req.User; u != "" && u != s.user {
		return "/usr/bin/sudo -n -E -H -u " + sbutil.Quote(u) + " -- /bin/sh -c " + sbutil.Quote(script.String()) + " < /dev/null", nil
	}
	return "/bin/sh -c " + sbutil.Quote(script.String()) + " < /dev/null", nil
}

func validEnvName(k string) bool {
	if k == "" {
		return false
	}
	for i, r := range k {
		switch {
		case r == '_', r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// logStream is the background reader of a command's log WebSocket.
type logStream struct {
	done chan struct{} // closed when the stream has ended and been drained
	err  error         // the stream's error; read only after done
}

// streamLogs follows the command's logs and feeds them to out. The SDK
// closes both channels when the stream ends; the drain goroutine keeps
// reading after Exec returns so the SDK never blocks on a send.
func (s *daytonaSandbox) streamLogs(ctx context.Context, sessionID, cmdID string, out *execOutput) *logStream {
	ls := &logStream{done: make(chan struct{})}
	stdoutCh := make(chan string, 64)
	stderrCh := make(chan string, 64)
	errCh := make(chan error, 1)
	go func() {
		errCh <- s.sb.Process.GetSessionCommandLogsStream(ctx, sessionID, cmdID, stdoutCh, stderrCh)
	}()
	go func() {
		defer close(ls.done)
		o, e := stdoutCh, stderrCh
		for o != nil || e != nil {
			select {
			case chunk, ok := <-o:
				if !ok {
					o = nil
					continue
				}
				out.write(false, chunk)
			case chunk, ok := <-e:
				if !ok {
					e = nil
					continue
				}
				out.write(true, chunk)
			}
		}
		ls.err = <-errCh
	}()
	return ls
}

// waitExit polls the command until it has an exit code. It polls slowly
// while the log stream is open and quickly once it has ended.
func (s *daytonaSandbox) waitExit(ctx context.Context, sessionID, cmdID string, streamDone <-chan struct{}) (int, error) {
	interval := pollSlow
	timer := time.NewTimer(interval)
	defer timer.Stop()
	failures := 0
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-streamDone:
			streamDone = nil
			interval = pollFastStart
		case <-timer.C:
		}
		info, err := s.sb.Process.GetSessionCommand(ctx, sessionID, cmdID)
		if err != nil {
			if ctx.Err() != nil {
				return 0, ctx.Err()
			}
			if failures++; failures >= maxPollFailures {
				return 0, err
			}
		} else {
			failures = 0
			if code, ok := exitCodeOf(info["exitCode"]); ok {
				return code, nil
			}
		}
		if streamDone == nil {
			interval = min(interval*2, pollFastMax)
		}
		timer.Reset(interval)
	}
}

func exitCodeOf(v any) (int, bool) {
	switch c := v.(type) {
	case int32:
		return int(c), true
	case int:
		return c, true
	case int64:
		return int(c), true
	case float64:
		return int(c), true
	}
	return 0, false
}

// fillOutput fetches the command's full logs after the stream failed or was
// cut short, and adds what the stream had not delivered (nOut and nErr
// bytes). onOutput gets the rest of stdout, then the rest of stderr.
func (s *daytonaSandbox) fillOutput(ctx context.Context, sessionID, cmdID string, out *execOutput, nOut, nErr int64) {
	ctx, cancel := context.WithTimeout(ctx, sessionCallTimeout)
	defer cancel()
	logs, err := s.sb.Process.GetSessionCommandLogs(ctx, sessionID, cmdID)
	if err != nil {
		s.p.log.Warn("daytona: exec: output may be incomplete; fetching command logs failed",
			"sandbox", s.id, "err", err)
		return
	}
	stdout, stderr := logs.Stdout, logs.Stderr
	if stdout == "" && stderr == "" {
		stdout = logs.Output // daemons that do not separate the streams
	}
	if rest := suffixAfter(stdout, nOut); rest != "" {
		out.emit(false, rest)
	}
	if rest := suffixAfter(stderr, nErr); rest != "" {
		out.emit(true, rest)
	}
}

func suffixAfter(s string, n int64) string {
	if n >= int64(len(s)) {
		return ""
	}
	return s[n:]
}

// deleteSession removes the session, which also stops a command still
// running in it. It runs even when ctx is already done.
func (s *daytonaSandbox) deleteSession(ctx context.Context, sessionID string) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), sessionCallTimeout)
	defer cancel()
	if err := s.sb.Process.DeleteSession(ctx, sessionID); err != nil {
		s.p.log.Debug("daytona: delete exec session", "sandbox", s.id, "session", sessionID, "err", err)
	}
}

// execOutput collects one command's output and forwards it to onOutput.
// The stream goroutine writes until close; after close only the Exec
// goroutine emits (the fallback fill), so onOutput is never called
// concurrently or after Exec returns.
type execOutput struct {
	onOutput func([]byte)
	stdout   *sbutil.TailBuffer
	stderr   *sbutil.TailBuffer

	mu     sync.Mutex
	closed bool
	nOut   int64
	nErr   int64
}

func newExecOutput(onOutput func([]byte)) *execOutput {
	return &execOutput{
		onOutput: onOutput,
		stdout:   sbutil.NewTailBuffer(sandbox.MaxExecOutput),
		stderr:   sbutil.NewTailBuffer(sandbox.MaxExecOutput),
	}
}

// write records a streamed chunk unless the output is closed.
func (o *execOutput) write(isErr bool, chunk string) {
	if chunk == "" {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return
	}
	if isErr {
		o.nErr += int64(len(chunk))
	} else {
		o.nOut += int64(len(chunk))
	}
	o.emit(isErr, chunk)
}

// emit appends to the tail and calls onOutput.
func (o *execOutput) emit(isErr bool, chunk string) {
	b := []byte(chunk)
	if isErr {
		_, _ = o.stderr.Write(b)
	} else {
		_, _ = o.stdout.Write(b)
	}
	if o.onOutput != nil {
		o.onOutput(b)
	}
}

// close stops streamed writes and reports how many bytes of each stream
// were delivered. It is idempotent.
func (o *execOutput) close() (nOut, nErr int64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.closed = true
	return o.nOut, o.nErr
}
