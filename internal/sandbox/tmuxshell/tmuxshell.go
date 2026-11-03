// Package tmuxshell implements sandbox.Shells with tmux inside any
// sandbox.Sandbox: the agent's create_shell, shell_exec, shell_read and
// shell_interrupt tools.
//
// Each named shell is a tmux session (k-<name>) on a private tmux socket,
// running bash with a generated rc file. tmux pipe-pane appends the pane's
// raw output to a log file, which the Manager reads by offset through
// sandbox.Exec, so it works on any provider with no daemon in the sandbox.
//
// Exec writes the command to a file and types a short line that prints a
// start marker carrying a random ID and sources the file; bash's
// PROMPT_COMMAND then prints a done marker with the ID and $?. The Manager
// recovers the exit status from that marker, which also fires after
// Ctrl-C (status 130). See scan.go for how markers are parsed.
//
// Exec waits with the progressive timeout described on WithTimeouts; on
// timeout it returns the output tail with ExitCode -1 and TimedOut set and
// the command keeps running: Read returns what it prints next and
// Interrupt stops it. A shell whose bash exits (exit, set -e) is restarted
// in WorkspaceDir and the exit status is reported.
//
// Requirements in the sandbox: tmux, bash 5+, and coreutils-style stat,
// tail, head, base64 and grep (Debian and Alpine images both qualify).
//
// Limitations: shells assume a bash prompt. A command that replaces the
// shell with something that is not bash (python, a non-bash sh) never
// prints the done marker, so Exec times out until it exits. Logs grow
// without rotation for the sandbox's life.
package tmuxshell

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// Defaults for options.
const (
	// DefaultTailBytes caps ShellResult.OutputTail and Read's result.
	DefaultTailBytes = 16 << 10
	// DefaultStateDir holds the rc file, logs and command files.
	DefaultStateDir = "/tmp/keith-shells"
	// DefaultSocket is the tmux socket name (tmux -L).
	DefaultSocket = "keith"
)

// Wait slices: Exec polls the sandbox at most maxPollWait at a time, and
// Interrupt waits up to interruptWait for the command to exit.
const (
	maxPollWait   = time.Second
	interruptWait = 2 * time.Second
)

var nameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$`)

// Manager is a sandbox.Shells over tmux in one sandbox. It is safe for
// concurrent use. Calls for one shell are serialized, except Interrupt,
// which never waits for a running Exec (it is how a stuck Exec is
// stopped).
type Manager struct {
	t         term
	onOutput  func(shell string, chunk []byte)
	timeout   time.Duration
	silence   time.Duration
	maxWait   time.Duration
	tailBytes int
	now       func() time.Time
	newID     func() string

	mu     sync.Mutex
	shells map[string]*shell
	order  []string
}

var _ sandbox.Shells = (*Manager)(nil)

// Option configures a Manager.
type Option func(*Manager)

// WithOutput streams live terminal output: fn receives each shell's raw
// output (escape sequences included, markers and the typed command line
// removed) as it is read, for tool_call.output events. It is called from
// the goroutine of the Exec, Read or Interrupt call that read the output,
// never concurrently for one shell.
func WithOutput(fn func(shell string, chunk []byte)) Option {
	return func(m *Manager) { m.onOutput = fn }
}

// WithTimeouts sets the progressive timeout. Exec's timeout argument (or
// defaultTimeout when it is 0) is how long Exec waits for the command; if
// the command is still printing when that passes, Exec keeps waiting
// until it has printed nothing for silence, but never longer than maxWait
// in total. Zero values keep the defaults (DefaultTimeout,
// DefaultSilence, DefaultMaxWait).
func WithTimeouts(defaultTimeout, silence, maxWait time.Duration) Option {
	return func(m *Manager) {
		if defaultTimeout > 0 {
			m.timeout = defaultTimeout
		}
		if silence > 0 {
			m.silence = silence
		}
		if maxWait > 0 {
			m.maxWait = maxWait
		}
	}
}

// WithTailBytes caps the output returned by Exec and Read (default
// DefaultTailBytes). Longer output keeps its end, with a note saying how
// much was omitted.
func WithTailBytes(n int) Option {
	return func(m *Manager) {
		if n > 0 {
			m.tailBytes = n
		}
	}
}

// WithStateDir sets the directory in the sandbox for the rc file, logs
// and command files (default DefaultStateDir).
func WithStateDir(dir string) Option {
	return func(m *Manager) {
		if st, ok := m.t.(*sbTerm); ok && dir != "" {
			st.dir = strings.TrimRight(dir, "/")
		}
	}
}

// WithUser runs tmux, and so every shell, as user (default: the
// sandbox's default user).
func WithUser(user string) Option {
	return func(m *Manager) {
		if st, ok := m.t.(*sbTerm); ok {
			st.user = user
		}
	}
}

// New returns a Manager for shells in sb. It does not touch the sandbox
// until the first call.
func New(sb sandbox.Sandbox, opts ...Option) *Manager {
	return newManager(&sbTerm{sb: sb, dir: DefaultStateDir, socket: DefaultSocket}, opts...)
}

func newManager(t term, opts ...Option) *Manager {
	m := &Manager{
		t:         t,
		timeout:   DefaultTimeout,
		silence:   DefaultSilence,
		maxWait:   DefaultMaxWait,
		tailBytes: DefaultTailBytes,
		now:       time.Now,
		newID:     randomID,
		shells:    map[string]*shell{},
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

func randomID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

type phase int

const (
	phaseIdle     phase = iota
	phasePreStart       // command typed, start marker not seen yet
	phaseRunning
)

type shell struct {
	name string
	op   sync.Mutex // serializes Create/Exec/Read/refresh for this shell

	// Guarded by op.
	gone     bool
	off      int64 // log bytes consumed
	phase    phase
	id       string
	sc       *scanner
	strip    stripper
	out      *sbutil.TailBuffer // current command's output (escapes stripped)
	read     *sbutil.TailBuffer // output since the last Exec or Read returned
	finished bool               // done marker seen and not yet reported
	exitCode int
}

func (m *Manager) newShell(name string, off int64) *shell {
	return &shell{
		name: name,
		off:  off,
		sc:   newScanner(""),
		out:  sbutil.NewTailBuffer(4 * m.tailBytes),
		read: sbutil.NewTailBuffer(4 * m.tailBytes),
	}
}

func notFound(name string) error {
	return fmt.Errorf("tmuxshell: shell %q: %w", name, sandbox.ErrNotFound)
}

func (m *Manager) lookup(name string) (*shell, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	sh, ok := m.shells[name]
	if !ok {
		return nil, notFound(name)
	}
	return sh, nil
}

// drop forgets a shell whose tmux session is gone. Caller holds sh.op.
func (m *Manager) drop(sh *shell) {
	sh.gone = true
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.shells[sh.name] == sh {
		delete(m.shells, sh.name)
		m.order = slices.DeleteFunc(m.order, func(n string) bool { return n == sh.name })
	}
}

// Create implements sandbox.Shells. Names must match
// [A-Za-z0-9][A-Za-z0-9_-]{0,62}.
func (m *Manager) Create(ctx context.Context, name string) error {
	if !nameRE.MatchString(name) {
		return fmt.Errorf("tmuxshell: invalid shell name %q: use letters, digits, '-' and '_' (max 63)", name)
	}
	sh := m.newShell(name, 0)
	sh.op.Lock()
	defer sh.op.Unlock()

	m.mu.Lock()
	if _, ok := m.shells[name]; ok {
		m.mu.Unlock()
		return fmt.Errorf("tmuxshell: shell %q: %w", name, sandbox.ErrShellExists)
	}
	m.shells[name] = sh
	m.mu.Unlock()

	if err := m.t.create(ctx, name); err != nil {
		m.drop(sh)
		if errors.Is(err, errExists) {
			return fmt.Errorf("tmuxshell: shell %q (tmux session left by an earlier manager; see Reattach): %w", name, sandbox.ErrShellExists)
		}
		return err
	}
	m.mu.Lock()
	m.order = append(m.order, name)
	m.mu.Unlock()
	return nil
}

// Reattach adopts shells that already exist in the sandbox but not in
// this Manager, for a worker that resumes a session another process
// started. Adopted shells start reading at the end of their log and are
// treated as idle. It returns the adopted names.
func (m *Manager) Reattach(ctx context.Context) ([]string, error) {
	found, err := m.t.list(ctx)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(found))
	for n := range found {
		names = append(names, n)
	}
	slices.Sort(names)
	m.mu.Lock()
	defer m.mu.Unlock()
	var adopted []string
	for _, n := range names {
		if _, ok := m.shells[n]; ok || !nameRE.MatchString(n) {
			continue
		}
		m.shells[n] = m.newShell(n, found[n])
		m.order = append(m.order, n)
		adopted = append(adopted, n)
	}
	return adopted, nil
}

// List implements sandbox.Shells.
func (m *Manager) List() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.order)
}

// Close kills a shell and its running command, and forgets it.
func (m *Manager) Close(ctx context.Context, name string) error {
	sh, err := m.lookup(name)
	if err != nil {
		return err
	}
	if err := m.t.kill(ctx, name); err != nil {
		return err
	}
	sh.op.Lock()
	defer sh.op.Unlock()
	m.drop(sh)
	return nil
}

// Exec implements sandbox.Shells.
func (m *Manager) Exec(ctx context.Context, name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
	sh, err := m.lookup(name)
	if err != nil {
		return sandbox.ShellResult{}, err
	}
	sh.op.Lock()
	defer sh.op.Unlock()
	if sh.gone {
		return sandbox.ShellResult{}, notFound(name)
	}
	if sh.phase != phaseIdle {
		// The previous command timed out; it may have finished since.
		if err := m.refresh(ctx, sh, 0); err != nil {
			return sandbox.ShellResult{}, err
		}
		if sh.phase != phaseIdle {
			return sandbox.ShellResult{}, fmt.Errorf("tmuxshell: shell %q: %w", name, sandbox.ErrShellBusy)
		}
	}
	if timeout <= 0 {
		timeout = m.timeout
	}

	id := m.newID()
	sh.phase = phasePreStart
	sh.id = id
	sh.sc.reset(id)
	sh.strip.reset()
	sh.out.Reset()
	sh.read.Reset()
	sh.finished = false

	start := m.now()
	dl := newDeadline(start, timeout, m.silence, m.maxWait)
	cmdPath := m.t.cmdPath(name)
	line := " __KEITH_ID=" + id + `; printf '` + startPrefix + `%s` + markerTail + `\n' "$__KEITH_ID"; . ` + sbutil.Quote(cmdPath)
	pr, err := m.t.send(ctx, name, cmdPath, cmd, line, m.spec(sh, dl, start))
	if err != nil {
		// Assume the line was not typed, so the shell stays usable. If it
		// was, it runs after whatever is running now and its markers are
		// ignored.
		sh.phase = phaseIdle
		sh.sc.reset("")
		return sandbox.ShellResult{}, err
	}
	for {
		grew := m.consume(sh, pr)
		now := m.now()
		if sh.finished {
			return m.result(sh, start, now, false), nil
		}
		switch pr.state {
		case paneDead:
			if err := m.restart(ctx, sh, pr.status); err != nil {
				return sandbox.ShellResult{}, err
			}
			return m.result(sh, start, now, false), nil
		case paneGone:
			res := m.result(sh, start, now, true)
			m.drop(sh)
			return res, fmt.Errorf("tmuxshell: shell %q was closed: %w", name, sandbox.ErrNotFound)
		}
		dl.observe(now, grew)
		if dl.expired(now) {
			return m.result(sh, start, now, true), nil
		}
		if err := ctx.Err(); err != nil {
			return m.result(sh, start, now, true), err
		}
		pr, err = m.t.poll(ctx, name, m.spec(sh, dl, now))
		if err != nil {
			if ctx.Err() != nil {
				return m.result(sh, start, m.now(), true), ctx.Err()
			}
			return sandbox.ShellResult{}, err
		}
	}
}

// spec is the next poll for an Exec.
func (m *Manager) spec(sh *shell, dl *deadline, now time.Time) pollSpec {
	return pollSpec{off: sh.off, wait: min(maxPollWait, dl.remaining(now)), mark: sh.mark()}
}

// mark is the done-marker prefix to wait for, if a command is running.
func (sh *shell) mark() string {
	if sh.phase == phaseIdle {
		return ""
	}
	return donePrefix + sh.id + "_"
}

// result builds a ShellResult from the current command's output. It
// resets the Read buffer: Read returns what comes after.
func (m *Manager) result(sh *shell, start, now time.Time, timedOut bool) sandbox.ShellResult {
	r := sandbox.ShellResult{
		OutputTail: tail(normalize(sh.out.String()), m.tailBytes, sh.out.Dropped()),
		Duration:   now.Sub(start),
	}
	if timedOut {
		r.ExitCode = -1
		r.TimedOut = true
	} else {
		r.ExitCode = sh.exitCode
		sh.finished = false
	}
	sh.read.Reset()
	return r
}

// Read implements sandbox.Shells. If the command that was running
// finished, the result ends with a line giving its exit code.
func (m *Manager) Read(ctx context.Context, name string) (string, error) {
	sh, err := m.lookup(name)
	if err != nil {
		return "", err
	}
	sh.op.Lock()
	defer sh.op.Unlock()
	if sh.gone {
		return "", notFound(name)
	}
	if err := m.refresh(ctx, sh, 0); err != nil {
		return "", err
	}
	out := tail(normalize(sh.read.String()), m.tailBytes, sh.read.Dropped())
	if sh.finished {
		if out != "" && !strings.HasSuffix(out, "\n") {
			out += "\n"
		}
		out += "[command exited with code " + strconv.Itoa(sh.exitCode) + "]"
		sh.finished = false
	}
	sh.read.Reset()
	return out, nil
}

// Interrupt implements sandbox.Shells: it sends Ctrl-C, then waits
// briefly for the command to exit so the shell is free for the next Exec.
func (m *Manager) Interrupt(ctx context.Context, name string) error {
	sh, err := m.lookup(name)
	if err != nil {
		return err
	}
	if err := m.t.interrupt(ctx, name); err != nil {
		if errors.Is(err, errGone) {
			sh.op.Lock()
			m.drop(sh)
			sh.op.Unlock()
			return notFound(name)
		}
		return err
	}
	if !sh.op.TryLock() {
		return nil // an Exec is waiting on this shell and will see the exit
	}
	defer sh.op.Unlock()
	if sh.gone || sh.phase == phaseIdle {
		return nil
	}
	return m.refresh(ctx, sh, interruptWait)
}

// refresh polls once and processes the result. Caller holds sh.op.
func (m *Manager) refresh(ctx context.Context, sh *shell, wait time.Duration) error {
	pr, err := m.t.poll(ctx, sh.name, pollSpec{off: sh.off, wait: wait, mark: sh.mark()})
	if err != nil {
		return err
	}
	m.consume(sh, pr)
	switch pr.state {
	case paneDead:
		return m.restart(ctx, sh, pr.status)
	case paneGone:
		m.drop(sh)
		return fmt.Errorf("tmuxshell: shell %q was closed: %w", sh.name, sandbox.ErrNotFound)
	}
	return nil
}

// restart handles a shell whose bash exited: the running command (if any)
// finishes with the shell's exit status, and a fresh bash starts.
func (m *Manager) restart(ctx context.Context, sh *shell, status int) error {
	note := fmt.Sprintf("\r\n[shell exited with status %d; started a new shell in %s]\r\n", status, sandbox.WorkspaceDir)
	running := sh.phase != phaseIdle
	m.emit(sh, []byte(note), running)
	if running {
		m.finish(sh, status)
	}
	return m.t.respawn(ctx, sh.name)
}

// consume feeds a poll result through the marker scanner. It reports
// whether the running command printed anything. Caller holds sh.op.
func (m *Manager) consume(sh *shell, pr pollResult) (grew bool) {
	if pr.from > sh.off {
		skipped := pr.from - sh.off
		sh.sc.flush()
		sh.strip.reset()
		if sh.phase == phasePreStart {
			sh.phase = phaseRunning
		}
		m.emit(sh, fmt.Appendf(nil, "\r\n[... %d bytes of output skipped ...]\r\n", skipped), sh.phase == phaseRunning)
		grew = true
		if pr.skippedDone != "" && sh.phase != phaseIdle {
			code := -1
			if i := strings.LastIndexByte(strings.TrimSuffix(pr.skippedDone, markerTail), '_'); i >= 0 {
				code, _ = strconv.Atoi(strings.TrimSuffix(pr.skippedDone[i+1:], markerTail))
			}
			m.finish(sh, code)
		}
	}
	sh.off = pr.from + int64(len(pr.data))
	for _, seg := range sh.sc.feed(pr.data) {
		switch seg.kind {
		case evText:
			switch sh.phase {
			case phasePreStart:
				// The echo of the typed line (and anything printed before
				// it): not part of the command's output.
			case phaseRunning:
				grew = true
				m.emit(sh, seg.raw, true)
			case phaseIdle:
				m.emit(sh, seg.raw, false)
			}
		case evStart:
			if sh.phase == phasePreStart {
				sh.phase = phaseRunning
			}
		case evDone:
			m.finish(sh, seg.code)
		}
	}
	if sh.phase == phaseIdle && sh.sc.id != "" {
		// Done: stop looking for this command's markers and release
		// anything held back as possible marker text.
		skip := sh.sc.skipNL
		rest := sh.sc.flush()
		sh.sc.reset("")
		sh.sc.skipNL = skip
		for _, seg := range sh.sc.feed(rest) {
			m.emit(sh, seg.raw, false)
		}
	}
	return grew
}

func (m *Manager) finish(sh *shell, code int) {
	sh.phase = phaseIdle
	sh.exitCode = code
	sh.finished = true
}

// emit delivers terminal output: raw to the live callback, stripped of
// escapes to the Read buffer and, for the running command, to its output.
func (m *Manager) emit(sh *shell, raw []byte, running bool) {
	if len(raw) == 0 {
		return
	}
	if m.onOutput != nil {
		m.onOutput(sh.name, raw)
	}
	vis := sh.strip.write(raw)
	if running {
		sh.out.Write(vis)
	}
	sh.read.Write(vis)
}
