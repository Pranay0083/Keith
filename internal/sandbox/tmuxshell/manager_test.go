package tmuxshell

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
)

// step is one thing a fake command does, at an offset from its start.
type step struct {
	at   time.Duration
	out  string
	exit int  // with done: the exit status
	done bool // the command returns (PROMPT_COMMAND prints the marker)
	die  bool // the shell itself exits with status exit
}

type fakeShell struct {
	log     []byte
	id      string
	start   time.Time
	steps   []step
	running bool
	dead    bool
	status  int
}

// fakeTerm simulates bash in tmux on a fake clock: time only moves while
// a poll waits (or when a test calls advance).
type fakeTerm struct {
	mu       sync.Mutex
	now      time.Time
	shells   map[string]*fakeShell
	progs    map[string][]step
	respawns int
}

func newFakeTerm() *fakeTerm {
	return &fakeTerm{now: time.Unix(1_000_000, 0), shells: map[string]*fakeShell{}, progs: map[string][]step{}}
}

func (f *fakeTerm) clock() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

func (f *fakeTerm) advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.now = f.now.Add(d)
	for _, sh := range f.shells {
		f.deliver(sh)
	}
}

// deliver runs the steps that are due. Caller holds mu.
func (f *fakeTerm) deliver(sh *fakeShell) {
	for len(sh.steps) > 0 && sh.running && !sh.start.Add(sh.steps[0].at).After(f.now) {
		s := sh.steps[0]
		sh.steps = sh.steps[1:]
		sh.log = append(sh.log, s.out...)
		switch {
		case s.die:
			sh.running, sh.dead, sh.status = false, true, s.exit
		case s.done:
			sh.running = false
			sh.log = fmt.Appendf(sh.log, "__KEITH_D_%s_%d__\r\n", sh.id, s.exit)
		}
	}
}

func (f *fakeTerm) create(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.shells[name]; ok {
		return errExists
	}
	f.shells[name] = &fakeShell{}
	return nil
}

var idRE = regexp.MustCompile(`__KEITH_ID=([0-9a-f]+);`)

func (f *fakeTerm) send(ctx context.Context, name, cmdPath, cmd, line string, p pollSpec) (pollResult, error) {
	f.mu.Lock()
	sh, ok := f.shells[name]
	if !ok {
		f.mu.Unlock()
		return pollResult{state: paneGone}, nil
	}
	sh.id = idRE.FindStringSubmatch(line)[1]
	sh.log = append(sh.log, line+"\r\n"...)
	sh.log = fmt.Appendf(sh.log, "__KEITH_S_%s__\r\n", sh.id)
	steps, ok := f.progs[cmd]
	if !ok {
		steps = []step{{out: cmd + "\r\n", done: true}}
	}
	sh.steps = append([]step(nil), steps...)
	sh.start = f.now
	sh.running = true
	f.mu.Unlock()
	return f.poll(ctx, name, p)
}

func (f *fakeTerm) poll(ctx context.Context, name string, p pollSpec) (pollResult, error) {
	if err := ctx.Err(); err != nil {
		return pollResult{}, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	sh, ok := f.shells[name]
	if !ok {
		return pollResult{state: paneGone}, nil
	}
	f.deliver(sh)
	if int64(len(sh.log)) <= p.off && !sh.dead {
		// Wait for the next step or the end of the wait.
		target := f.now.Add(p.wait)
		if sh.running && len(sh.steps) > 0 {
			if next := sh.start.Add(sh.steps[0].at); next.Before(target) {
				target = next
			}
		}
		f.now = target
		f.deliver(sh)
	}
	pr := pollResult{from: p.off, data: append([]byte(nil), sh.log[p.off:]...)}
	if sh.dead {
		pr.state, pr.status = paneDead, sh.status
	}
	return pr, nil
}

func (f *fakeTerm) interrupt(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	sh, ok := f.shells[name]
	if !ok {
		return errGone
	}
	sh.log = append(sh.log, "^C\r\n"...)
	if sh.running {
		sh.running = false
		sh.log = fmt.Appendf(sh.log, "__KEITH_D_%s_130__\r\n", sh.id)
	}
	return nil
}

func (f *fakeTerm) respawn(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.respawns++
	f.shells[name].dead = false
	return nil
}

func (f *fakeTerm) kill(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.shells, name)
	return nil
}

func (f *fakeTerm) list(ctx context.Context) (map[string]int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := map[string]int64{}
	for n, sh := range f.shells {
		out[n] = int64(len(sh.log))
	}
	return out, nil
}

func (f *fakeTerm) cmdPath(name string) string { return "/tmp/" + name + ".cmd" }

func newTestManager(t *testing.T, opts ...Option) (*Manager, *fakeTerm) {
	t.Helper()
	f := newFakeTerm()
	opts = append([]Option{WithTimeouts(10*time.Second, 3*time.Second, time.Minute)}, opts...)
	m := newManager(f, opts...)
	m.now = f.clock
	if err := m.Create(context.Background(), "main"); err != nil {
		t.Fatal(err)
	}
	return m, f
}

func TestExecExitCodeAndOutput(t *testing.T) {
	ctx := context.Background()
	m, f := newTestManager(t)
	f.progs["make"] = []step{
		{at: time.Second, out: "\x1b[1mbuilding\x1b[0m\r\n"},
		{at: 2 * time.Second, out: "error: x\r\n", exit: 2, done: true},
	}
	res, err := m.Exec(ctx, "main", "make", 0)
	if err != nil {
		t.Fatal(err)
	}
	want := sandbox.ShellResult{OutputTail: "building\nerror: x\n", ExitCode: 2, Duration: 2 * time.Second}
	if res != want {
		t.Fatalf("got %+v\nwant %+v", res, want)
	}
	// The shell is free again.
	res, err = m.Exec(ctx, "main", "echo hi", 0)
	if err != nil || res.ExitCode != 0 || res.OutputTail != "echo hi\n" {
		t.Fatalf("second exec = %+v, %v", res, err)
	}
}

func TestExecTimeoutThenReadInterrupt(t *testing.T) {
	ctx := context.Background()
	m, f := newTestManager(t)
	f.progs["npm run dev"] = []step{
		{at: time.Second, out: "ready on :3000\r\n"},
		{at: 30 * time.Second, out: "GET /\r\n"},
	}
	res, err := m.Exec(ctx, "main", "npm run dev", 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if !res.TimedOut || res.ExitCode != -1 || res.OutputTail != "ready on :3000\n" || res.Duration != 5*time.Second {
		t.Fatalf("timeout result = %+v", res)
	}
	if _, err := m.Exec(ctx, "main", "ls", 0); !errors.Is(err, sandbox.ErrShellBusy) {
		t.Fatalf("exec on busy shell: %v", err)
	}
	if out, err := m.Read(ctx, "main"); err != nil || out != "" {
		t.Fatalf("Read before new output = %q, %v", out, err)
	}
	f.advance(30 * time.Second)
	if out, err := m.Read(ctx, "main"); err != nil || out != "GET /\n" {
		t.Fatalf("Read = %q, %v", out, err)
	}
	if err := m.Interrupt(ctx, "main"); err != nil {
		t.Fatal(err)
	}
	if out, err := m.Read(ctx, "main"); err != nil || out != "^C\n[command exited with code 130]" {
		t.Fatalf("Read after interrupt = %q, %v", out, err)
	}
	if res, err := m.Exec(ctx, "main", "ls", 0); err != nil || res.ExitCode != 0 {
		t.Fatalf("exec after interrupt = %+v, %v", res, err)
	}
}

func TestExecFinishedAfterTimeoutFreesShell(t *testing.T) {
	ctx := context.Background()
	m, f := newTestManager(t)
	f.progs["slow"] = []step{{at: 20 * time.Second, out: "done\r\n", exit: 0, done: true}}
	res, _ := m.Exec(ctx, "main", "slow", 5*time.Second)
	if !res.TimedOut {
		t.Fatalf("want timeout, got %+v", res)
	}
	f.advance(20 * time.Second)
	// Exec notices the previous command finished and runs the new one.
	if res, err := m.Exec(ctx, "main", "ls", 0); err != nil || res.ExitCode != 0 || res.OutputTail != "ls\n" {
		t.Fatalf("exec = %+v, %v", res, err)
	}
}

func TestProgressiveTimeout(t *testing.T) {
	ctx := context.Background()
	chatty := func(until time.Duration) []step {
		var s []step
		for at := time.Second; at <= until; at += time.Second {
			s = append(s, step{at: at, out: fmt.Sprintf("tick %v\r\n", at)})
		}
		return append(s, step{at: time.Hour, done: true})
	}

	t.Run("extends while printing, stops after silence", func(t *testing.T) {
		m, f := newTestManager(t)
		f.progs["build"] = chatty(14 * time.Second) // timeout 10s, silence 3s
		res, err := m.Exec(ctx, "main", "build", 0)
		if err != nil {
			t.Fatal(err)
		}
		if !res.TimedOut || res.Duration != 17*time.Second || !strings.HasSuffix(res.OutputTail, "tick 14s\n") {
			t.Fatalf("got %+v", res)
		}
	})
	t.Run("hard max", func(t *testing.T) {
		m, f := newTestManager(t, WithTimeouts(0, 0, 25*time.Second))
		f.progs["build"] = chatty(time.Minute)
		res, _ := m.Exec(ctx, "main", "build", 0)
		if !res.TimedOut || res.Duration != 25*time.Second {
			t.Fatalf("got %+v", res)
		}
	})
	t.Run("output tail is capped", func(t *testing.T) {
		m, f := newTestManager(t, WithTailBytes(64))
		f.progs["spam"] = []step{{out: strings.Repeat("0123456789\r\n", 100), done: true}}
		res, _ := m.Exec(ctx, "main", "spam", 0)
		if !strings.HasPrefix(res.OutputTail, "[... ") || !strings.HasSuffix(res.OutputTail, "0123456789\n") || len(res.OutputTail) > 64+64 {
			t.Fatalf("tail = %q", res.OutputTail)
		}
	})
}

func TestShellDiesAndRestarts(t *testing.T) {
	ctx := context.Background()
	m, f := newTestManager(t)
	f.progs["exit 3"] = []step{{at: time.Second, exit: 3, die: true}}
	res, err := m.Exec(ctx, "main", "exit 3", 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.ExitCode != 3 || res.TimedOut || !strings.Contains(res.OutputTail, "shell exited with status 3") || f.respawns != 1 {
		t.Fatalf("got %+v (respawns %d)", res, f.respawns)
	}
	if res, err := m.Exec(ctx, "main", "pwd", 0); err != nil || res.ExitCode != 0 {
		t.Fatalf("exec after restart = %+v, %v", res, err)
	}
}

func TestLiveOutput(t *testing.T) {
	ctx := context.Background()
	var got strings.Builder
	m, f := newTestManager(t, WithOutput(func(shell string, chunk []byte) {
		if shell != "main" {
			t.Errorf("shell = %q", shell)
		}
		got.Write(chunk)
	}))
	f.progs["t"] = []step{{at: time.Second, out: "\x1b[32mok\x1b[0m\r\n", done: true}}
	if _, err := m.Exec(ctx, "main", "t", 0); err != nil {
		t.Fatal(err)
	}
	if got.String() != "\x1b[32mok\x1b[0m\r\n" {
		t.Fatalf("live output = %q", got.String())
	}
}

func TestShellErrors(t *testing.T) {
	ctx := context.Background()
	m, f := newTestManager(t)
	if err := m.Create(ctx, "main"); !errors.Is(err, sandbox.ErrShellExists) {
		t.Fatalf("duplicate create: %v", err)
	}
	if err := m.Create(ctx, "bad name"); err == nil {
		t.Fatal("invalid name accepted")
	}
	if _, err := m.Exec(ctx, "nope", "ls", 0); !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("unknown shell: %v", err)
	}
	if _, err := m.Read(ctx, "nope"); !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("unknown shell read: %v", err)
	}
	if err := m.Create(ctx, "web"); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(m.List(), ","); got != "main,web" {
		t.Fatalf("List = %s", got)
	}
	// A session killed behind the manager's back.
	_ = f.kill(ctx, "web")
	if _, err := m.Exec(ctx, "web", "ls", 0); !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("exec on vanished shell: %v", err)
	}
	if got := strings.Join(m.List(), ","); got != "main" {
		t.Fatalf("List after vanish = %s", got)
	}
	// A tmux session left over from another manager.
	m2 := newManager(f)
	if err := m2.Create(ctx, "main"); !errors.Is(err, sandbox.ErrShellExists) {
		t.Fatalf("create over leftover session: %v", err)
	}
	adopted, err := m2.Reattach(ctx)
	if err != nil || strings.Join(adopted, ",") != "main" {
		t.Fatalf("Reattach = %v, %v", adopted, err)
	}
	if res, err := m2.Exec(ctx, "main", "ls", 0); err != nil || res.OutputTail != "ls\n" {
		t.Fatalf("exec on adopted shell = %+v, %v", res, err)
	}
}

func TestExecContextCancel(t *testing.T) {
	m, f := newTestManager(t)
	f.progs["hang"] = []step{{at: time.Hour, done: true}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Exec(ctx, "main", "hang", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}
