package tmuxshell_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbtest"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
)

type liveLog struct {
	mu  sync.Mutex
	out map[string]*strings.Builder
}

func (l *liveLog) write(shell string, chunk []byte) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.out[shell] == nil {
		l.out[shell] = &strings.Builder{}
	}
	l.out[shell].Write(chunk)
}

func (l *liveLog) get(shell string) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.out[shell] == nil {
		return ""
	}
	return l.out[shell].String()
}

func TestTmuxShellsInDocker(t *testing.T) {
	p := sbtest.DockerProvider(t)
	sb := sbtest.Sandbox(t, p, sandbox.Spec{})
	ctx := context.Background()
	live := &liveLog{out: map[string]*strings.Builder{}}
	m := tmuxshell.New(sb, tmuxshell.WithOutput(live.write), tmuxshell.WithTimeouts(20*time.Second, 2*time.Second, time.Minute))

	exec := func(t *testing.T, shell, cmd string, timeout time.Duration) sandbox.ShellResult {
		t.Helper()
		res, err := m.Exec(ctx, shell, cmd, timeout)
		if err != nil {
			t.Fatalf("Exec(%s, %q): %v", shell, cmd, err)
		}
		return res
	}

	for _, n := range []string{"server", "work"} {
		if err := m.Create(ctx, n); err != nil {
			t.Fatalf("Create %s: %v", n, err)
		}
	}
	if err := m.Create(ctx, "work"); !errors.Is(err, sandbox.ErrShellExists) {
		t.Fatalf("duplicate Create: %v", err)
	}
	if got := strings.Join(m.List(), ","); got != "server,work" {
		t.Fatalf("List = %s", got)
	}

	t.Run("exit codes and persistent state", func(t *testing.T) {
		if r := exec(t, "work", "echo hello", 0); r.ExitCode != 0 || r.OutputTail != "hello\n" || r.TimedOut {
			t.Fatalf("echo = %+v", r)
		}
		if r := exec(t, "work", "false", 0); r.ExitCode != 1 || r.OutputTail != "" {
			t.Fatalf("false = %+v", r)
		}
		if r := exec(t, "work", "ls /nonexistent", 0); r.ExitCode != 2 || !strings.Contains(r.OutputTail, "No such file") {
			t.Fatalf("ls = %+v", r)
		}
		if r := exec(t, "work", "(exit 42)", 0); r.ExitCode != 42 {
			t.Fatalf("exit 42 = %+v", r)
		}
		exec(t, "work", "cd /tmp && export FOO=bar", 0)
		if r := exec(t, "work", `echo "$(pwd) $FOO"`, 0); r.OutputTail != "/tmp bar\n" {
			t.Fatalf("state = %+v", r)
		}
		exec(t, "work", "cd /workspace", 0)
	})

	t.Run("awkward commands", func(t *testing.T) {
		heredoc := "cat <<'EOF'\nit's \"quoted\" $HOME\t`tab`\nEOF\nprintf 'no newline'"
		if r := exec(t, "work", heredoc, 0); r.OutputTail != "it's \"quoted\" $HOME\t`tab`\nno newline" || r.ExitCode != 0 {
			t.Fatalf("heredoc = %q (exit %d)", r.OutputTail, r.ExitCode)
		}
		fake := "echo __KEITH_D_deadbeefdeadbeef_0__; echo '__KEITH_S_x__'; echo __KEITH_D_"
		if r := exec(t, "work", fake, 0); r.OutputTail != "__KEITH_D_deadbeefdeadbeef_0__\n__KEITH_S_x__\n__KEITH_D_\n" {
			t.Fatalf("marker-like output = %q", r.OutputTail)
		}
		if r := exec(t, "work", `printf '\033[1;31mred\033[0m\n'; printf '10%%\r50%%\r100%%\n'`, 0); r.OutputTail != "red\n100%\n" {
			t.Fatalf("ansi = %q", r.OutputTail)
		}
		if r := exec(t, "work", "echo 'syntax error (", 0); r.ExitCode == 0 {
			t.Fatalf("syntax error exit = %+v", r)
		}
		big := exec(t, "work", "seq 1 200000", 0)
		if big.ExitCode != 0 || !strings.HasSuffix(big.OutputTail, "199999\n200000\n") || len(big.OutputTail) > tmuxshell.DefaultTailBytes+100 ||
			!strings.HasPrefix(big.OutputTail, "[... ") {
			t.Fatalf("big output: exit %d, %d bytes, starts %q", big.ExitCode, len(big.OutputTail), big.OutputTail[:min(60, len(big.OutputTail))])
		}
		long := strings.Repeat("x", 100<<10) // over the inline limit
		if r := exec(t, "work", "echo "+long+" | wc -c", 0); strings.TrimSpace(r.OutputTail) != "102401" {
			t.Fatalf("long command = %q", r.OutputTail)
		}
	})

	t.Run("dev server: timeout, read, interrupt, parallel shell", func(t *testing.T) {
		if err := sb.WriteFile(ctx, "www/index.html", []byte("ok"), 0); err != nil {
			t.Fatal(err)
		}
		server := `echo starting; sleep 0.3; echo "ready on :3000"; busybox httpd -f -p 3000 -h /workspace/www & srv=$!; ` +
			`sleep 3.7; echo "late output"; wait $srv`
		start := time.Now()
		r := exec(t, "server", server, time.Second)
		if !r.TimedOut || r.ExitCode != -1 || r.OutputTail != "starting\nready on :3000\n" {
			t.Fatalf("server exec = %+v", r)
		}
		// The progressive timeout ends 2s (the silence limit) after "ready".
		if d := time.Since(start); d < time.Second || d > 6*time.Second {
			t.Fatalf("server exec took %v", d)
		}
		if _, err := m.Exec(ctx, "server", "echo hi", 0); !errors.Is(err, sandbox.ErrShellBusy) {
			t.Fatalf("exec on busy shell: %v", err)
		}

		// The other shell works in parallel with a busy one.
		var wg sync.WaitGroup
		results := make([]sandbox.ShellResult, 2)
		errs := make([]error, 2)
		for i, cmd := range []string{"busybox wget -qO- http://127.0.0.1:3000/index.html", "sleep 1; echo slept"} {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if i == 1 {
					if err := m.Create(ctx, "extra"); err != nil {
						errs[i] = err
						return
					}
					results[i], errs[i] = m.Exec(ctx, "extra", cmd, 0)
					return
				}
				results[i], errs[i] = m.Exec(ctx, "work", cmd, 0)
			}()
		}
		wg.Wait()
		if errs[0] != nil || results[0].OutputTail != "ok" || results[1].OutputTail != "slept\n" || errs[1] != nil {
			t.Fatalf("parallel: %+v %v / %+v %v", results[0], errs[0], results[1], errs[1])
		}

		deadline := time.Now().Add(10 * time.Second)
		var got string
		for !strings.Contains(got, "late output") && time.Now().Before(deadline) {
			out, err := m.Read(ctx, "server")
			if err != nil {
				t.Fatal(err)
			}
			got += out
			time.Sleep(200 * time.Millisecond)
		}
		if got != "late output\n" {
			t.Fatalf("Read = %q", got)
		}
		if err := m.Interrupt(ctx, "server"); err != nil {
			t.Fatal(err)
		}
		out, err := m.Read(ctx, "server")
		if err != nil || !strings.HasSuffix(out, "[command exited with code 130]") {
			t.Fatalf("Read after interrupt = %q, %v", out, err)
		}
		if r := exec(t, "server", "echo free", 0); r.OutputTail != "free\n" {
			t.Fatalf("exec after interrupt = %+v", r)
		}
		if !strings.Contains(live.get("server"), "ready on :3000") || !strings.Contains(live.get("server"), "late output") ||
			strings.Contains(live.get("server"), "__KEITH_") {
			t.Fatalf("live output = %q", live.get("server"))
		}
	})

	t.Run("timeout then finish", func(t *testing.T) {
		r := exec(t, "work", "sleep 1.5; echo finally", 500*time.Millisecond)
		if !r.TimedOut {
			t.Fatalf("want timeout, got %+v", r)
		}
		time.Sleep(2 * time.Second)
		out, err := m.Read(ctx, "work")
		if err != nil || out != "finally\n[command exited with code 0]" {
			t.Fatalf("Read = %q, %v", out, err)
		}
	})

	t.Run("shell exit is reported and the shell restarts", func(t *testing.T) {
		r := exec(t, "work", "echo bye; exit 7", 0)
		if r.ExitCode != 7 || !strings.Contains(r.OutputTail, "bye") || !strings.Contains(r.OutputTail, "shell exited with status 7") {
			t.Fatalf("exit = %+v", r)
		}
		if r := exec(t, "work", "pwd", 0); r.OutputTail != "/workspace\n" || r.ExitCode != 0 {
			t.Fatalf("after restart = %+v", r)
		}
	})

	t.Run("reattach from a new manager", func(t *testing.T) {
		m2 := tmuxshell.New(sb)
		names, err := m2.Reattach(ctx)
		if err != nil || strings.Join(names, ",") != "extra,server,work" {
			t.Fatalf("Reattach = %v, %v", names, err)
		}
		if r, err := m2.Exec(ctx, "server", "echo again", 0); err != nil || r.OutputTail != "again\n" {
			t.Fatalf("exec via new manager = %+v, %v", r, err)
		}
		if err := m2.Close(ctx, "extra"); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Exec(ctx, "extra", "true", 0); !errors.Is(err, sandbox.ErrNotFound) {
			t.Fatalf("exec on closed shell: %v", err)
		}
	})
}
