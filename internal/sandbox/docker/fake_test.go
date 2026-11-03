package docker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The test binary doubles as a fake docker CLI: with KEITH_FAKE_DOCKER set
// to a directory, it logs each invocation (argv and environment) to
// <dir>/calls.jsonl and answers as described on fakeDocker.
func TestMain(m *testing.M) {
	if dir := os.Getenv("KEITH_FAKE_DOCKER"); dir != "" {
		os.Exit(fakeDocker(dir, os.Args[1:]))
	}
	os.Exit(m.Run())
}

// fakeCall is one logged invocation of the fake docker CLI.
type fakeCall struct {
	Args []string `json:"args"`
	Env  []string `json:"env"`
}

// fakeDocker behaves like docker for the calls the provider makes:
//
//   - exec running readScript: prints zeros forever when the mode file
//     says "zeros" (a subverted head), else nothing;
//   - exec running execWrapper: prints the wrapper's PID line; in mode
//     "hang" it then sleeps until a kill exec (below) kills it;
//   - exec running the kill script: kills the hanging exec, if any;
//   - anything else: exits 0.
func fakeDocker(dir string, args []string) int {
	logCall(dir, fakeCall{Args: args, Env: os.Environ()})
	mode, _ := os.ReadFile(filepath.Join(dir, "mode"))

	isExec := len(args) > 0 && args[0] == "exec"
	switch {
	case isExec && slices.Contains(args, execWrapper):
		fmt.Fprintf(os.Stderr, "%s%d\n", pidPrefix, os.Getpid())
		if strings.TrimSpace(string(mode)) == "hang" {
			_ = os.WriteFile(filepath.Join(dir, "hang.pid"), []byte(strconv.Itoa(os.Getpid())), 0o600)
			time.Sleep(time.Minute)
			return 1
		}
	case isExec && slices.Contains(args, killScript):
		if b, err := os.ReadFile(filepath.Join(dir, "hang.pid")); err == nil {
			if pid, err := strconv.Atoi(string(b)); err == nil {
				_ = syscall.Kill(pid, syscall.SIGKILL)
			}
		}
	case isExec && slices.Contains(args, readScript):
		if strings.TrimSpace(string(mode)) == "zeros" {
			buf := make([]byte, 64<<10)
			for {
				if _, err := os.Stdout.Write(buf); err != nil {
					return 1
				}
			}
		}
	}
	return 0
}

// logCall appends c to <dir>/calls.jsonl. It runs before the fake acts,
// since an endless writer is killed by SIGPIPE.
func logCall(dir string, c fakeCall) {
	f, err := os.OpenFile(filepath.Join(dir, "calls.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	_ = json.NewEncoder(f).Encode(c)
}

// fakeCLI points a Provider at the fake docker CLI and returns a function
// that reads the calls logged so far.
func fakeCLI(t *testing.T, mode string) (*Provider, func() []fakeCall) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "mode"), []byte(mode), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("KEITH_FAKE_DOCKER", dir)
	p := New(Options{Docker: os.Args[0]})
	return p, func() []fakeCall {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, "calls.jsonl"))
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		var calls []fakeCall
		for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			if line == "" {
				continue
			}
			var c fakeCall
			if err := json.Unmarshal([]byte(line), &c); err != nil {
				t.Fatalf("calls.jsonl: %v", err)
			}
			calls = append(calls, c)
		}
		return calls
	}
}
