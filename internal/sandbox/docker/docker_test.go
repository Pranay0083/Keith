package docker

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

func TestEnvArgsKeepValuesOutOfArgv(t *testing.T) {
	args, vars, err := envArgs(map[string]string{"B": "secret value", "A": "x\ny"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args, " "); got != "-e A -e B" {
		t.Fatalf("args = %q", got)
	}
	if got := strings.Join(vars, "|"); got != "A=x\ny|B=secret value" {
		t.Fatalf("vars = %q", got)
	}
	if _, _, err := envArgs(map[string]string{"BAD=KEY": "v"}); err == nil {
		t.Fatal("invalid key accepted")
	}
}

func TestPIDWriter(t *testing.T) {
	for _, chunks := range [][]string{
		{"__KEITH_PID=42\nerr output\n"},
		{"__KEITH_", "PID=4", "2\ner", "r output\n"},
	} {
		var out bytes.Buffer
		pid := make(chan int, 1)
		w := &pidWriter{w: &out, pid: pid}
		for _, c := range chunks {
			w.Write([]byte(c))
		}
		if got := <-pid; got != 42 || out.String() != "err output\n" {
			t.Fatalf("chunks %q: pid %d, out %q", chunks, got, out.String())
		}
	}
	// Not the wrapper's line (docker failed before running it).
	var out bytes.Buffer
	pid := make(chan int, 1)
	w := &pidWriter{w: &out, pid: pid}
	w.Write([]byte("Error response from daemon: Container x is paused\n"))
	if len(pid) != 0 || out.String() != "Error response from daemon: Container x is paused\n" {
		t.Fatalf("out %q", out.String())
	}
}

func TestWrapNotFound(t *testing.T) {
	for _, msg := range []string{
		"Error response from daemon: No such container: keith-sb-x",
		"Error: No such object: keith-sb-x",
		"Unable to find image 'sha256:abc' locally",
	} {
		if err := wrap(&cliError{args: []string{"x"}, code: 1, stderr: msg}); !errors.Is(err, sandbox.ErrNotFound) {
			t.Errorf("%q not mapped to ErrNotFound", msg)
		}
	}
	if err := wrap(&cliError{args: []string{"x"}, code: 1, stderr: "permission denied"}); errors.Is(err, sandbox.ErrNotFound) {
		t.Error("unrelated error mapped to ErrNotFound")
	}
}

func TestStreamWriterTail(t *testing.T) {
	tail := sbutil.NewTailBuffer(4)
	ch := make(chan []byte, 2)
	w := &streamWriter{tail: tail, chunks: ch}
	w.Write([]byte("abc"))
	w.Write([]byte("def"))
	if tail.String() != "cdef" || string(<-ch) != "abc" || string(<-ch) != "def" {
		t.Fatal("streamWriter")
	}
}

func TestLimitBuffer(t *testing.T) {
	b := &limitBuffer{max: 5}
	if n, err := b.Write([]byte("abc")); n != 3 || err != nil {
		t.Fatalf("write = %d, %v", n, err)
	}
	if n, err := b.Write([]byte("defg")); n != 2 || !errors.Is(err, errOutputLimit) || !b.over || string(b.buf) != "abcde" {
		t.Fatalf("write past max = %d, %v, over=%v, buf=%q", n, err, b.over, b.buf)
	}
}

// Regression: ReadFile buffered whatever the sandbox printed, so a read
// of a symlink to /dev/zero (or a planted head) could exhaust the
// worker's memory. The docker CLI's output is now cut off at limit+1.
func TestReadFileLimitCapsCLIOutput(t *testing.T) {
	p, calls := fakeCLI(t, "zeros")
	sb := &Sandbox{p: p, id: "keith-sb-fake"}
	done := make(chan error, 1)
	go func() {
		_, err := sb.ReadFileLimit(context.Background(), "zero", 1<<20)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, sandbox.ErrFileTooLarge) {
			t.Fatalf("ReadFileLimit = %v; want ErrFileTooLarge", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("ReadFileLimit of an endless stream did not return")
	}
	if _, err := sb.ReadFile(context.Background(), "zero"); !errors.Is(err, sandbox.ErrFileTooLarge) {
		t.Fatalf("ReadFile = %v; want ErrFileTooLarge", err)
	}
	var limits []string
	for _, c := range calls() {
		if len(c.Args) > 0 && c.Args[0] == "exec" {
			limits = append(limits, c.Args[len(c.Args)-1])
		}
	}
	if want := []string{"1048576", strconv.Itoa(sandbox.MaxReadFile)}; !slices.Equal(limits, want) {
		t.Fatalf("limits passed to the read script = %v; want %v", limits, want)
	}
}
