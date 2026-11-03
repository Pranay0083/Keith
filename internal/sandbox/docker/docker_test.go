package docker

import (
	"bytes"
	"errors"
	"strings"
	"testing"

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
