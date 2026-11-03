package docker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

func TestWriteEnvFile(t *testing.T) {
	f, err := writeEnvFile(map[string]string{"B": "secret value", "A": " x=y "})
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f)
	b, _ := os.ReadFile(f)
	st, _ := os.Stat(f)
	if string(b) != "A= x=y \nB=secret value\n" || st.Mode().Perm() != 0o600 {
		t.Fatalf("env file = %q, mode %v", b, st.Mode())
	}
	for _, env := range []map[string]string{{"BAD=KEY": "v"}, {"A": "x\ny"}, {"A": "x\r"}, {"A": "\xff"}} {
		if f, err := writeEnvFile(env); err == nil {
			os.Remove(f)
			t.Errorf("%q accepted", env)
		}
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

// argsWith returns the args of the first logged call whose argv contains
// marker, or nil.
func argsWith(calls []fakeCall, marker string) []string {
	for _, c := range calls {
		if slices.Contains(c.Args, marker) {
			return c.Args
		}
	}
	return nil
}

// Regression: root execs ran `kill` and the root setup script's tools by
// the container's PATH, whose first entry (~/.local/bin) the agent can
// write, so the agent could run code as root. Every exec now runs
// /bin/sh, root execs get PATH=sandbox.SystemPath after any requested env,
// and the timeout kill uses the shell's builtin kill.
func TestRootExecsUseSystemPath(t *testing.T) {
	ctx := context.Background()
	p, calls := fakeCLI(t, "")
	sb := &Sandbox{p: p, id: "keith-sb-fake"}

	for _, user := range []string{"root", "0", "0:0", "root:root"} {
		if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "id -u", User: user,
			Env: map[string]string{"PATH": "/home/agent/.local/bin:/usr/bin"}}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "whoami"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := sb.ReadFile(ctx, "a.txt"); err != nil {
		t.Fatal(err)
	}
	var execs int
	for _, c := range calls() {
		if c.Args[0] != "exec" {
			continue
		}
		execs++
		i := slices.Index(c.Args, "keith-sb-fake")
		if i < 0 || c.Args[i+1] != "/bin/sh" {
			t.Errorf("exec does not run /bin/sh: %q", c.Args)
			continue
		}
		var paths []string
		for j := 0; j+1 < i; j++ {
			if c.Args[j] == "-e" && strings.HasPrefix(c.Args[j+1], "PATH") {
				paths = append(paths, c.Args[j+1])
			}
		}
		whoami := slices.Contains(c.Args, "whoami")
		switch {
		case whoami && len(paths) != 0:
			t.Errorf("default-user exec sets PATH: %q", c.Args)
		case !whoami && (len(paths) == 0 || paths[len(paths)-1] != "PATH="+sandbox.SystemPath):
			t.Errorf("exec does not end with PATH=%s: %q", sandbox.SystemPath, c.Args)
		}
	}
	if execs != 6 {
		t.Fatalf("%d execs logged", execs)
	}

	// A timeout kills the process group as root, with the builtin kill.
	p, calls = fakeCLI(t, "hang")
	sb = &Sandbox{p: p, id: "keith-sb-fake"}
	res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "sleep 30", Timeout: 200 * time.Millisecond}, nil)
	if err != nil || !res.TimedOut {
		t.Fatalf("Exec = %+v, %v; want a timeout", res, err)
	}
	got := argsWith(calls(), killScript)
	want := []string{"exec", "-u", "0", "-e", "PATH=" + sandbox.SystemPath, "keith-sb-fake", "/bin/sh", "-c", killScript, "sh"}
	if len(got) != len(want)+1 || !slices.Equal(got[:len(want)], want) {
		t.Fatalf("kill args = %q; want %q <pid>", got, want)
	}
}

// createRequest returns the logged POST /containers/create body.
func createRequest(t *testing.T, calls []fakeCall) createConfig {
	t.Helper()
	for _, c := range calls {
		if strings.HasPrefix(c.API, "POST /containers/create?name=keith-sb-") {
			var cfg createConfig
			if err := json.Unmarshal([]byte(c.APIBody), &cfg); err != nil {
				t.Fatal(err)
			}
			return cfg
		}
	}
	t.Fatalf("no create request in %+v", calls)
	return createConfig{}
}

// Regression: sandboxes kept Docker's default capabilities, so anything
// that reached root had DAC_OVERRIDE, NET_RAW, MKNOD and more. They drop
// all of them but CHOWN and KILL.
func TestCreateDropsCapabilities(t *testing.T) {
	p, calls := fakeCLI(t, "")
	if _, err := p.Create(context.Background(), sandbox.Spec{}); err != nil {
		t.Fatal(err)
	}
	hc := createRequest(t, calls()).HostConfig
	if !slices.Equal(hc.CapDrop, []string{"ALL"}) || !slices.Equal(hc.CapAdd, []string{"CHOWN", "KILL"}) ||
		!slices.Equal(hc.SecurityOpt, []string{"no-new-privileges:true"}) || !hc.Init {
		t.Fatalf("host config = %+v", hc)
	}
}

// Regression: Create passed Spec.Env (the org's secrets) as `-e NAME` with
// NAME=value in the docker CLI's own environment, so a secret named
// DOCKER_HOST, DOCKER_CONTEXT or HTTPS_PROXY redirected the CLI itself.
// Secrets now go to the Engine API over dial-stdio's stdin, and Exec's
// env through a private --env-file: never the CLI's argv or environment.
func TestSecretsStayOutOfTheCLI(t *testing.T) {
	ctx := context.Background()
	p, calls := fakeCLI(t, "")
	p.opts.ExtraHosts = []string{"host.docker.internal:host-gateway"}
	secrets := map[string]string{
		"DOCKER_HOST":    "tcp://evil.invalid:2375",
		"DOCKER_CONTEXT": "evil",
		"HTTPS_PROXY":    "http://evil.invalid:3128",
		"PEM":            "-----BEGIN KEY-----\nc2VjcmV0\n-----END KEY-----\n",
	}
	sbx, err := p.Create(ctx, sandbox.Spec{Env: secrets, Labels: map[string]string{"keith.session": "s1"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sbx.Exec(ctx, sandbox.ExecRequest{Cmd: "true", Env: map[string]string{"DOCKER_HOST": "tcp://evil.invalid:1"}}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := sbx.Exec(ctx, sandbox.ExecRequest{Cmd: "true", Env: map[string]string{"A": "multi\nline"}}, nil); err == nil {
		t.Fatal("exec env with a newline: want an error")
	}

	got := calls()
	for _, c := range got {
		for _, kv := range c.Env {
			k, v, _ := strings.Cut(kv, "=")
			if _, ok := secrets[k]; ok && strings.Contains(v, "evil") {
				t.Errorf("docker %s ran with %s in its environment", c.Args[0], kv)
			}
		}
		for _, a := range c.Args {
			if strings.Contains(a, "evil") || strings.Contains(a, "c2VjcmV0") {
				t.Errorf("docker %s argv holds a secret: %q", c.Args[0], c.Args)
			}
		}
	}
	cfg := createRequest(t, got)
	want := []string{"DOCKER_CONTEXT=evil", "DOCKER_HOST=tcp://evil.invalid:2375", "HTTPS_PROXY=http://evil.invalid:3128",
		"PEM=-----BEGIN KEY-----\nc2VjcmV0\n-----END KEY-----\n"}
	if !slices.Equal(cfg.Env, want) {
		t.Errorf("container env = %q", cfg.Env)
	}
	if cfg.Labels[LabelSandbox] != "1" || cfg.Labels["keith.session"] != "s1" || cfg.User != DefaultUser ||
		cfg.WorkingDir != sandbox.WorkspaceDir || cfg.HostConfig.PortBindings["3000/tcp"][0].HostIp != "127.0.0.1" ||
		!slices.Equal(cfg.HostConfig.ExtraHosts, p.opts.ExtraHosts) || cfg.HostConfig.Memory != DefaultMemoryMB<<20 {
		t.Errorf("create config = %+v", cfg)
	}
	if args := argsWith(got, "start"); len(args) != 2 || args[1] != sbx.ID() {
		t.Errorf("start args = %q", args)
	}
	var files map[string]string
	for _, c := range got {
		if c.EnvFiles != nil {
			files = c.EnvFiles
		}
	}
	if len(files) != 1 {
		t.Fatalf("env files = %v", files)
	}
	for f, content := range files {
		if content != "DOCKER_HOST=tcp://evil.invalid:1\n" {
			t.Errorf("env file = %q", content)
		}
		if _, err := os.Stat(f); !os.IsNotExist(err) {
			t.Errorf("env file %s left behind: %v", f, err)
		}
	}
}
