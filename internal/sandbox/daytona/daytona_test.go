package daytona

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	osexec "os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// TestBuildCommandRunsInSh runs the wrapped command with the local sh to
// check the quoting, working directory, env and stdin end to end.
func TestBuildCommandRunsInSh(t *testing.T) {
	if _, err := osexec.LookPath("sh"); err != nil {
		t.Skip("no sh")
	}
	dir := t.TempDir()
	s := &daytonaSandbox{user: "daytona"}
	cmd, err := s.buildCommand(sandbox.ExecRequest{
		Cmd: `printf '%s|%s|%s|' "$FOO" "$BAR" "it's"; pwd; read x || echo no-stdin`,
		Dir: dir,
		Env: map[string]string{"FOO": "a 'quoted' $value", "BAR": "line1\nline2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := osexec.Command("sh", "-c", cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("sh: %v: %s", err, out)
	}
	if realDir, err := filepath.EvalSymlinks(dir); err == nil {
		out = bytes.ReplaceAll(out, []byte(realDir), []byte(dir))
	}
	want := "a 'quoted' $value|line1\nline2|it's|" + dir + "\nno-stdin\n"
	if string(out) != want {
		t.Errorf("output = %q; want %q", out, want)
	}
}

// A root command runs with sandbox.SystemPath even if the sandbox's own
// PATH or the request's Env puts a user-writable directory first.
func TestBuildCommandRootPath(t *testing.T) {
	if _, err := osexec.LookPath("sh"); err != nil {
		t.Skip("no sh")
	}
	s := &daytonaSandbox{user: "root"} // root by default, so no sudo here
	cmd, err := s.buildCommand(sandbox.ExecRequest{
		Cmd: `echo "$PATH"`, Dir: t.TempDir(), User: "root",
		Env: map[string]string{"PATH": "/home/agent/.local/bin:/usr/bin"},
	})
	if err != nil {
		t.Fatal(err)
	}
	c := osexec.Command("sh", "-c", cmd)
	c.Env = []string{"PATH=/home/agent/.local/bin:/usr/bin:/bin"}
	out, err := c.CombinedOutput()
	if err != nil || string(out) != sandbox.SystemPath+"\n" {
		t.Fatalf("PATH in a root command = %q, %v", out, err)
	}
}

func ctxT(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// shortTimings speeds up Exec's polling for one test. Tests that call it
// must not run in parallel.
func shortTimings(t *testing.T) {
	t.Helper()
	oldSlow, oldGrace := pollSlow, streamGrace
	pollSlow, streamGrace = 50*time.Millisecond, 200*time.Millisecond
	t.Cleanup(func() { pollSlow, streamGrace = oldSlow, oldGrace })
}

func TestNewRequiresAPIKey(t *testing.T) {
	if _, err := New(Options{}); err == nil {
		t.Fatal("New without APIKey: want error")
	}
	p, err := New(Options{APIKey: "k", APIURL: "http://127.0.0.1:1/api"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Name() != Name || Name != "daytona" {
		t.Fatalf("Name() = %q", p.Name())
	}
}

func TestNewIntervals(t *testing.T) {
	cases := []struct {
		name          string
		opts          Options
		stop, archive int
		deleteSet     bool
		deleteMinutes int
	}{
		{"defaults", Options{}, 15, 7 * 24 * 60, false, 0},
		{"rounded up", Options{AutoStopInterval: 90 * time.Second, AutoArchiveInterval: time.Hour, AutoDeleteInterval: 61 * time.Second}, 2, 60, true, 2},
		{"disabled", Options{AutoStopInterval: -1, AutoArchiveInterval: -1, AutoDeleteInterval: -1}, 0, 0, true, -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			c.opts.APIKey = "k"
			c.opts.APIURL = "http://127.0.0.1:1/api"
			p, err := New(c.opts)
			if err != nil {
				t.Fatal(err)
			}
			if p.autoStop != c.stop || p.autoArchive != c.archive {
				t.Errorf("autoStop, autoArchive = %d, %d; want %d, %d", p.autoStop, p.autoArchive, c.stop, c.archive)
			}
			if (p.autoDelete != nil) != c.deleteSet || (p.autoDelete != nil && *p.autoDelete != c.deleteMinutes) {
				t.Errorf("autoDelete = %v; want set=%v %d", p.autoDelete, c.deleteSet, c.deleteMinutes)
			}
		})
	}
}

func TestParseTemplate(t *testing.T) {
	cases := []struct {
		in      string
		isImage bool
		ref     string
	}{
		{"", false, ""},
		{"keith-base", false, "keith-base"},
		{"node:22", true, "node:22"},
		{"ghcr.io/org/img", true, "ghcr.io/org/img"},
		{"alpine@sha256:abc", true, "alpine@sha256:abc"},
		{"snapshot:daytonaio/sandbox:0.4.3", false, "daytonaio/sandbox:0.4.3"},
		{"image:ubuntu", true, "ubuntu"},
	}
	for _, c := range cases {
		isImage, ref := parseTemplate(c.in)
		if isImage != c.isImage || ref != c.ref {
			t.Errorf("parseTemplate(%q) = %v, %q; want %v, %q", c.in, isImage, ref, c.isImage, c.ref)
		}
	}
}

func TestSplitEgress(t *testing.T) {
	cidrs, domains := splitEgress([]string{
		"github.com", " registry.npmjs.org ", "https://proxy.golang.org/x", "api.github.com:443",
		"10.0.0.0/8", "1.2.3.4", "*.pypi.org", "GitHub.com.", "", "[2001:db8::1]:22",
	})
	wantCIDRs := []string{"10.0.0.0/8", "1.2.3.4/32", "2001:db8::1/128"}
	wantDomains := []string{"github.com", "registry.npmjs.org", "proxy.golang.org", "api.github.com", "*.pypi.org"}
	if !slices.Equal(cidrs, wantCIDRs) {
		t.Errorf("cidrs = %q; want %q", cidrs, wantCIDRs)
	}
	if !slices.Equal(domains, wantDomains) {
		t.Errorf("domains = %q; want %q", domains, wantDomains)
	}
}

func TestUnitConversions(t *testing.T) {
	for mb, gb := range map[int]int{0: 0, 1: 1, 1024: 1, 1500: 2, 4096: 4, 4097: 5} {
		if got := memoryGB(mb); got != gb {
			t.Errorf("memoryGB(%d) = %d; want %d", mb, got, gb)
		}
	}
	if got := snapshotName("My Repo / Template!!"); !strings.HasPrefix(got, "keith-my-repo-template-") {
		t.Errorf("snapshotName = %q", got)
	}
	if got := snapshotName("???"); !strings.HasPrefix(got, "keith-snapshot-") {
		t.Errorf("snapshotName(???) = %q", got)
	}
}

func TestCreateFromImage(t *testing.T) {
	f := newFake(t)
	p := f.provider(Options{Labels: map[string]string{"env": "test", "keith.session": "overridden"}})

	sb, err := p.Create(ctxT(t), sandbox.Spec{
		Template:        "node:22",
		CPUs:            2,
		MemoryMB:        1500,
		Env:             map[string]string{"SECRET_TOKEN": "s3cr3t"},
		Labels:          map[string]string{"keith.session": "sess-1", "keith.org": "org-1"},
		EgressAllowlist: []string{"github.com", "10.0.0.0/8"},
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if sb.ID() != "sb-1" {
		t.Errorf("ID = %q", sb.ID())
	}
	if st := f.state("sb-1"); st != "started" {
		t.Errorf("state = %q; want started", st)
	}

	var body map[string]any
	var executes []string
	f.read(func() { body, executes = f.createBody[0], slices.Clone(f.executes) })
	labels, _ := body["labels"].(map[string]any)
	for k, v := range map[string]string{"keith.sandbox": "1", "env": "test", "keith.session": "sess-1", "keith.org": "org-1"} {
		if labels[k] != v {
			t.Errorf("label %s = %v; want %s (labels %v)", k, labels[k], v, labels)
		}
	}
	if env, _ := body["env"].(map[string]any); env["SECRET_TOKEN"] != "s3cr3t" {
		t.Errorf("env = %v", body["env"])
	}
	if body["cpu"] != float64(2) || body["memory"] != float64(2) {
		t.Errorf("cpu, memory = %v, %v; want 2, 2", body["cpu"], body["memory"])
	}
	if body["autoStopInterval"] != float64(15) || body["autoArchiveInterval"] != float64(7*24*60) {
		t.Errorf("autoStop, autoArchive = %v, %v", body["autoStopInterval"], body["autoArchiveInterval"])
	}
	if _, ok := body["autoDeleteInterval"]; ok {
		t.Errorf("autoDeleteInterval sent: %v", body["autoDeleteInterval"])
	}
	if bi, _ := body["buildInfo"].(map[string]any); bi["dockerfileContent"] != "FROM node:22" {
		t.Errorf("buildInfo = %v", body["buildInfo"])
	}
	if _, ok := body["snapshot"]; ok {
		t.Errorf("snapshot sent for image create: %v", body["snapshot"])
	}
	if body["networkAllowList"] != "10.0.0.0/8" || body["domainAllowList"] != "github.com" {
		t.Errorf("allow lists = %v, %v", body["networkAllowList"], body["domainAllowList"])
	}

	if len(executes) != 1 || !strings.Contains(executes[0], "'/workspace'") {
		t.Errorf("workspace preparation = %q", executes)
	}
}

func TestCreateFromSnapshot(t *testing.T) {
	f := newFake(t)
	p := f.provider(Options{DefaultSnapshot: "keith-default", AutoDeleteInterval: -1})

	if _, err := p.Create(ctxT(t), sandbox.Spec{SnapshotID: "keith-repo-1", CPUs: 4, MemoryMB: 8192}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := p.Create(ctxT(t), sandbox.Spec{}); err != nil {
		t.Fatalf("Create default: %v", err)
	}
	if _, err := p.Create(ctxT(t), sandbox.Spec{Template: "keith-base"}); err != nil {
		t.Fatalf("Create template snapshot: %v", err)
	}

	var bodies []map[string]any
	f.read(func() { bodies = slices.Clone(f.createBody) })
	for i, want := range []string{"keith-repo-1", "keith-default", "keith-base"} {
		body := bodies[i]
		if body["snapshot"] != want {
			t.Errorf("create %d: snapshot = %v; want %s", i, body["snapshot"], want)
		}
		if _, ok := body["cpu"]; ok {
			t.Errorf("create %d: cpu sent for a snapshot", i)
		}
		if body["autoDeleteInterval"] != float64(-1) {
			t.Errorf("create %d: autoDeleteInterval = %v; want -1", i, body["autoDeleteInterval"])
		}
	}
}

func TestGetNotFound(t *testing.T) {
	f := newFake(t)
	p := f.provider(Options{})
	_, err := p.Get(ctxT(t), "missing")
	if !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("Get(missing) = %v; want ErrNotFound", err)
	}

	f.addSandbox("sb-x", "stopped")
	sb, err := p.Get(ctxT(t), "sb-x")
	if err != nil || sb.ID() != "sb-x" {
		t.Fatalf("Get(sb-x) = %v, %v", sb, err)
	}
}

func TestExecStreams(t *testing.T) {
	shortTimings(t)
	f := newFake(t)
	f.runner = func(cmd string) fakeRun { return fakeRun{stdout: "hello\nworld\n", stderr: "oops\n", exit: 3} }
	f.addSandbox("sb-1", "started")
	p := f.provider(Options{})
	sb, err := p.Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var streamed bytes.Buffer
	res, err := sb.Exec(ctxT(t), sandbox.ExecRequest{
		Cmd: "echo 'hi'", Dir: "sub", Env: map[string]string{"FOO": "bar baz"},
	}, func(chunk []byte) {
		mu.Lock()
		defer mu.Unlock()
		streamed.Write(chunk)
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 3 || res.TimedOut {
		t.Errorf("ExitCode, TimedOut = %d, %v; want 3, false", res.ExitCode, res.TimedOut)
	}
	if res.Stdout != "hello\nworld\n" || res.Stderr != "oops\n" {
		t.Errorf("Stdout, Stderr = %q, %q", res.Stdout, res.Stderr)
	}
	mu.Lock()
	got := streamed.String()
	mu.Unlock()
	// stdout and stderr chunks interleave in arrival order, so compare the
	// bytes regardless of order.
	sorted := func(s string) string { b := []byte(s); slices.Sort(b); return string(b) }
	if sorted(got) != sorted("hello\nworld\noops\n") {
		t.Errorf("streamed = %q", got)
	}

	cmd := f.lastCommand()
	inner := "cd " + sbutil.Quote("/workspace/sub") + " && FOO=" + sbutil.Quote("bar baz") + " /bin/sh -c " + sbutil.Quote("echo 'hi'")
	if want := "/bin/sh -c " + sbutil.Quote(inner) + " < /dev/null"; cmd.command != want {
		t.Errorf("command =\n %s\nwant\n %s", cmd.command, want)
	}
	if !f.sessionDeleted(cmd.session) {
		t.Errorf("session %s not deleted", cmd.session)
	}
}

func TestExecFallsBackWithoutStream(t *testing.T) {
	shortTimings(t)
	f := newFake(t)
	f.wsMode = "fail"
	f.runner = func(string) fakeRun { return fakeRun{stdout: "out", stderr: "err", exit: 0} }
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	var chunks []string
	res, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "true"}, func(c []byte) { chunks = append(chunks, string(c)) })
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 0 || res.Stdout != "out" || res.Stderr != "err" {
		t.Errorf("result = %+v", res)
	}
	if strings.Join(chunks, "") != "outerr" {
		t.Errorf("chunks = %q", chunks)
	}
}

func TestExecStreamStaysOpen(t *testing.T) {
	// The stream does not close when the command exits: Exec must notice
	// the exit code by polling and fill the output from the logs.
	shortTimings(t)
	f := newFake(t)
	f.wsMode = "hang"
	f.runner = func(string) fakeRun { return fakeRun{stdout: "abcdef", exit: 7} }
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var streamed bytes.Buffer
	res, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "x"}, func(c []byte) {
		mu.Lock()
		defer mu.Unlock()
		streamed.Write(c)
	})
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 7 || res.Stdout != "abcdef" {
		t.Errorf("result = %+v", res)
	}
	mu.Lock()
	defer mu.Unlock()
	if streamed.String() != "abcdef" {
		t.Errorf("streamed = %q (output must not be duplicated)", streamed.String())
	}
}

func TestExecTimeout(t *testing.T) {
	shortTimings(t)
	f := newFake(t)
	f.runner = func(string) fakeRun { return fakeRun{stdout: "partial\n", hang: true} }
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	res, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "sleep 100", Timeout: 300 * time.Millisecond}, nil)
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if !res.TimedOut || res.ExitCode != -1 {
		t.Errorf("TimedOut, ExitCode = %v, %d; want true, -1", res.TimedOut, res.ExitCode)
	}
	if res.Stdout != "partial\n" {
		t.Errorf("Stdout = %q; want the output before the timeout", res.Stdout)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("Exec took %v", d)
	}
	if cmd := f.lastCommand(); !f.sessionDeleted(cmd.session) {
		t.Errorf("timed-out session not deleted")
	}
}

func TestExecCancelled(t *testing.T) {
	shortTimings(t)
	f := newFake(t)
	f.runner = func(string) fakeRun { return fakeRun{hang: true} }
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "sleep 100"}, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Exec with expired ctx = %v; want DeadlineExceeded", err)
	}
}

func TestExecUserAndEnv(t *testing.T) {
	shortTimings(t)
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "id", User: "root"}, nil); err != nil {
		t.Fatal(err)
	}
	// Regression: a root command must not look up sudo, sh or anything it
	// runs in a PATH the sandbox user can write to.
	if c := f.lastCommand().command; !strings.HasPrefix(c, "/usr/bin/sudo -n -E -H -u 'root' -- /bin/sh -c ") ||
		!strings.Contains(c, "PATH="+sandbox.SystemPath+" /bin/sh -c ") {
		t.Errorf("root command = %s", c)
	}
	if _, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "id", User: "daytona"}, nil); err != nil {
		t.Fatal(err)
	}
	if c := f.lastCommand().command; strings.Contains(c, "sudo") || strings.Contains(c, "PATH=") {
		t.Errorf("default-user command uses sudo or sets PATH: %s", c)
	}
	if _, err := sb.Exec(ctxT(t), sandbox.ExecRequest{Cmd: "id", Env: map[string]string{"BAD-NAME": "x"}}, nil); err == nil {
		t.Error("invalid env name: want error")
	}
}

func TestFiles(t *testing.T) {
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	ctx := ctxT(t)

	big := bytes.Repeat([]byte("0123456789abcdef"), 3<<16) // 3 MiB
	if err := sb.WriteFile(ctx, "src/big.bin", big, 0o640); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	f.read(func() {
		if got := f.files["/workspace/src/big.bin"]; !bytes.Equal(got.data, big) || got.mode != "0640" {
			t.Errorf("stored file: %d bytes, mode %q", len(got.data), got.mode)
		}
		if !slices.Contains(f.folders, "/workspace/src") {
			t.Errorf("parent folder not created: %v", f.folders)
		}
	})
	data, err := sb.ReadFile(ctx, "/workspace/src/big.bin")
	if err != nil || !bytes.Equal(data, big) {
		t.Fatalf("ReadFile: %d bytes, %v", len(data), err)
	}

	if _, err := sb.ReadFile(ctx, "nope.txt"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("ReadFile(missing) = %v; want fs.ErrNotExist", err)
	}
	if err := sb.RemoveFile(ctx, "src/big.bin"); err != nil {
		t.Fatalf("RemoveFile: %v", err)
	}
	if _, err := sb.ReadFile(ctx, "src/big.bin"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("ReadFile(removed) = %v; want fs.ErrNotExist", err)
	}
	if err := sb.RemoveFile(ctx, "src/big.bin"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("RemoveFile(missing) = %v; want fs.ErrNotExist", err)
	}
	if err := sb.WriteFile(ctx, "/top.txt", []byte("x"), 0); err != nil {
		t.Fatalf("WriteFile(/top.txt): %v", err)
	}
	f.read(func() {
		if got := f.files["/top.txt"].mode; got != "0644" {
			t.Errorf("default mode = %q; want 0644", got)
		}
	})
}

func TestPauseResume(t *testing.T) {
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	ctx := ctxT(t)
	for range 2 {
		if err := sb.Pause(ctx); err != nil {
			t.Fatalf("Pause: %v", err)
		}
	}
	if f.state("sb-1") != "stopped" || f.countRequests("POST /api/sandbox/sb-1/stop") != 1 {
		t.Errorf("state %q, stop calls %d", f.state("sb-1"), f.countRequests("POST /api/sandbox/sb-1/stop"))
	}
	for range 2 {
		if err := sb.Resume(ctx); err != nil {
			t.Fatalf("Resume: %v", err)
		}
	}
	if f.state("sb-1") != "started" || f.countRequests("POST /api/sandbox/sb-1/start") != 1 {
		t.Errorf("state %q, start calls %d", f.state("sb-1"), f.countRequests("POST /api/sandbox/sb-1/start"))
	}
}

func TestDeleteIdempotent(t *testing.T) {
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	p := f.provider(Options{})
	sb, err := p.Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		if err := sb.Delete(ctxT(t)); err != nil {
			t.Fatalf("Delete #%d: %v", i+1, err)
		}
	}
	if _, err := p.Get(ctxT(t), "sb-1"); !errors.Is(err, sandbox.ErrNotFound) {
		t.Errorf("Get after Delete = %v; want ErrNotFound", err)
	}
	if err := sb.Pause(ctxT(t)); !errors.Is(err, sandbox.ErrNotFound) {
		t.Errorf("Pause after Delete = %v; want ErrNotFound", err)
	}
}

func TestPortURL(t *testing.T) {
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	u, err := sb.PortURL(ctxT(t), 3000)
	if err != nil || u != "https://3000-tok.preview.test" {
		t.Fatalf("PortURL = %q, %v", u, err)
	}
	f.read(func() {
		if len(f.previewQS) != 1 || f.previewQS[0] != "expiresInSeconds=3600" {
			t.Errorf("query = %q", f.previewQS)
		}
	})
	if _, err := sb.PortURL(ctxT(t), 0); err == nil {
		t.Error("PortURL(0): want error")
	}
}

func TestSnapshot(t *testing.T) {
	for _, class := range []string{"container", "linux-vm"} {
		t.Run(class, func(t *testing.T) {
			f := newFake(t)
			f.sandboxClass = class
			f.addSandbox("sb-1", "started")
			sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
			if err != nil {
				t.Fatal(err)
			}
			name, err := sb.Snapshot(ctxT(t), "repo template")
			if err != nil {
				t.Fatalf("Snapshot: %v", err)
			}
			f.read(func() {
				if !strings.HasPrefix(name, "keith-repo-template-") || f.snapshots[name] == nil {
					t.Errorf("name = %q; snapshots %v", name, f.snapshots)
				}
			})
			if f.state("sb-1") != "started" {
				t.Errorf("state after snapshot = %q; want started", f.state("sb-1"))
			}
			stops := f.countRequests("POST /api/sandbox/sb-1/stop")
			if want := map[string]int{"container": 0, "linux-vm": 1}[class]; stops != want {
				t.Errorf("stop calls = %d; want %d", stops, want)
			}
		})
	}
}

// Regression: ReadFile io.ReadAll'd the download, so a symlink to
// /dev/zero (or any huge file) could exhaust the worker's memory. A check
// command now refuses special and oversized files before the download,
// and the download stops after limit+1 bytes.
func TestReadFileLimits(t *testing.T) {
	f := newFake(t)
	f.addSandbox("sb-1", "started")
	sb, err := f.provider(Options{}).Get(ctxT(t), "sb-1")
	if err != nil {
		t.Fatal(err)
	}
	ctx := ctxT(t)
	f.read(func() {
		f.files["/workspace/ten"] = fakeFile{data: []byte("0123456789")}
		f.files["/workspace/zero"] = fakeFile{kind: "special"}
		f.files["/workspace/swapped"] = fakeFile{kind: "endless"}
	})
	downloads := func() int { return f.countRequests("POST /toolbox/sb-1/files/bulk-download") }

	if got, err := sb.ReadFileLimit(ctx, "ten", 10); err != nil || string(got) != "0123456789" {
		t.Fatalf("ReadFileLimit(ten, 10) = %q, %v", got, err)
	}
	before := downloads()
	if _, err := sb.ReadFileLimit(ctx, "ten", 9); !errors.Is(err, sandbox.ErrFileTooLarge) {
		t.Fatalf("ReadFileLimit(ten, 9) = %v; want ErrFileTooLarge", err)
	}
	if _, err := sb.ReadFile(ctx, "zero"); !errors.Is(err, sandbox.ErrNotRegular) {
		t.Fatalf("ReadFile(zero) = %v; want ErrNotRegular", err)
	}
	if _, err := sb.ReadFile(ctx, "missing"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("ReadFile(missing) = %v; want fs.ErrNotExist", err)
	}
	if n := downloads() - before; n != 0 {
		t.Fatalf("%d downloads of refused files", n)
	}

	if _, err := sb.ReadFileLimit(ctx, "swapped", 1<<20); !errors.Is(err, sandbox.ErrFileTooLarge) {
		t.Fatalf("ReadFileLimit(swapped) = %v; want ErrFileTooLarge", err)
	}
	// The server stops once the client hangs up; it can only have written
	// what fit in the client's limit and the connection's buffers.
	deadline := time.Now().Add(10 * time.Second)
	for {
		var n1, n2 int64
		f.read(func() { n1 = f.downloaded })
		time.Sleep(100 * time.Millisecond)
		f.read(func() { n2 = f.downloaded })
		if n1 == n2 {
			if n2 > 64<<20 {
				t.Fatalf("the endless download wrote %d bytes", n2)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the endless download is still being read (%d bytes)", n2)
		}
	}
}
