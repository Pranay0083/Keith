package docker_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbtest"
)

func TestDockerRoundTrip(t *testing.T) {
	p := sbtest.DockerProvider(t)
	ctx := context.Background()

	sb := sbtest.Sandbox(t, p, sandbox.Spec{
		Env:             map[string]string{"KEITH_SECRET": "s3cr3t value", "MULTI": "a\nb"},
		Labels:          map[string]string{"keith.session": "it"},
		EgressAllowlist: []string{"github.com"},
	})

	t.Run("get", func(t *testing.T) {
		got, err := p.Get(ctx, sb.ID())
		if err != nil || got.ID() != sb.ID() {
			t.Fatalf("Get = %v, %v", got, err)
		}
		if _, err := p.Get(ctx, "keith-sb-doesnotexist"); !errors.Is(err, sandbox.ErrNotFound) {
			t.Fatalf("Get unknown: %v", err)
		}
		ids, err := p.List(ctx, map[string]string{"keith.session": "it"})
		if err != nil || !strings.Contains(strings.Join(ids, " "), sb.ID()) {
			t.Fatalf("List = %v, %v", ids, err)
		}
	})

	t.Run("exec", func(t *testing.T) {
		var mu sync.Mutex
		var streamed bytes.Buffer
		res, err := sb.Exec(ctx, sandbox.ExecRequest{
			Cmd: `id -un; pwd; echo "$KEITH_SECRET|$MULTI|$EXTRA"; echo oops >&2; exit 3`,
			Env: map[string]string{"EXTRA": "e"},
		}, func(c []byte) { mu.Lock(); streamed.Write(c); mu.Unlock() })
		if err != nil {
			t.Fatal(err)
		}
		if res.ExitCode != 3 || res.TimedOut {
			t.Fatalf("exit = %d timedOut=%v", res.ExitCode, res.TimedOut)
		}
		if want := "agent\n/workspace\ns3cr3t value|a\nb|e\n"; res.Stdout != want {
			t.Fatalf("stdout = %q, want %q", res.Stdout, want)
		}
		if res.Stderr != "oops\n" {
			t.Fatalf("stderr = %q", res.Stderr)
		}
		mu.Lock()
		defer mu.Unlock()
		if !strings.Contains(streamed.String(), "s3cr3t value") || !strings.Contains(streamed.String(), "oops") {
			t.Fatalf("streamed = %q", streamed.String())
		}

		res, err = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "id -u", User: "root", Dir: "/tmp"}, nil)
		if err != nil || strings.TrimSpace(res.Stdout) != "0" {
			t.Fatalf("exec as root = %+v, %v", res, err)
		}
	})

	t.Run("exec timeout kills the process group", func(t *testing.T) {
		start := time.Now()
		res, err := sb.Exec(ctx, sandbox.ExecRequest{
			Cmd:     `echo started; (sleep 300; echo child-survived > /tmp/survivor) & sleep 300`,
			Timeout: time.Second,
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !res.TimedOut || res.ExitCode != -1 || res.Stdout != "started\n" {
			t.Fatalf("timeout result = %+v", res)
		}
		if d := time.Since(start); d > 8*time.Second {
			t.Fatalf("timeout took %v", d)
		}
		res, _ = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "ps -eo args | grep -c '[s]leep 300' || true"}, nil)
		if strings.TrimSpace(res.Stdout) != "0" {
			t.Fatalf("sleep processes left after timeout: %q", res.Stdout)
		}
	})

	t.Run("exec context cancel", func(t *testing.T) {
		cctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
		defer cancel()
		if _, err := sb.Exec(cctx, sandbox.ExecRequest{Cmd: "sleep 30"}, nil); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("files", func(t *testing.T) {
		big := make([]byte, 8<<20)
		_, _ = rand.Read(big)
		if err := sb.WriteFile(ctx, "repo/deep/dir/big.bin", big, 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := sb.ReadFile(ctx, "/workspace/repo/deep/dir/big.bin")
		if err != nil || !bytes.Equal(got, big) {
			t.Fatalf("ReadFile big: len %d, err %v", len(got), err)
		}
		res, _ := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "stat -c '%U %a %s' repo/deep/dir/big.bin"}, nil)
		if strings.TrimSpace(res.Stdout) != "agent 600 8388608" {
			t.Fatalf("stat = %q", res.Stdout)
		}
		if err := sb.WriteFile(ctx, "repo/a.txt", []byte("one"), 0); err != nil {
			t.Fatal(err)
		}
		if err := sb.WriteFile(ctx, "repo/a.txt", []byte("two"), 0); err != nil {
			t.Fatal(err)
		}
		if got, _ := sb.ReadFile(ctx, "repo/a.txt"); string(got) != "two" {
			t.Fatalf("overwrite = %q", got)
		}
		if err := sb.RemoveFile(ctx, "repo/a.txt"); err != nil {
			t.Fatal(err)
		}
		if _, err := sb.ReadFile(ctx, "repo/a.txt"); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("read removed: %v", err)
		}
		if err := sb.RemoveFile(ctx, "repo/a.txt"); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("remove missing: %v", err)
		}
		if _, err := sb.ReadFile(ctx, "repo"); err == nil || errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("read dir: %v", err)
		}
		if err := sb.WriteFile(ctx, "empty", nil, 0); err != nil {
			t.Fatal(err)
		}
		if got, err := sb.ReadFile(ctx, "empty"); err != nil || len(got) != 0 {
			t.Fatalf("empty file: %q, %v", got, err)
		}
	})

	t.Run("port url", func(t *testing.T) {
		if err := sb.WriteFile(ctx, "www/index.html", []byte("hello from the sandbox"), 0); err != nil {
			t.Fatal(err)
		}
		res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "busybox httpd -p 3000 -h /workspace/www"}, nil)
		if err != nil || res.ExitCode != 0 {
			t.Fatalf("start httpd: %+v %v", res, err)
		}
		u, err := sb.PortURL(ctx, 3000)
		if err != nil || !strings.HasPrefix(u, "http://127.0.0.1:") {
			t.Fatalf("PortURL = %q, %v", u, err)
		}
		var body []byte
		for i := 0; i < 50; i++ {
			resp, err := http.Get(u + "/index.html")
			if err == nil {
				body, _ = io.ReadAll(resp.Body)
				resp.Body.Close()
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		if string(body) != "hello from the sandbox" {
			t.Fatalf("GET %s = %q", u, body)
		}
		if _, err := sb.PortURL(ctx, 9999); !errors.Is(err, sandbox.ErrNotSupported) {
			t.Fatalf("unpublished port: %v", err)
		}
	})

	t.Run("pause resume", func(t *testing.T) {
		if res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "echo before > /tmp/state"}, nil); err != nil || res.ExitCode != 0 {
			t.Fatalf("write state: %+v %v", res, err)
		}
		for i := 0; i < 2; i++ { // second call is a no-op
			if err := sb.Pause(ctx); err != nil {
				t.Fatalf("pause %d: %v", i, err)
			}
		}
		if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "true"}, nil); err == nil {
			t.Fatal("exec in a paused sandbox succeeded")
		}
		for i := 0; i < 2; i++ {
			if err := sb.Resume(ctx); err != nil {
				t.Fatalf("resume %d: %v", i, err)
			}
		}
		res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "cat /tmp/state"}, nil)
		if err != nil || res.Stdout != "before\n" {
			t.Fatalf("after resume: %+v %v", res, err)
		}
	})

	t.Run("snapshot and create from it", func(t *testing.T) {
		if err := sb.WriteFile(ctx, "repo/node_modules/dep.js", []byte("module.exports = 1"), 0); err != nil {
			t.Fatal(err)
		}
		snap, err := sb.Snapshot(ctx, "Repo Template/v1")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = p.DeleteSnapshot(context.Background(), snap) })
		if !strings.HasPrefix(snap, "sha256:") {
			t.Fatalf("snapshot id = %q", snap)
		}
		sb2 := sbtest.Sandbox(t, p, sandbox.Spec{SnapshotID: snap})
		got, err := sb2.ReadFile(ctx, "repo/node_modules/dep.js")
		if err != nil || string(got) != "module.exports = 1" {
			t.Fatalf("file in snapshot sandbox: %q, %v", got, err)
		}
		res, _ := sb2.Exec(ctx, sandbox.ExecRequest{Cmd: "id -un"}, nil)
		if res.Stdout != "agent\n" {
			t.Fatalf("user in snapshot sandbox = %q", res.Stdout)
		}
		// Spec.Env (the org's secrets) must not be baked into the image.
		out, err := exec.Command("docker", "image", "inspect", "--format", "{{json .Config.Env}}", snap).Output()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(out), "s3cr3t") || strings.Contains(string(out), "MULTI=a") {
			t.Fatalf("snapshot image env holds a secret: %s", out)
		}
		if !strings.Contains(string(out), "PATH=") {
			t.Fatalf("snapshot image env lost the image's own variables: %s", out)
		}
		res, _ = sb2.Exec(ctx, sandbox.ExecRequest{Cmd: `echo "[$KEITH_SECRET]"`}, nil)
		if res.Stdout != "[]\n" {
			t.Fatalf("secret in snapshot sandbox env = %q", res.Stdout)
		}
		if _, err := p.Create(ctx, sandbox.Spec{SnapshotID: "sha256:" + strings.Repeat("0", 64)}); !errors.Is(err, sandbox.ErrNotFound) {
			t.Fatalf("create from unknown snapshot: %v", err)
		}
	})

	t.Run("delete", func(t *testing.T) {
		if err := sb.Delete(ctx); err != nil {
			t.Fatal(err)
		}
		if err := sb.Delete(ctx); err != nil {
			t.Fatalf("second delete: %v", err)
		}
		if _, err := p.Get(ctx, sb.ID()); !errors.Is(err, sandbox.ErrNotFound) {
			t.Fatalf("Get deleted: %v", err)
		}
		if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "true"}, nil); !errors.Is(err, sandbox.ErrNotFound) {
			t.Fatalf("exec in deleted sandbox: %v", err)
		}
	})
}
