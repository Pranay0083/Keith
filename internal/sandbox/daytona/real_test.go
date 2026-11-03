package daytona

import (
	"bytes"
	"context"
	"errors"
	"io/fs"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
)

// TestRealRoundTrip runs against a real Daytona organization. It needs
// DAYTONA_API_KEY; DAYTONA_API_URL, DAYTONA_TARGET and
// KEITH_DAYTONA_SNAPSHOT (a snapshot name) are optional. Set
// KEITH_DAYTONA_TEST_SNAPSHOT=1 to also snapshot the sandbox and boot a
// second one from it (slow; the snapshot is left in the organization).
func TestRealRoundTrip(t *testing.T) {
	key := os.Getenv("DAYTONA_API_KEY")
	if key == "" {
		t.Skip("DAYTONA_API_KEY not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	p, err := New(Options{
		APIKey:           key,
		APIURL:           os.Getenv("DAYTONA_API_URL"),
		Target:           os.Getenv("DAYTONA_TARGET"),
		DefaultSnapshot:  os.Getenv("KEITH_DAYTONA_SNAPSHOT"),
		AutoStopInterval: 10 * time.Minute,
		Labels:           map[string]string{"keith.test": "real-roundtrip"},
	})
	if err != nil {
		t.Fatal(err)
	}

	create := func(spec sandbox.Spec) sandbox.Sandbox {
		t.Helper()
		sb, err := p.Create(ctx, spec)
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		t.Cleanup(func() {
			dctx, dcancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer dcancel()
			if err := sb.Delete(dctx); err != nil {
				t.Errorf("cleanup Delete %s: %v", sb.ID(), err)
			}
		})
		return sb
	}

	sb := create(sandbox.Spec{
		Env:    map[string]string{"KEITH_SECRET": "from-spec"},
		Labels: map[string]string{"keith.session": "test"},
	})
	t.Logf("sandbox %s", sb.ID())

	// Exec: output, exit code, env, dir.
	var streamed bytes.Buffer
	res, err := sb.Exec(ctx, sandbox.ExecRequest{
		Cmd: `echo "out:$KEITH_SECRET:$PER_CMD:$(pwd)"; echo err >&2; exit 4`,
		Env: map[string]string{"PER_CMD": "x y"},
	}, func(c []byte) { streamed.Write(c) })
	if err != nil {
		t.Fatalf("Exec: %v", err)
	}
	if res.ExitCode != 4 || res.Stdout != "out:from-spec:x y:/workspace\n" || res.Stderr != "err\n" {
		t.Errorf("Exec result = %+v", res)
	}
	if !strings.Contains(streamed.String(), "out:") {
		t.Errorf("streamed = %q", streamed.String())
	}

	res, err = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "sleep 30", Timeout: 2 * time.Second}, nil)
	if err != nil || !res.TimedOut || res.ExitCode != -1 {
		t.Errorf("timeout Exec = %+v, %v", res, err)
	}

	// Files.
	big := bytes.Repeat([]byte("keith-"), 2<<20) // 12 MiB
	if err := sb.WriteFile(ctx, "dir/sub/big.bin", big, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	got, err := sb.ReadFile(ctx, "/workspace/dir/sub/big.bin")
	if err != nil || !bytes.Equal(got, big) {
		t.Fatalf("ReadFile: %d bytes, %v", len(got), err)
	}
	res, err = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "stat -c %a dir/sub/big.bin"}, nil)
	if err != nil || strings.TrimSpace(res.Stdout) != "600" {
		t.Errorf("mode = %+v, %v", res, err)
	}
	if err := sb.RemoveFile(ctx, "dir/sub/big.bin"); err != nil {
		t.Fatalf("RemoveFile: %v", err)
	}
	if _, err := sb.ReadFile(ctx, "dir/sub/big.bin"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("ReadFile after remove = %v; want fs.ErrNotExist", err)
	}

	if u, err := sb.PortURL(ctx, 3000); err != nil || u == "" {
		t.Errorf("PortURL = %q, %v", u, err)
	}

	// Pause / Resume, each twice to check the no-op path.
	for range 2 {
		if err := sb.Pause(ctx); err != nil {
			t.Fatalf("Pause: %v", err)
		}
	}
	for range 2 {
		if err := sb.Resume(ctx); err != nil {
			t.Fatalf("Resume: %v", err)
		}
	}
	res, err = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "echo back"}, nil)
	if err != nil || res.Stdout != "back\n" {
		t.Errorf("Exec after resume = %+v, %v", res, err)
	}

	if os.Getenv("KEITH_DAYTONA_TEST_SNAPSHOT") == "1" {
		if err := sb.WriteFile(ctx, "marker.txt", []byte("snap"), 0o644); err != nil {
			t.Fatal(err)
		}
		snap, err := sb.Snapshot(ctx, "keith real test")
		if err != nil {
			t.Fatalf("Snapshot: %v", err)
		}
		t.Logf("snapshot %s (not deleted)", snap)
		sb2 := create(sandbox.Spec{SnapshotID: snap})
		if data, err := sb2.ReadFile(ctx, "marker.txt"); err != nil || string(data) != "snap" {
			t.Errorf("marker from snapshot = %q, %v", data, err)
		}
	}

	// Delete twice, then Get is not found.
	if err := sb.Delete(ctx); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := sb.Delete(ctx); err != nil {
		t.Errorf("second Delete: %v", err)
	}
	if _, err := p.Get(ctx, sb.ID()); !errors.Is(err, sandbox.ErrNotFound) {
		t.Errorf("Get after Delete = %v; want ErrNotFound", err)
	}
}
