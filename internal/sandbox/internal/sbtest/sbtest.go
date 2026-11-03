// Package sbtest has helpers for the sandbox integration tests, which run
// only with KEITH_IT=1 and a Docker daemon.
package sbtest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/docker"
)

// Integration skips the test unless KEITH_IT=1.
func Integration(t testing.TB) {
	t.Helper()
	if os.Getenv("KEITH_IT") != "1" {
		t.Skip("integration test: set KEITH_IT=1 (needs a Docker daemon)")
	}
}

// Labels are put on every container and image the tests create, so a
// leftover can be found: KEITH_IT_LABEL (key=value), or keith.test=it.
func Labels() map[string]string {
	k, v, ok := strings.Cut(os.Getenv("KEITH_IT_LABEL"), "=")
	if !ok || k == "" {
		return map[string]string{"keith.test": "it"}
	}
	return map[string]string{k: v}
}

// RepoRoot is the module root.
func RepoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "..")
}

var (
	imageOnce sync.Once
	imageTag  string
	imageErr  string
)

// TestImage builds sandbox/test.Dockerfile and returns its tag. The tag
// carries a hash of the Dockerfile, so the image is built once per
// Dockerfile version and reused across runs.
func TestImage(t testing.TB) string {
	t.Helper()
	imageOnce.Do(func() {
		root := RepoRoot()
		df := filepath.Join(root, "sandbox", "test.Dockerfile")
		b, err := os.ReadFile(df)
		if err != nil {
			imageErr = err.Error()
			return
		}
		sum := sha256.Sum256(b)
		tag := "keith-sandbox-test:" + hex.EncodeToString(sum[:])[:12]
		if exec.Command("docker", "image", "inspect", tag).Run() == nil {
			imageTag = tag
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		out, err := exec.CommandContext(ctx, "docker", "build", "-q", "-f", df, "-t", tag, filepath.Join(root, "sandbox")).CombinedOutput()
		if err != nil {
			imageErr = "docker build: " + err.Error() + "\n" + string(out)
			return
		}
		imageTag = tag
	})
	if imageErr != "" {
		t.Fatal(imageErr)
	}
	return imageTag
}

// DockerProvider returns a docker provider on the test image, with small
// limits and the test labels.
func DockerProvider(t testing.TB) *docker.Provider {
	t.Helper()
	Integration(t)
	return docker.New(docker.Options{
		Image:    TestImage(t),
		CPUs:     1,
		MemoryMB: 512,
		Ports:    []int{3000, 8080},
		Labels:   Labels(),
	})
}

// Sandbox creates a sandbox and deletes it when the test ends.
func Sandbox(t testing.TB, p sandbox.Provider, spec sandbox.Spec) sandbox.Sandbox {
	t.Helper()
	sb, err := p.Create(context.Background(), spec)
	if err != nil {
		t.Fatalf("create sandbox: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if err := sb.Delete(ctx); err != nil {
			t.Errorf("delete sandbox %s: %v", sb.ID(), err)
		}
	})
	return sb
}
