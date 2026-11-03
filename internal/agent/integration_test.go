package agent_test

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

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/docker"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
)

// itLabels tag this package's containers for cleanup.
var itLabels = map[string]string{"keith.test": "c1"}

// testImage builds sandbox/test.Dockerfile with the same tag scheme as the
// sandbox integration tests (keith-sandbox-test:<hash>), so they share it.
func testImage(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..")
	df := filepath.Join(root, "sandbox", "test.Dockerfile")
	b, err := os.ReadFile(df)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	tag := "keith-sandbox-test:" + hex.EncodeToString(sum[:])[:12]
	if exec.Command("docker", "image", "inspect", tag).Run() == nil {
		return tag
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "docker", "build", "-q", "-f", df, "-t", tag, filepath.Join(root, "sandbox")).CombinedOutput(); err != nil {
		t.Fatalf("docker build: %v\n%s", err, out)
	}
	return tag
}

// TestAgentInDockerSandbox runs a scripted model through the real shell
// and file tools: a docker sandbox, tmux shells and ripgrep.
//
//	KEITH_IT=1 go test ./internal/agent -run TestAgentInDockerSandbox
func TestAgentInDockerSandbox(t *testing.T) {
	if os.Getenv("KEITH_IT") != "1" {
		t.Skip("integration test: set KEITH_IT=1 (needs a Docker daemon)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	p := docker.New(docker.Options{Image: testImage(t), CPUs: 1, MemoryMB: 512, Ports: []int{}, Labels: itLabels})
	sb, err := p.Create(ctx, sandbox.Spec{Labels: itLabels})
	if err != nil {
		t.Fatalf("create sandbox: %v", err)
	}
	t.Cleanup(func() {
		dctx, dcancel := context.WithTimeout(context.Background(), time.Minute)
		defer dcancel()
		if err := sb.Delete(dctx); err != nil {
			t.Errorf("delete sandbox %s: %v", sb.ID(), err)
		}
	})

	rec := &agenttest.Recorder{}
	run := &agent.Run{SessionID: "it", Sandbox: sb, Emit: rec.Emit,
		Repos: []agent.RepoContext{{FullName: "acme/demo", Path: "/workspace/demo", Branch: "agent/it", DefaultBranch: "main"}}}
	var mu sync.Mutex
	outputCalls := map[string]bool{}
	run.Shells = tmuxshell.New(sb,
		tmuxshell.WithTimeouts(20*time.Second, 5*time.Second, time.Minute),
		tmuxshell.WithOutput(func(shell string, chunk []byte) {
			// What the worker does: tag the chunk with the reading call.
			mu.Lock()
			defer mu.Unlock()
			if id := run.CurrentCall(shell); id != "" {
				outputCalls[id] = true
			}
		}))

	prov := agenttest.Record(llmtest.New(
		llmtest.ToolCall("c1", "create_shell", map[string]string{"name": "main"}),
		llmtest.ToolCall("c2", "write_file", map[string]string{"path": "demo/hello.sh", "content": "#!/bin/sh\necho \"hello from $(whoami)\"\n"}),
		llmtest.ToolCall("c3", "shell_exec", map[string]string{"shell": "main", "command": "cd /workspace/demo && sh hello.sh"}),
		llmtest.ToolCall("c4", "edit_file", map[string]string{"path": "/workspace/demo/hello.sh", "old_string": "hello from", "new_string": "goodbye from"}),
		llmtest.ToolCalls(
			call("c5", "shell_exec", map[string]string{"shell": "main", "command": "sh hello.sh; false"}),
			call("c6", "search", map[string]string{"pattern": "goodbye", "path": "demo"}),
			call("c7", "read_file", map[string]string{"path": "demo/hello.sh"}),
		),
		llmtest.ToolCall("c8", "undo_edit", map[string]string{"path": "demo/hello.sh"}),
		llmtest.Text("done"),
	))
	r, err := agent.NewRunner(agent.Config{Provider: prov, Model: "test", Tools: agent.DefaultTools(agent.ToolOptions{}), NewID: agenttest.IDs()})
	if err != nil {
		t.Fatal(err)
	}
	st := agent.NewState("integration")
	out := r.Run(ctx, run, st)
	if out.Kind != agent.OutcomeEndTurn {
		t.Fatalf("outcome = %+v", out)
	}
	rec.Check(t)
	calls := prov.Calls()
	agenttest.CheckAppendOnly(t, calls)

	result := func(call int, i int) llm.Block { return agenttest.ToolResults(calls[call].Req)[i] }
	if r := result(3, 0); r.IsError || !strings.Contains(text(r), "Exit code: 0") || !strings.Contains(text(r), "hello from agent") {
		t.Fatalf("first run = %q", text(r))
	}
	if r := result(5, 0); !r.IsError || !strings.Contains(text(r), "Exit code: 1") || !strings.Contains(text(r), "goodbye from agent") {
		t.Fatalf("second run = %q", text(r))
	}
	if r := result(5, 1); r.IsError || !strings.Contains(text(r), "demo/hello.sh:2:") {
		t.Fatalf("search = %q", text(r))
	}
	if r := result(5, 2); !strings.Contains(text(r), "     2\techo \"goodbye from $(whoami)\"") {
		t.Fatalf("read_file = %q", text(r))
	}
	got, err := sb.ReadFile(ctx, "/workspace/demo/hello.sh")
	if err != nil || !strings.Contains(string(got), "hello from") {
		t.Fatalf("after undo: %q %v", got, err)
	}
	mu.Lock()
	if !outputCalls["c3"] || !outputCalls["c5"] {
		t.Errorf("shell output not bound to its calls: %v", outputCalls)
	}
	mu.Unlock()
	if n := len(agenttest.Payloads[events.ShellCreatedPayload](rec, events.TypeShellCreated)); n != 1 {
		t.Errorf("%d shell.created", n)
	}
	if n := len(agenttest.Payloads[events.FileChangedPayload](rec, events.TypeFileChanged)); n != 3 {
		t.Errorf("%d file.changed", n)
	}
}
