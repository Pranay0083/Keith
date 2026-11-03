package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

func TestSandboxBench(t *testing.T) {
	p := memsandbox.NewProvider()
	p.NewHandler = func(sandbox.Spec) memsandbox.Handler { return memsandbox.OK("ok\n") }

	var out, progress bytes.Buffer
	if err := runSandboxBench(context.Background(), &out, &progress, p, benchOptions{N: 3, Parallel: 2, Labels: map[string]string{"team": "x"}}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"provider mem", "create", "first_exec", "pause", "resume", "exec_after_resume", "delete", "median"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report lacks %q:\n%s", want, out.String())
		}
	}
	specs := p.Specs()
	if len(specs) != 3 || specs[0].Labels["keith.bench"] != "1" || specs[0].Labels["team"] != "x" {
		t.Fatalf("specs = %+v", specs)
	}
	for i := 1; i <= 3; i++ {
		sb, ok := p.Sandbox("mem-p" + string(rune('0'+i)))
		if !ok || !sb.Deleted() {
			t.Errorf("sandbox %d not deleted", i)
		}
	}

	// JSON output and a failing exec.
	p.NewHandler = func(sandbox.Spec) memsandbox.Handler { return memsandbox.Exit(1, "", "boom") }
	out.Reset()
	err := runSandboxBench(context.Background(), &out, &progress, p, benchOptions{N: 1, JSON: true})
	if err == nil || !strings.Contains(err.Error(), "1 of 1 runs failed") {
		t.Fatalf("err = %v", err)
	}
	var rep benchReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Provider != "mem" || len(rep.Runs) != 1 || !strings.Contains(rep.Runs[0].Error, "first_exec") || rep.Runs[0].PhasesMS["create"] < 0 {
		t.Fatalf("report = %+v", rep)
	}
	if sb, _ := p.Sandbox(rep.Runs[0].SandboxID); !sb.Deleted() {
		t.Error("failed run's sandbox not deleted")
	}
}
