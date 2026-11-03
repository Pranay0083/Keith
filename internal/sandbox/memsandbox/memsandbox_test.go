package memsandbox_test

import (
	"context"
	"errors"
	"io/fs"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

func TestFiles(t *testing.T) {
	ctx := context.Background()
	sb := memsandbox.New(memsandbox.WithFiles(map[string]string{"repo/README.md": "hi"}))

	got, err := sb.ReadFile(ctx, "/workspace/repo/README.md")
	if err != nil || string(got) != "hi" {
		t.Fatalf("ReadFile = %q, %v", got, err)
	}
	if err := sb.WriteFile(ctx, "repo/a/b.txt", []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if mode, ok := sb.FileMode("/workspace/repo/a/b.txt"); !ok || mode != 0o600 {
		t.Fatalf("mode = %v, %v", mode, ok)
	}
	if err := sb.RemoveFile(ctx, "repo/a/b.txt"); err != nil {
		t.Fatal(err)
	}
	if _, err := sb.ReadFile(ctx, "repo/a/b.txt"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("ReadFile after remove: %v", err)
	}
	if err := sb.RemoveFile(ctx, "nope"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("RemoveFile missing: %v", err)
	}

	var ops []string
	for _, c := range sb.Calls() {
		ops = append(ops, string(c.Op))
	}
	if got, want := strings.Join(ops, " "), "read write remove read remove"; got != want {
		t.Fatalf("ops = %s, want %s", got, want)
	}
}

func TestExecRouting(t *testing.T) {
	ctx := context.Background()
	sb := memsandbox.New(memsandbox.WithHandler(memsandbox.Route(
		memsandbox.On(memsandbox.Prefix("go test"), memsandbox.Sequence(
			memsandbox.Exit(1, "FAIL\n", ""),
			memsandbox.OK("ok\n"),
		)),
		memsandbox.On(memsandbox.Regexp(`^git (status|diff)`), memsandbox.OK("clean\n")),
	)))

	var streamed strings.Builder
	res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "go test ./..."}, func(c []byte) { streamed.Write(c) })
	if err != nil || res.ExitCode != 1 || res.Stdout != "FAIL\n" || streamed.String() != "FAIL\n" {
		t.Fatalf("first go test = %+v, %v (streamed %q)", res, err, streamed.String())
	}
	res, _ = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "go test ./..."}, nil)
	if res.ExitCode != 0 {
		t.Fatalf("second go test exit = %d", res.ExitCode)
	}
	res, _ = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "git status"}, nil)
	if res.Stdout != "clean\n" {
		t.Fatalf("git status = %+v", res)
	}
	res, _ = sb.Exec(ctx, sandbox.ExecRequest{Cmd: "make"}, nil)
	if res.ExitCode != 127 {
		t.Fatalf("unrouted exit = %d", res.ExitCode)
	}
	if n := len(sb.ExecRequests()); n != 4 {
		t.Fatalf("recorded %d execs", n)
	}

	boom := errors.New("boom")
	sb.SetHandler(memsandbox.Fail(boom))
	if _, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "x"}, nil); !errors.Is(err, boom) {
		t.Fatalf("Fail handler err = %v", err)
	}
}

func TestLifecycle(t *testing.T) {
	ctx := context.Background()
	p := memsandbox.NewProvider()
	p.NewHandler = func(sandbox.Spec) memsandbox.Handler { return memsandbox.OK("hi") }

	s, err := p.Create(ctx, sandbox.Spec{Template: "base", Labels: map[string]string{"session": "s1"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.WriteFile(ctx, "x", []byte("1"), 0); err != nil {
		t.Fatal(err)
	}
	if err := s.Pause(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Exec(ctx, sandbox.ExecRequest{Cmd: "true"}, nil); err == nil {
		t.Fatal("Exec on paused sandbox succeeded")
	}
	if err := s.Resume(ctx); err != nil {
		t.Fatal(err)
	}
	snap, err := s.Snapshot(ctx, "tmpl")
	if err != nil {
		t.Fatal(err)
	}
	s2, err := p.Create(ctx, sandbox.Spec{SnapshotID: snap})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s2.ReadFile(ctx, "x"); err != nil || string(got) != "1" {
		t.Fatalf("from snapshot: %q, %v", got, err)
	}
	if _, err := p.Create(ctx, sandbox.Spec{SnapshotID: "nope"}); !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("unknown snapshot: %v", err)
	}
	if err := s.Delete(ctx); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(ctx); err != nil {
		t.Fatalf("second delete: %v", err)
	}
	if _, err := p.Get(ctx, s.ID()); !errors.Is(err, sandbox.ErrNotFound) {
		t.Fatalf("Get deleted: %v", err)
	}
	if got, err := p.Get(ctx, s2.ID()); err != nil || got.ID() != s2.ID() {
		t.Fatalf("Get = %v, %v", got, err)
	}
	if specs := p.Specs(); len(specs) != 3 || specs[0].Labels["session"] != "s1" {
		t.Fatalf("specs = %+v", specs)
	}
}

func TestReadFileLimit(t *testing.T) {
	ctx := context.Background()
	sb := memsandbox.New(memsandbox.WithFiles(map[string]string{"a.txt": "12345"}))
	if got, err := sb.ReadFileLimit(ctx, "a.txt", 5); err != nil || string(got) != "12345" {
		t.Fatalf("at the limit = %q, %v", got, err)
	}
	if _, err := sb.ReadFileLimit(ctx, "a.txt", 4); !errors.Is(err, sandbox.ErrFileTooLarge) {
		t.Fatalf("over the limit = %v", err)
	}
	calls := sb.Calls()
	if len(calls) != 2 || calls[0].Limit != 5 || calls[1].Limit != 4 {
		t.Fatalf("calls = %+v", calls)
	}
	sb.ResetCalls()
	if _, err := sb.ReadFile(ctx, "a.txt"); err != nil || sb.Calls()[0].Limit != sandbox.MaxReadFile {
		t.Fatalf("ReadFile: %v, calls %+v", err, sb.Calls())
	}
}
