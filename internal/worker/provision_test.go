package worker

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// An interrupted provision (the lease lost, or a crash, during the
// clone) leaves sessions.sandbox_id set. The retry must not resume that
// sandbox, which has no clone and no setup: it provisions a new one.
func TestInterruptedProvisionIsProvisionedAgain(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	lost := false
	e.sbRules = append(e.sbRules, memsandbox.On(func(req sandbox.ExecRequest) bool {
		return !lost && strings.Contains(req.Cmd, "git clone")
	}, func(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		lost = true
		e.leases.Lose(lease.SessionKey(s.ID))
		return nil, context.Canceled
	}))
	if err := e.run(s.ID); err == nil {
		t.Fatal("expected ErrBusy after the lost lease")
	}
	half := e.sandbox(s.ID)

	setupRan := false
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if strings.Contains(cmd, "npm ci") {
			setupRan = true
		}
		return sandbox.ShellResult{OutputTail: "ok\n"}, nil
	}
	e.llm.Push(llmtest.Text("Done."))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	sb := e.sandbox(s.ID)
	if sb.ID() == half.ID() {
		t.Fatal("the retry resumed the half-provisioned sandbox")
	}
	if !half.Deleted() {
		t.Error("the half-provisioned sandbox was not deleted")
	}
	if n := len(e.execs(sb, "git clone")); n != 1 || !setupRan {
		t.Errorf("new sandbox: clones = %d, setup ran = %v", n, setupRan)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("status = %s", st)
	}

	// The finished sandbox is resumed, not replaced, on a follow-up.
	if _, err := e.st.AddInput(s.ID, store.InputMessage, "and one more", "", testUser); err != nil {
		t.Fatal(err)
	}
	e.llm.Push(llmtest.Text("Done again."))
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if got := e.sandbox(s.ID); got.ID() != sb.ID() || len(e.execs(got, "git clone")) != 1 {
		t.Errorf("follow-up: sandbox %s (was %s), clones %d", got.ID(), sb.ID(), len(e.execs(got, "git clone")))
	}
	if n := len(e.sbx.Specs()); n != 2 {
		t.Errorf("sandboxes created = %d, want 2", n)
	}
}

// A failed clone (clone_failed) leaves sessions.sandbox_id set; a
// follow-up message must not run the agent in that sandbox, which has
// no repo.
func TestFailedCloneIsProvisionedAgainOnFollowUp(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	fail := true
	e.sbRules = append(e.sbRules, memsandbox.On(func(req sandbox.ExecRequest) bool {
		return fail && strings.Contains(req.Cmd, "git clone")
	}, memsandbox.Exit(128, "", "fatal: unable to access: 502")))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonCloneFailed {
		t.Fatalf("after the clone failure: %s/%s", x.Status, x.StatusReason)
	}
	broken := e.sandbox(s.ID)

	fail = false
	if _, err := e.st.AddInput(s.ID, store.InputMessage, "please retry", "", testUser); err != nil {
		t.Fatal(err)
	}
	e.llm.Push(llmtest.Text("Done."))
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	sb := e.sandbox(s.ID)
	if sb.ID() == broken.ID() || !broken.Deleted() {
		t.Errorf("follow-up ran in the sandbox whose clone failed (%s, deleted %v)", sb.ID(), broken.Deleted())
	}
	if n := len(e.execs(sb, "git clone")); n != 1 {
		t.Errorf("clones in the new sandbox = %d", n)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("status = %s", st)
	}
}

// The credential helper is installed by a root script. Root must not
// run anything found through the sandbox user's PATH, whose first entry
// (~/.local/bin) the agent can write: every command is a shell builtin
// or an absolute path.
func TestRootScriptUsesAbsolutePaths(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	var roots []sandbox.ExecRequest
	for _, r := range e.sandbox(s.ID).ExecRequests() {
		if r.User == "root" {
			roots = append(roots, r)
		}
	}
	if len(roots) != 1 {
		t.Fatalf("root execs = %d", len(roots))
	}
	builtins := map[string]bool{"set": true, "export": true, "printf": true}
	for _, line := range strings.Split(roots[0].Cmd, "\n") {
		for _, part := range strings.Split(line, "|") {
			f := strings.Fields(part)
			if len(f) == 0 || strings.HasPrefix(f[0], "PATH=/usr/sbin:/usr/bin:/sbin:/bin") {
				continue
			}
			if !builtins[f[0]] && !strings.HasPrefix(f[0], "/") {
				t.Errorf("root script runs %q through PATH", f[0])
			}
		}
	}
}
