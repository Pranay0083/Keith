package worker

import (
	"testing"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// A deterministic git failure at the end of a turn (the agent removed
// the origin remote, the default branch was renamed and pruned) used to
// be returned as a plain task error: every retry failed the same way,
// asynq archived the task and the session stayed running forever. It
// fails the session at once instead.
func TestShipGitErrorFailsTheSession(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.sbRules = append(e.sbRules, memsandbox.On(memsandbox.Contains("git status --porcelain | head -c 1"),
		memsandbox.Exit(128, "", "fatal: bad revision 'origin/main..HEAD'")))
	e.llm.Push(llmtest.Text("Done."))
	if err := e.run(s.ID); err != nil {
		t.Fatalf("task error %v; want the session failed", err)
	}
	if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonShipFailed {
		t.Errorf("session = %s/%s", x.Status, x.StatusReason)
	}

	// Verify's git check fails the same way.
	e2 := newEnv(t)
	e2.profile(store.RepoProfile{RepoID: testRepo, TestCmd: "npm test"})
	s2 := e2.newSession()
	e2.sbRules = append(e2.sbRules, memsandbox.On(memsandbox.Contains("git status --porcelain | head -c 1"),
		memsandbox.Exit(128, "", "fatal: not a git repository")))
	e2.llm.Push(llmtest.Text("Done."))
	if err := e2.run(s2.ID); err != nil {
		t.Fatalf("verify: task error %v; want the session failed", err)
	}
	if x := e2.st.Session(s2.ID); x.Status != events.StatusFailed {
		t.Errorf("verify: session = %s/%s", x.Status, x.StatusReason)
	}
}
