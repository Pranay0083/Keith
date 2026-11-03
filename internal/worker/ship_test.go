package worker

import (
	"testing"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
)

// The agent left the repo on another branch: the worker's ship must not
// commit there, push a stale session branch, or open a PR for it.
func TestShipChecksHEADIsTheSessionBranch(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.Push(llmtest.Text("Done."))
	e.setChanges(true, 0)
	e.mu.Lock()
	e.branch = "main"
	e.mu.Unlock()
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonShipFailed {
		t.Errorf("session = %s/%s", x.Status, x.StatusReason)
	}
	sb := e.sandbox(s.ID)
	if n, m := len(e.execs(sb, " commit ")), len(e.execs(sb, "git push")); n+m != 0 {
		t.Errorf("commits %d, pushes %d on the wrong branch", n, m)
	}
	if prs := e.gh.host.PRRequests(); len(prs) != 0 {
		t.Errorf("PRs = %+v", prs)
	}
}
