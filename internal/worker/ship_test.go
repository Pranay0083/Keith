package worker

import (
	"strings"
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
	if n, m := len(e.execs(sb, " commit ")), len(e.execs(sb, "git push"))+len(e.pushes()); n+m != 0 {
		t.Errorf("commits %d, pushes %d on the wrong branch", n, m)
	}
	if prs := e.gh.host.PRRequests(); len(prs) != 0 {
		t.Errorf("PRs = %+v", prs)
	}
}

// budget_exceeded is cumulative per session, so a follow-up message
// fails the same way (a known limitation): the failure message must not
// tell the user to send one to continue.
func TestBudgetExceededDoesNotSuggestAFollowUp(t *testing.T) {
	e := newEnv(t)
	e.opts.MaxSessionTokens = 10 // one model call (15 tokens) uses it up
	s := e.newSession()
	e.llm.Push(toolCall("c1", "message_user", map[string]any{"text": "working"}))
	e.llm.Push(llmtest.Text("more"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if x := e.st.Session(s.ID); x.StatusReason != ReasonBudgetExceeded {
		t.Fatalf("session = %s/%s", x.Status, x.StatusReason)
	}
	errs := e.events(s.ID, events.TypeError)
	if len(errs) != 1 {
		t.Fatalf("error events = %d", len(errs))
	}
	msg := payload[events.ErrorPayload](t, errs[0]).Message
	if strings.Contains(msg, "Send a message to continue") || !strings.Contains(msg, "MAX_SESSION_TOKENS") {
		t.Errorf("budget message = %q", msg)
	}
}
