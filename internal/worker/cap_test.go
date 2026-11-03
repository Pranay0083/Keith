package worker

import (
	"context"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/store"
)

// The per-org active-session cap holds when a parked session is resumed:
// two resumes the API queued together can both pass its check, so the
// worker re-checks, under the org lock, when it makes the session active
// again.
func TestResumeOverTheOrgSessionCapFails(t *testing.T) {
	ctx := context.Background()
	t.Run("message", func(t *testing.T) {
		e := newEnv(t)
		e.opts.MaxActiveSessionsPerOrg = 1
		s := e.newSession()
		e.llm.Push(llmtest.Text("one"))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		other := e.newSession() // queued: active, at the cap
		if _, err := e.st.AddInput(s.ID, store.InputMessage, "more please", "", testUser); err != nil {
			t.Fatal(err)
		}
		if err := e.resume(s.ID); err != nil {
			t.Fatal(err)
		}
		if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonSessionCap {
			t.Fatalf("over the cap: %s/%s", x.Status, x.StatusReason)
		}
		if len(e.llm.Requests()) != 1 {
			t.Error("the agent ran over the cap")
		}
		if p, _ := e.st.ListPendingInputs(ctx, s.ID); len(p) != 1 {
			t.Errorf("pending inputs = %d; the message must wait", len(p))
		}

		// Room again: the next message carries both.
		if err := e.st.SetSessionStatus(ctx, other.ID, events.StatusCompleted, ""); err != nil {
			t.Fatal(err)
		}
		var last string
		e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
			last = req.Messages[len(req.Messages)-1].Text()
			return llmtest.Text("two"), nil
		})
		if _, err := e.st.AddInput(s.ID, store.InputMessage, "and this", "", testUser); err != nil {
			t.Fatal(err)
		}
		if err := e.resume(s.ID); err != nil {
			t.Fatal(err)
		}
		if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
			t.Errorf("under the cap: %s", st)
		}
		if !strings.Contains(last, "more please") || !strings.Contains(last, "and this") {
			t.Errorf("model got %q", last)
		}
	})
	t.Run("answer", func(t *testing.T) {
		e := newEnv(t)
		e.opts.MaxActiveSessionsPerOrg = 1
		s := e.newSession()
		e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "Which?"}))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		e.newSession() // queued: active, at the cap
		if _, err := e.st.AddInput(s.ID, store.InputAnswer, "dayjs", "q1", testUser); err != nil {
			t.Fatal(err)
		}
		if err := e.resume(s.ID); err != nil {
			t.Fatal(err)
		}
		if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonSessionCap {
			t.Fatalf("over the cap: %s/%s", x.Status, x.StatusReason)
		}
		if len(e.llm.Requests()) != 1 {
			t.Error("the agent ran over the cap")
		}
	})
	t.Run("first run is never capped", func(t *testing.T) {
		e := newEnv(t)
		e.opts.MaxActiveSessionsPerOrg = 1
		e.newSession() // queued too: the API admitted both
		s := e.newSession()
		e.llm.Push(llmtest.Text("one"))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
			t.Errorf("status = %s", st)
		}
	})
}
