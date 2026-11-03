package worker

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// unpairedToolUses returns the tool_use IDs in req that the next message
// does not answer with a tool_result (what Anthropic rejects with a 400).
func unpairedToolUses(req llm.Request) []string {
	var out []string
	for i, m := range req.Messages {
		if m.Role != llm.RoleAssistant {
			continue
		}
		for _, b := range m.Content {
			if b.Type != llm.BlockToolUse {
				continue
			}
			found := false
			if i+1 < len(req.Messages) {
				for _, nb := range req.Messages[i+1].Content {
					if nb.Type == llm.BlockToolResult && nb.ToolUseID == b.ID {
						found = true
					}
				}
			}
			if !found {
				out = append(out, b.ID)
			}
		}
	}
	return out
}

// A lease loss (or crash) while a tool runs leaves the post-generate
// checkpoint: the assistant's tool_use with no result. The retry must
// answer it and call the model (not end the turn with the tool never
// run), and no later request may carry an unanswered tool_use.
func TestLeaseLostDuringToolIsAnsweredOnRetry(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	key := lease.SessionKey(s.ID)
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if cmd == "make build" {
			e.leases.Lose(key)
			time.Sleep(50 * time.Millisecond)
			return sandbox.ShellResult{OutputTail: "built\n"}, nil
		}
		return sandbox.ShellResult{OutputTail: "ok\n"}, nil
	}
	e.llm.Push(toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "make build"}))
	if err := e.run(s.ID); err == nil {
		t.Fatal("expected ErrBusy after the lost lease")
	}

	// The retry (another worker) resumes from agent_state.
	e.onExec = nil
	var unpaired [][]string
	check := func(req llm.Request) { unpaired = append(unpaired, unpairedToolUses(req)) }
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		check(req)
		last := req.Messages[len(req.Messages)-1]
		if len(last.Content) == 0 || last.Content[0].ToolUseID != "c1" || !last.Content[0].IsError {
			t.Errorf("retry request does not answer c1 with an error result: %+v", last.Content)
		}
		return llmtest.Text("ok"), nil
	})
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if e.llm.Remaining() != 0 {
		t.Fatal("the retry ended the turn without calling the model")
	}
	finished := 0
	for _, ev := range e.events(s.ID, events.TypeToolCallFinished) {
		if p := payload[events.ToolCallFinishedPayload](t, ev); p.CallID == "c1" && p.IsError && p.Summary == "interrupted" {
			finished++
		}
	}
	if finished != 1 {
		t.Errorf("tool_call.finished (interrupted) for c1 = %d, want 1", finished)
	}

	// A follow-up message.
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		check(req)
		return llmtest.Text("also done"), nil
	})
	if _, err := e.st.AddInput(s.ID, store.InputMessage, "and also X", "", testUser); err != nil {
		t.Fatal(err)
	}
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if len(unpaired) != 2 {
		t.Fatalf("model calls = %d, want 2", len(unpaired))
	}
	for i, u := range unpaired {
		if len(u) > 0 {
			t.Errorf("request %d has tool_use without tool_result: %v", i+1, u)
		}
	}
}

// SIGTERM mid-tool: tool_call.finished must still be stored (on a
// context detached from the cancellation), so the call does not stay
// running in the timeline, and the retry must not announce it again.
func TestSigtermMidToolStillFinishesTheCall(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if cmd == "make build" {
			cancel()
			return sandbox.ShellResult{}, context.Canceled
		}
		return sandbox.ShellResult{OutputTail: "ok\n"}, nil
	}
	e.llm.Push(toolCall("c1", "shell_exec", map[string]any{"shell": "main", "command": "make build"}))
	if err := e.worker().HandleRunSession(ctx, queue.RunSessionPayload{SessionID: s.ID}); err == nil {
		t.Fatal("expected ErrBusy after SIGTERM")
	}
	e.onExec = nil
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		if u := unpairedToolUses(req); len(u) > 0 {
			t.Errorf("retry request has tool_use without tool_result: %v", u)
		}
		return llmtest.Text("ok"), nil
	})
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	started, finished := 0, 0
	for _, ev := range e.evs.All(s.ID) {
		switch ev.Type {
		case events.TypeToolCallStarted:
			if payload[events.ToolCallStartedPayload](t, ev).CallID == "c1" {
				started++
			}
		case events.TypeToolCallFinished:
			if payload[events.ToolCallFinishedPayload](t, ev).CallID == "c1" {
				finished++
			}
		}
	}
	if started != 1 || finished != 1 {
		t.Errorf("c1: tool_call.started %d, tool_call.finished %d; want 1 and 1", started, finished)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("status = %s", st)
	}
}

// SIGTERM during setup: the worker's own setup call is finished too.
func TestSigtermDuringSetupStillFinishesTheCall(t *testing.T) {
	e := newEnv(t)
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	s := e.newSession()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e.onExec = func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		cancel()
		return sandbox.ShellResult{}, context.Canceled
	}
	if err := e.worker().HandleRunSession(ctx, queue.RunSessionPayload{SessionID: s.ID}); err == nil {
		t.Fatal("expected ErrBusy after SIGTERM")
	}
	started := e.events(s.ID, events.TypeToolCallStarted)
	finished := e.events(s.ID, events.TypeToolCallFinished)
	if len(started) != 1 || len(finished) != 1 {
		t.Fatalf("setup call: started %d, finished %d; want 1 and 1", len(started), len(finished))
	}
	if p := payload[events.ToolCallFinishedPayload](t, finished[0]); !p.IsError || p.Summary != "interrupted" {
		t.Errorf("finished = %+v", p)
	}
}

// racyLeaser's leases report a loss from KeepAlive when they are
// released, as a renewal in flight while Release deletes the key does.
type racyLeaser struct{ *MemLeaser }

func (r racyLeaser) Acquire(ctx context.Context, key string, ttl time.Duration) (Lease, error) {
	l, err := r.MemLeaser.Acquire(ctx, key, ttl)
	if err != nil {
		return nil, err
	}
	return &racyLease{Lease: l, renewal: make(chan error, 1)}, nil
}

type racyLease struct {
	Lease
	renewal chan error
}

func (l *racyLease) KeepAlive(ctx context.Context) <-chan error {
	out := make(chan error, 1)
	go func() {
		defer close(out)
		select {
		case <-ctx.Done():
		case err := <-l.renewal:
			out <- err
		}
	}()
	return out
}

func (l *racyLease) Release(ctx context.Context) error {
	l.renewal <- fmt.Errorf("lease: renew: %w", lease.ErrLost)
	time.Sleep(20 * time.Millisecond) // let a KeepAlive reader act on it
	return l.Lease.Release(ctx)
}

// A renewal racing the lease's release must not turn a clean finish
// into ErrBusy (a retry of a session that is done).
func TestRenewalRacingReleaseIsACleanFinish(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	w := e.worker()
	w.d.Leases = racyLeaser{e.leases}
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s.ID); err != nil {
		t.Fatalf("clean finish returned %v", err)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("status = %s", st)
	}
}

// crashAfterDeliverStore simulates a worker dying right after Delivered
// saved the state with the answer in it: the ack loses the lease and
// fails, once.
type crashAfterDeliverStore struct {
	Store
	arm  func() bool
	lose func()
}

func (c *crashAfterDeliverStore) MarkInputsConsumed(ctx context.Context, sessionID string, ids []string) (int64, error) {
	if len(ids) > 0 && c.arm() {
		c.lose()
		return 0, context.Canceled
	}
	return c.Store.MarkInputsConsumed(ctx, sessionID, ids)
}

// A crash between saving an answer and setting running: the retry finds
// the answer already delivered (it is acked as stale, and Delivered does
// not run again), yet the session must be running while the agent works.
func TestCrashAfterSavingAnAnswerStillSetsRunning(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	armed := false
	e.store = &crashAfterDeliverStore{Store: e.st,
		arm:  func() bool { a := armed; armed = false; return a },
		lose: func() { e.leases.Lose(lease.SessionKey(s.ID)) },
	}
	e.llm.Push(toolCall("q1", "ask_user", map[string]any{"question": "Which?"}))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusWaitingForUser {
		t.Fatalf("status = %s", st)
	}
	if _, err := e.st.AddInput(s.ID, store.InputAnswer, "dayjs", "q1", testUser); err != nil {
		t.Fatal(err)
	}
	armed = true
	if err := e.resume(s.ID); err == nil {
		t.Fatal("expected ErrBusy after the crash")
	}

	var during events.SessionStatus
	var openQ string
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		during, openQ = e.st.Session(s.ID).Status, e.st.Session(s.ID).OpenQuestionID
		return llmtest.Text("done"), nil
	})
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if during != events.StatusRunning || openQ != "" {
		t.Errorf("while the agent ran: status %s, open question %q; want running and none", during, openQ)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("final status = %s", st)
	}
	if p, _ := e.st.ListPendingInputs(context.Background(), s.ID); len(p) != 0 {
		t.Errorf("pending inputs = %d", len(p))
	}
}

// A message sent while the session runs gets no wakeup (the API enqueues
// one only for parked sessions). If the run then fails, the worker must
// not leave it pending until the user writes again: it gets a pass of
// its own, as a follow-up to the failed session.
func TestInputDuringAFailedRunIsNotStranded(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		if _, err := e.st.AddInput(s.ID, store.InputMessage, "also update the docs", "", testUser); err != nil {
			t.Error(err)
		}
		return nil, &llm.Error{Provider: "fake", Kind: llm.ErrInvalidRequest, Message: "bad request"}
	})
	var followUp string
	e.llm.PushFunc(func(req llm.Request) (*llm.Response, error) {
		followUp = req.Messages[len(req.Messages)-1].Text()
		return llmtest.Text("Docs updated."), nil
	})
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if p, _ := e.st.ListPendingInputs(context.Background(), s.ID); len(p) != 0 {
		t.Errorf("input left pending after the task: %d", len(p))
	}
	if !strings.Contains(followUp, "also update the docs") {
		t.Errorf("the model never got the message; last request text %q", followUp)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "running", "failed", "running", "no_changes"})
}
