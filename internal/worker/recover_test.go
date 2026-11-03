package worker

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// fakeQueue records enqueued wakeups.
type fakeQueue struct {
	mu  sync.Mutex
	got []queue.ResumeSessionPayload
}

func (q *fakeQueue) EnqueueResumeSession(_ context.Context, p queue.ResumeSessionPayload) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.got = append(q.got, p)
	return nil
}

func (q *fakeQueue) wakeups() []queue.ResumeSessionPayload {
	q.mu.Lock()
	defer q.mu.Unlock()
	return append([]queue.ResumeSessionPayload(nil), q.got...)
}

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

// A task asynq is about to archive: a real error fails the live session
// (once, under its lease), ErrBusy enqueues a fresh wakeup instead.
func TestTaskExhaustedFailsOrWakesTheSession(t *testing.T) {
	e := newEnv(t)
	q := &fakeQueue{}
	w := e.worker()
	w.d.Queue = q
	ctx := context.Background()
	s := e.newSession()
	if err := e.st.SetSessionStatus(ctx, s.ID, events.StatusRunning, ""); err != nil {
		t.Fatal(err)
	}

	w.TaskExhausted(ctx, s.ID, fmt.Errorf("worker: session %s: interrupted: %w", s.ID, queue.ErrBusy))
	if got := q.wakeups(); len(got) != 1 || got[0].SessionID != s.ID || got[0].Reason != queue.ResumeWakeup {
		t.Errorf("wakeups after ErrBusy = %+v", got)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusRunning {
		t.Errorf("ErrBusy changed the status to %s", st)
	}

	// Another worker holds the session: leave it alone.
	l, err := e.leases.Acquire(ctx, lease.SessionKey(s.ID), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	w.TaskExhausted(ctx, s.ID, errors.New("store: connection refused"))
	if st := e.st.Session(s.ID).Status; st != events.StatusRunning {
		t.Errorf("held session: status %s", st)
	}
	_ = l.Release(ctx)

	w.TaskExhausted(ctx, s.ID, errors.New("store: connection refused"))
	if x := e.st.Session(s.ID); x.Status != events.StatusFailed || x.StatusReason != ReasonInternal {
		t.Errorf("after an exhausted task: %s/%s", x.Status, x.StatusReason)
	}
	if n := len(e.events(s.ID, events.TypeError)); n != 1 {
		t.Errorf("error events = %d", n)
	}
	if e.leases.Held(lease.SessionKey(s.ID)) {
		t.Error("lease left held")
	}

	// A settled session is not touched.
	w.TaskExhausted(ctx, s.ID, errors.New("again"))
	if n := len(e.events(s.ID, events.TypeError)); n != 1 {
		t.Errorf("a failed session was failed again: %d error events", n)
	}
}

// A live session that no worker holds and that has not moved for a
// while has lost its task: the sweeper wakes it.
func TestSweepWakesALiveSessionNoWorkerHolds(t *testing.T) {
	e := newEnv(t)
	q := &fakeQueue{}
	w := e.worker()
	w.d.Queue = q
	ctx := context.Background()

	stranded := e.newSession()
	held := e.newSession()
	fresh := e.newSession()
	parked := e.newSession()
	for _, id := range []string{stranded.ID, held.ID, fresh.ID} {
		if err := e.st.SetSessionStatus(ctx, id, events.StatusRunning, ""); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.st.SetSessionStatus(ctx, parked.ID, events.StatusCompleted, ""); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{stranded.ID, held.ID, parked.ID} {
		e.st.Backdate(id, w.o.SweepAfter+time.Minute)
	}
	l, err := e.leases.Acquire(ctx, lease.SessionKey(held.ID), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Release(ctx)

	n, err := w.Sweep(ctx)
	if err != nil {
		t.Fatal(err)
	}
	got := q.wakeups()
	if n != 1 || len(got) != 1 || got[0].SessionID != stranded.ID || got[0].Reason != queue.ResumeWakeup {
		t.Fatalf("sweep woke %d: %+v; want only %s", n, got, stranded.ID)
	}
	if e.leases.Held(lease.SessionKey(stranded.ID)) {
		t.Error("sweep left the lease held")
	}

	// The wakeup resumes the session where it was.
	e.llm.Push(llmtest.Text("Recovered."))
	if err := e.worker().HandleResumeSession(ctx, got[0]); err != nil {
		t.Fatal(err)
	}
	if st := e.st.Session(stranded.ID).Status; st != events.StatusNoChanges {
		t.Errorf("after the wakeup: %s", st)
	}
}
