package worker

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
)

// Recovery for sessions no task would run again: a task asynq is about
// to archive (TaskExhausted), and active sessions no worker holds (Sweep).

// exhaustedTimeout bounds TaskExhausted's work.
const exhaustedTimeout = 30 * time.Second

// Enqueuer enqueues wakeups (*queue.Client).
type Enqueuer interface {
	EnqueueResumeSession(ctx context.Context, p queue.ResumeSessionPayload) error
}

var _ Enqueuer = (*queue.Client)(nil)

// TaskExhausted is the session queue's last word on a task asynq is about
// to archive (wire it with queue.SessionErrorHandler): no retry will run
// the session again.
//
//   - An error wrapping queue.ErrBusy (another worker held the lease, or
//     this one was shut down or lost the lease) says nothing about the
//     session: it may still need a run, so a fresh wakeup is enqueued.
//   - Any other error fails the session, under its lease, if it still
//     needs a run (active, or waiting with an answer or a Stop pending),
//     so it is not left live with nothing to run it.
func (w *Worker) TaskExhausted(ctx context.Context, sessionID string, taskErr error) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), exhaustedTimeout)
	defer cancel()
	log := w.log.With("session_id", sessionID)
	if errors.Is(taskErr, queue.ErrBusy) {
		if err := w.wake(ctx, sessionID); err != nil {
			log.Warn("enqueue a wakeup for an exhausted session task", "err", err)
		} else {
			log.Info("enqueued a wakeup for an exhausted session task")
		}
		return
	}
	l, err := w.d.Leases.Acquire(ctx, lease.SessionKey(sessionID), w.o.LeaseTTL)
	if errors.Is(err, lease.ErrHeld) {
		return // a worker is on it
	}
	if err != nil {
		log.Warn("lease an exhausted session", "err", err)
		return
	}
	defer func() {
		rctx, rcancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		_ = l.Release(rctx)
		rcancel()
	}()
	s := newSession(w, sessionID, l)
	sess, err := w.d.Store.GetSession(ctx, sessionID)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			log.Warn("load an exhausted session", "err", err)
		}
		return
	}
	s.sess = sess
	if need, err := s.needsRun(ctx); err != nil || !need {
		if err != nil {
			log.Warn("check an exhausted session", "err", err)
		}
		return
	}
	if err := s.loadSecrets(ctx); err != nil { // for redaction only
		log.Warn("load secrets of an exhausted session", "err", err)
	}
	log.Error("session task ran out of retries; failing the session", "err", s.em.redact(taskErr.Error()))
	if err := s.fail(ctx, ReasonInternal,
		"The worker could not run this session after several attempts, so it was stopped. Send a message to try again."); err != nil {
		log.Warn("fail an exhausted session", "err", err)
	}
}

// needsRun reports whether the session has work a task would do: it is
// active, or waiting with an input or a Stop pending.
func (s *session) needsRun(ctx context.Context) (bool, error) {
	if slices.Contains(store.ActiveStatuses, s.sess.Status) {
		return true, nil
	}
	if s.sess.Status != events.StatusWaitingForUser {
		return false, nil
	}
	if c, err := s.w.d.Store.IsCancelRequested(ctx, s.id); err != nil || c {
		return c, err
	}
	pending, err := s.w.d.Store.ListPendingInputs(ctx, s.id)
	return len(pending) > 0, err
}

// Sweep enqueues a wakeup for each active session (queued, provisioning,
// setting_up or running) not updated for SweepAfter that no worker
// holds: its task is gone (archived, or lost with Redis) and nothing else
// would run it. A session whose task is merely waiting gets a duplicate
// wakeup, which is harmless. It returns how many it woke. Without
// Deps.Queue it does nothing.
func (w *Worker) Sweep(ctx context.Context) (int, error) {
	if w.d.Queue == nil {
		return 0, nil
	}
	sessions, err := w.d.Store.ListStaleLiveSessions(ctx, w.d.Now().Add(-w.o.SweepAfter), 100)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, x := range sessions {
		if err := ctx.Err(); err != nil {
			return n, err
		}
		l, err := w.d.Leases.Acquire(ctx, lease.SessionKey(x.ID), w.o.LeaseTTL)
		if errors.Is(err, lease.ErrHeld) {
			continue // a worker is on it
		}
		if err != nil {
			w.log.Warn("sweep: lease a session", "session_id", x.ID, "err", err)
			continue
		}
		rctx, rcancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		_ = l.Release(rctx)
		rcancel()
		if err := w.wake(ctx, x.ID); err != nil {
			w.log.Warn("sweep: enqueue a wakeup", "session_id", x.ID, "err", err)
			continue
		}
		w.log.Info("woke a live session no worker holds", "session_id", x.ID, "status", x.Status, "updated_at", x.UpdatedAt)
		n++
	}
	return n, nil
}

func (w *Worker) wake(ctx context.Context, sessionID string) error {
	if w.d.Queue == nil {
		return errors.New("worker: no Deps.Queue to enqueue a wakeup")
	}
	return w.d.Queue.EnqueueResumeSession(ctx, queue.ResumeSessionPayload{SessionID: sessionID, Reason: queue.ResumeWakeup})
}
