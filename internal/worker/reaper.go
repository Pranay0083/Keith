package worker

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// reaperState remembers when each unreferenced sandbox was first seen.
type reaperState struct {
	mu      sync.Mutex
	orphans map[string]time.Time
}

// sandboxLister is implemented by providers that can list their
// sandboxes by label (docker).
type sandboxLister interface {
	List(ctx context.Context, labels map[string]string) ([]string, error)
}

// RunReaper reaps (Reap) and sweeps (Sweep) every ReapInterval until ctx
// ends.
func (w *Worker) RunReaper(ctx context.Context) {
	t := time.NewTicker(w.o.ReapInterval)
	defer t.Stop()
	for {
		n, err := w.Reap(ctx)
		if err != nil && ctx.Err() == nil {
			w.log.Warn("reap sandboxes", "err", err)
		}
		if n > 0 {
			w.log.Info("reaped sandboxes", "count", n)
		}
		if _, err := w.Sweep(ctx); err != nil && ctx.Err() == nil {
			w.log.Warn("sweep live sessions", "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// Reap runs one pass: it deletes the sandboxes of sessions parked
// (terminal or waiting_for_user) for longer than IdleTimeout, then
// orphaned sandboxes (see Options.OrphanLabels). It returns how many it
// deleted. A sandbox that cannot be deleted is logged and skipped, so
// it never holds up the others; the error is only for a pass that could
// not run at all.
func (w *Worker) Reap(ctx context.Context) (int, error) {
	n, ierr := w.reapIdle(ctx)
	m, oerr := w.reapOrphans(ctx)
	return n + m, errors.Join(ierr, oerr)
}

func (w *Worker) reapIdle(ctx context.Context) (int, error) {
	cutoff := w.d.Now().Add(-w.o.IdleTimeout)
	sessions, err := w.d.Store.ListIdleSandboxSessions(ctx, cutoff, 100)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, x := range sessions {
		if err := ctx.Err(); err != nil {
			return n, err
		}
		if x.SandboxProvider != w.d.Sandboxes.Name() {
			continue
		}
		ok, err := w.reapLeased(ctx, x.ID, x.SandboxID, cutoff)
		if err != nil {
			if ctx.Err() == nil {
				w.log.Warn("reap an idle sandbox; skipping it", "session_id", x.ID, "sandbox_id", x.SandboxID, "err", err)
			}
			continue
		}
		if ok {
			n++
		}
	}
	return n, nil
}

// reapLeased runs reapOne under the session's lease, kept alive while
// the delete runs. A session another worker holds is skipped.
func (w *Worker) reapLeased(ctx context.Context, sessionID, sandboxID string, cutoff time.Time) (bool, error) {
	l, err := w.d.Leases.Acquire(ctx, lease.SessionKey(sessionID), w.o.LeaseTTL)
	if errors.Is(err, lease.ErrHeld) {
		return false, nil // a worker is on it
	}
	if err != nil {
		return false, err
	}
	lctx, cancel := context.WithCancelCause(ctx)
	kaDone := make(chan struct{})
	go func() {
		defer close(kaDone)
		for err := range l.KeepAlive(lctx) {
			cancel(err)
		}
	}()
	ok, err := w.reapOne(lctx, sessionID, sandboxID, cutoff)
	cancel(nil)
	<-kaDone
	rctx, rcancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	_ = l.Release(rctx)
	rcancel()
	return ok, err
}

// reapOne deletes a session's sandbox if, under the lease, the session
// is still parked on it since before cutoff. It then clears the
// session's sandbox only if it is still that one (compare-and-set), so
// a sandbox another worker recorded meanwhile is kept.
func (w *Worker) reapOne(ctx context.Context, sessionID, sandboxID string, cutoff time.Time) (bool, error) {
	cur, err := w.d.Store.GetSession(ctx, sessionID)
	if err != nil {
		return false, err
	}
	parked := cur.Status.Terminal() || cur.Status == events.StatusWaitingForUser
	if !parked || cur.SandboxID != sandboxID || !cur.UpdatedAt.Before(cutoff) {
		return false, nil
	}
	sb, err := w.d.Sandboxes.Get(ctx, sandboxID)
	switch {
	case errors.Is(err, sandbox.ErrNotFound):
	case err != nil:
		return false, err
	default:
		if err := sb.Delete(ctx); err != nil {
			return false, err
		}
	}
	cleared, err := w.d.Store.ClearSessionSandbox(ctx, sessionID, sandboxID)
	if err != nil {
		return false, err
	}
	if !cleared {
		w.log.Info("deleted an idle sandbox; the session has moved to another one", "session_id", sessionID, "sandbox_id", sandboxID)
	} else {
		w.log.Info("deleted an idle sandbox", "session_id", sessionID, "sandbox_id", sandboxID)
	}
	return true, nil
}

func (w *Worker) reapOrphans(ctx context.Context) (int, error) {
	lister, ok := w.d.Sandboxes.(sandboxLister)
	if !ok || len(w.o.OrphanLabels) == 0 {
		return 0, nil
	}
	ids, err := lister.List(ctx, w.o.OrphanLabels)
	if err != nil {
		return 0, err
	}
	used, err := w.d.Store.SandboxIDsInUse(ctx, ids)
	if err != nil {
		return 0, err
	}
	now := w.d.Now()
	w.reaper.mu.Lock()
	if w.reaper.orphans == nil {
		w.reaper.orphans = map[string]time.Time{}
	}
	seen := map[string]bool{}
	var due []string
	for _, id := range ids {
		if used[id] {
			continue
		}
		seen[id] = true
		first, ok := w.reaper.orphans[id]
		if !ok {
			w.reaper.orphans[id] = now
			continue
		}
		if now.Sub(first) >= w.o.OrphanGrace {
			due = append(due, id)
		}
	}
	for id := range w.reaper.orphans {
		if !seen[id] {
			delete(w.reaper.orphans, id)
		}
	}
	w.reaper.mu.Unlock()

	n := 0
	for _, id := range due {
		sb, err := w.d.Sandboxes.Get(ctx, id)
		if err == nil {
			err = sb.Delete(ctx)
		}
		if err != nil && !errors.Is(err, sandbox.ErrNotFound) {
			w.log.Warn("delete an orphaned sandbox", "sandbox_id", id, "err", err)
			continue
		}
		w.reaper.mu.Lock()
		delete(w.reaper.orphans, id)
		w.reaper.mu.Unlock()
		w.log.Info("deleted an orphaned sandbox", "sandbox_id", id)
		n++
	}
	return n, nil
}
