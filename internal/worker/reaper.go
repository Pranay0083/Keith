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

// RunReaper reaps every ReapInterval until ctx ends.
func (w *Worker) RunReaper(ctx context.Context) {
	t := time.NewTicker(w.o.ReapInterval)
	defer t.Stop()
	for {
		if n, err := w.Reap(ctx); err != nil && ctx.Err() == nil {
			w.log.Warn("reap sandboxes", "err", err)
		} else if n > 0 {
			w.log.Info("reaped sandboxes", "count", n)
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
// deleted.
func (w *Worker) Reap(ctx context.Context) (int, error) {
	n, err := w.reapIdle(ctx)
	if err != nil {
		return n, err
	}
	m, err := w.reapOrphans(ctx)
	return n + m, err
}

func (w *Worker) reapIdle(ctx context.Context) (int, error) {
	cutoff := w.d.Now().Add(-w.o.IdleTimeout)
	sessions, err := w.d.Store.ListIdleSandboxSessions(ctx, cutoff, 100)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, x := range sessions {
		if x.SandboxProvider != w.d.Sandboxes.Name() {
			continue
		}
		l, err := w.d.Leases.Acquire(ctx, lease.SessionKey(x.ID), w.o.LeaseTTL)
		if errors.Is(err, lease.ErrHeld) {
			continue // a worker is on it
		}
		if err != nil {
			return n, err
		}
		ok, err := w.reapOne(ctx, x.ID, x.SandboxID, cutoff)
		rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		_ = l.Release(rctx)
		cancel()
		if err != nil {
			return n, err
		}
		if ok {
			n++
		}
	}
	return n, nil
}

// reapOne deletes a session's sandbox if, under the lease, the session
// is still parked on it since before cutoff.
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
	if err := w.d.Store.SetSessionSandbox(ctx, sessionID, "", ""); err != nil {
		return false, err
	}
	w.log.Info("deleted an idle sandbox", "session_id", sessionID, "sandbox_id", sandboxID)
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
