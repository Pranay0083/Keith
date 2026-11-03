package memstore

import (
	"context"
	"slices"
	"sort"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store"
)

// ClearSessionSandbox implements worker.Store: it clears the session's
// sandbox only if it is still sandboxID.
func (m *Store) ClearSessionSandbox(_ context.Context, id, sandboxID string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok || sandboxID == "" || x.s.SandboxID != sandboxID {
		return false, nil
	}
	x.s.SandboxID, x.s.SandboxProvider = "", ""
	m.touch(x)
	return true, nil
}

// ListStaleLiveSessions implements worker.Store.
func (m *Store) ListStaleLiveSessions(_ context.Context, before time.Time, limit int) ([]store.Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if limit <= 0 {
		limit = 100
	}
	var out []store.Session
	for _, x := range m.sessions {
		if slices.Contains(store.ActiveStatuses, x.s.Status) && x.s.UpdatedAt.Before(before) {
			s := x.s
			s.Repos = nil
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].UpdatedAt.Before(out[j].UpdatedAt) })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// SetRepos replaces a session's repos (tests: a repo lost meanwhile).
func (m *Store) SetRepos(id string, repos []store.SessionRepo) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if x, ok := m.get(id); ok {
		x.s.Repos = slices.Clone(repos)
	}
}

// SetSessionStatusCapped implements worker.Store, with the store's cap
// rule (see store.Store.SetSessionStatusCapped).
func (m *Store) SetSessionStatusCapped(ctx context.Context, orgID, id string, status events.SessionStatus, reason string, max int) (bool, error) {
	if !slices.Contains(store.ActiveStatuses, status) || max <= 0 {
		return true, m.SetSessionStatus(ctx, id, status, reason)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok || x.s.OrgID != orgID {
		return false, notFound("set session status capped")
	}
	if !slices.Contains(store.ActiveStatuses, x.s.Status) {
		n := 0
		for _, y := range m.sessions {
			if y.s.OrgID == orgID && slices.Contains(store.ActiveStatuses, y.s.Status) {
				n++
			}
		}
		if n >= max {
			return false, nil
		}
	}
	x.s.Status, x.s.StatusReason, x.s.FinishedAt = status, reason, nil
	x.statusLog = append(x.statusLog, status)
	m.touch(x)
	return true, nil
}
