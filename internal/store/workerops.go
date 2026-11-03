package store

import (
	"context"
	"slices"
	"time"

	"github.com/Pranay0083/keith/internal/events"
)

// Worker queries added after wave D (the reaper's compare-and-set, the
// sweeper and the session cap on resume). They live in their own file so
// the control-plane files stay as they were.

// ClearSessionSandbox clears the session's sandbox only if it is still
// sandboxID, and reports whether it did. The reaper uses it so that a
// sandbox another worker recorded meanwhile is never forgotten.
func (s *Store) ClearSessionSandbox(ctx context.Context, id, sandboxID string) (bool, error) {
	const op = "clear session sandbox"
	if !validID(id) || sandboxID == "" {
		return false, nil
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET sandbox_id = NULL, sandbox_provider = NULL, updated_at = now()
		WHERE id = $1 AND sandbox_id = $2`, id, sandboxID)
	if err != nil {
		return false, wrap(op, err)
	}
	return tag.RowsAffected() == 1, nil
}

// ListStaleLiveSessions returns sessions in an active status (queued,
// provisioning, setting_up or running) last updated before before, oldest
// first, at most limit of them (limit <= 0 means 100). The worker's
// sweeper wakes those no worker holds: their task was archived or lost.
func (s *Store) ListStaleLiveSessions(ctx context.Context, before time.Time, limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 100
	}
	return s.listSessions(ctx, "list stale live sessions", `
		SELECT `+sessionCols+` FROM sessions s
		WHERE s.status = ANY($1::text[]) AND s.updated_at < $2
		ORDER BY s.updated_at, s.id
		LIMIT $3`, statusStrings(ActiveStatuses), before, limit)
}

// SetSessionStatusCapped sets the session's status like SetSessionStatus,
// except when that would make a parked session (finished or
// waiting_for_user) active again while its org already has max active
// sessions (ActiveStatuses): then it changes nothing and reports false.
// The count runs under the org row lock, taken before the session row
// lock (the API's order, so the two cannot deadlock), which makes the
// cap exact against other workers and the API.
func (s *Store) SetSessionStatusCapped(ctx context.Context, orgID, id string, status events.SessionStatus, reason string, max int) (bool, error) {
	const op = "set session status capped"
	if !slices.Contains(ActiveStatuses, status) || max <= 0 {
		return true, s.SetSessionStatus(ctx, id, status, reason)
	}
	if !validID(id) || !validID(orgID) {
		return false, errNotFound(op)
	}
	ok := false
	err := s.WithTx(ctx, func(tx *Store) error {
		if err := tx.LockOrg(ctx, orgID); err != nil {
			return err
		}
		var cur string
		err := tx.db.QueryRow(ctx,
			`SELECT status FROM sessions WHERE id = $1 AND org_id = $2 FOR UPDATE`, id, orgID).Scan(&cur)
		if err != nil {
			return notFound(op, err)
		}
		if !slices.Contains(ActiveStatuses, events.SessionStatus(cur)) {
			n, err := tx.CountActiveSessions(ctx, orgID)
			if err != nil {
				return err
			}
			if n >= max {
				return nil
			}
		}
		ok = true
		return tx.SetSessionStatus(ctx, id, status, reason)
	})
	if err != nil {
		return false, err
	}
	return ok, nil
}
