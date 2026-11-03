package store

import (
	"context"
	"time"
)

// Worker queries added after wave D (the reaper, the sweeper and the
// worker's repo access check). They live in their own file so the
// control-plane files stay as they were.

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
