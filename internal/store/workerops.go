package store

import (
	"context"
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
