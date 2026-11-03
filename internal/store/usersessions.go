package store

import (
	"context"
	"time"
)

// RevokeUserSessions revokes every session cookie issued to the user at
// or before at (logout). The cutoff only moves forward. An unknown user
// is ErrNotFound.
func (s *Store) RevokeUserSessions(ctx context.Context, userID string, at time.Time) error {
	const op = "revoke user sessions"
	if !validID(userID) {
		return errNotFound(op)
	}
	// GREATEST ignores a NULL cutoff.
	tag, err := s.db.Exec(ctx, `
		UPDATE users SET sessions_valid_after = GREATEST(sessions_valid_after, $2::timestamptz)
		WHERE id = $1`, userID, at)
	return affected(op, tag.RowsAffected(), err)
}
