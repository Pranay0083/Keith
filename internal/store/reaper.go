package store

import (
	"context"
	"time"

	"github.com/Pranay0083/keith/internal/events"
)

// Queries for the worker's sandbox reaper.

// ParkedStatuses are the statuses whose sandbox is parked (paused, or
// about to be): every terminal status plus waiting_for_user.
var ParkedStatuses = []events.SessionStatus{
	events.StatusWaitingForUser, events.StatusCompleted, events.StatusFailed,
	events.StatusCancelled, events.StatusNoChanges,
}

// ListIdleSandboxSessions returns sessions that still have a sandbox
// while in a ParkedStatus, and were last updated before idleSince, oldest
// first, at most limit of them (limit <= 0 means 100). The reaper deletes
// their sandboxes.
func (s *Store) ListIdleSandboxSessions(ctx context.Context, idleSince time.Time, limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 100
	}
	return s.listSessions(ctx, "list idle sandbox sessions", `
		SELECT `+sessionCols+` FROM sessions s
		WHERE s.sandbox_id IS NOT NULL AND s.status = ANY($1::text[]) AND s.updated_at < $2
		ORDER BY s.updated_at, s.id
		LIMIT $3`, statusStrings(ParkedStatuses), idleSince, limit)
}

// SandboxIDsInUse reports which of the given sandbox IDs some session
// still records (sessions.sandbox_id), for finding orphaned sandboxes.
func (s *Store) SandboxIDsInUse(ctx context.Context, ids []string) (map[string]bool, error) {
	out := make(map[string]bool)
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT DISTINCT sandbox_id FROM sessions WHERE sandbox_id = ANY($1::text[])`, ids)
	if err != nil {
		return nil, wrap("sandbox ids in use", err)
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, wrap("sandbox ids in use", err)
		}
		out[id] = true
	}
	return out, wrap("sandbox ids in use", rows.Err())
}
