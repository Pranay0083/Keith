package store

import (
	"context"
	"strings"
	"time"
	"unicode/utf8"
)

// MaxDiffBytes caps a saved diff (GET /api/sessions/{id}/diff serves at
// most 1 MiB per repo). SetSessionRepoDiff cuts longer diffs.
const MaxDiffBytes = 1 << 20

// SessionDiff is the latest diff the worker saved for one session repo.
type SessionDiff struct {
	RepoID   string
	FullName string
	// Diff is "" before any change.
	Diff      string
	Truncated bool
	// UpdatedAt is nil before any change.
	UpdatedAt *time.Time
}

// SetSessionRepoDiff saves the latest unified diff of one session repo,
// against the repo's default branch at clone time. The worker calls it
// after every file-changing tool call, before it emits file.changed, with
// secret values already redacted.
//
// A diff over MaxDiffBytes is cut at a UTF-8 boundary and saved with
// truncated = true. Invalid UTF-8 and NUL bytes, which Postgres text
// cannot hold, are replaced with U+FFFD.
func (s *Store) SetSessionRepoDiff(ctx context.Context, sessionID, repoID, diff string, truncated bool) error {
	const op = "set session repo diff"
	if !validID(sessionID) || !validID(repoID) {
		return errNotFound(op)
	}
	diff, cut := cleanDiff(diff)
	tag, err := s.db.Exec(ctx, `
		UPDATE session_repos SET last_diff = $3, last_diff_truncated = $4, last_diff_at = now()
		WHERE session_id = $1 AND repo_id = $2`,
		sessionID, repoID, diff, truncated || cut)
	return affected(op, tag.RowsAffected(), err)
}

// cleanDiff makes diff storable and at most MaxDiffBytes, and reports
// whether it was cut.
func cleanDiff(diff string) (string, bool) {
	if !utf8.ValidString(diff) {
		diff = strings.ToValidUTF8(diff, "�")
	}
	if strings.IndexByte(diff, 0) >= 0 {
		diff = strings.ReplaceAll(diff, "\x00", "�")
	}
	if len(diff) <= MaxDiffBytes {
		return diff, false
	}
	n := MaxDiffBytes
	for n > 0 && !utf8.RuneStart(diff[n]) {
		n--
	}
	return diff[:n], true
}

// ListSessionDiffs returns one entry per session repo, in session order.
func (s *Store) ListSessionDiffs(ctx context.Context, sessionID string) ([]SessionDiff, error) {
	const op = "list session diffs"
	if !validID(sessionID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT sr.repo_id, r.full_name, sr.last_diff, sr.last_diff_truncated, sr.last_diff_at
		FROM session_repos sr JOIN repos r ON r.id = sr.repo_id
		WHERE sr.session_id = $1
		ORDER BY sr.position, r.full_name, r.id`, sessionID)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []SessionDiff
	for rows.Next() {
		var d SessionDiff
		if err := rows.Scan(&d.RepoID, &d.FullName, &d.Diff, &d.Truncated, &d.UpdatedAt); err != nil {
			return nil, wrap(op, err)
		}
		out = append(out, d)
	}
	return out, wrap(op, rows.Err())
}
