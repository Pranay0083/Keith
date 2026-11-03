package store

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/events"
)

// BranchPrefix starts every branch the agent pushes; a session's default
// branch is BranchPrefix + session ID.
const BranchPrefix = "agent/"

// SessionBranch returns the default branch name for a session.
func SessionBranch(sessionID string) string { return BranchPrefix + sessionID }

// Session is one task.
type Session struct {
	ID    string
	OrgID string
	// CreatedBy is "" for sessions no person started.
	CreatedBy string
	Status    events.SessionStatus
	// StatusReason explains failed, cancelled and no_changes.
	StatusReason string
	Prompt       string
	// Provider and Model are pinned for the session's life.
	Provider        string
	Model           string
	SandboxID       string
	SandboxProvider string
	RoutineID       string
	// Trigger is nil when the session was not started by a routine.
	Trigger           json.RawMessage
	CancelRequestedAt *time.Time
	CreatedAt         time.Time
	UpdatedAt         time.Time
	// FinishedAt is set while the status is terminal.
	FinishedAt *time.Time
	// Repos are the session's repos by full name.
	Repos []SessionRepo
}

// SessionRepo is one repo a session works on, with its branch and PR.
type SessionRepo struct {
	SessionID string
	RepoID    string
	// FullName, InstallationID and DefaultBranch come from the repo, so the
	// worker can mint a token and clone without another lookup.
	FullName       string
	InstallationID int64
	DefaultBranch  string
	Branch         string
	// PRNumber is 0 until a PR is opened.
	PRNumber int
	PRURL    string
}

// CreateSessionParams describes a new session.
type CreateSessionParams struct {
	OrgID     string
	CreatedBy string
	Prompt    string
	Provider  string
	Model     string
	// RepoIDs must be accessible repos of OrgID. Each gets the branch
	// SessionBranch(id).
	RepoIDs   []string
	RoutineID string
	Trigger   json.RawMessage
}

const sessionCols = `s.id, s.org_id, s.created_by, s.status, s.status_reason, s.prompt, s.provider, s.model,
	s.sandbox_id, s.sandbox_provider, s.routine_id, s.trigger, s.cancel_requested_at,
	s.created_at, s.updated_at, s.finished_at`

func scanSession(row interface{ Scan(...any) error }) (Session, error) {
	var x Session
	var createdBy, sandboxID, sandboxProvider, routineID *string
	var trigger []byte
	err := row.Scan(&x.ID, &x.OrgID, &createdBy, &x.Status, &x.StatusReason, &x.Prompt, &x.Provider, &x.Model,
		&sandboxID, &sandboxProvider, &routineID, &trigger, &x.CancelRequestedAt,
		&x.CreatedAt, &x.UpdatedAt, &x.FinishedAt)
	x.CreatedBy, x.SandboxID, x.SandboxProvider, x.RoutineID =
		deref(createdBy), deref(sandboxID), deref(sandboxProvider), deref(routineID)
	if trigger != nil {
		x.Trigger = json.RawMessage(trigger)
	}
	return x, err
}

// CreateSession creates a queued session and its session_repos rows in
// one transaction. A repo ID that is not an accessible repo of the org
// fails the whole call with ErrNotFound.
func (s *Store) CreateSession(ctx context.Context, p CreateSessionParams) (Session, error) {
	const op = "create session"
	if !validID(p.OrgID) || (p.CreatedBy != "" && !validID(p.CreatedBy)) ||
		(p.RoutineID != "" && !validID(p.RoutineID)) || !validIDs(p.RepoIDs) {
		return Session{}, errNotFound(op)
	}
	if p.Provider == "" || p.Model == "" {
		return Session{}, fmt.Errorf("store: %s: provider and model are required", op)
	}
	repoIDs := slices.Clone(p.RepoIDs)
	slices.Sort(repoIDs)
	repoIDs = slices.Compact(repoIDs)

	var trigger any
	if len(p.Trigger) > 0 {
		trigger = string(p.Trigger)
	}

	var out Session
	err := s.WithTx(ctx, func(tx *Store) error {
		var err error
		out, err = scanSession(tx.db.QueryRow(ctx, `
			INSERT INTO sessions AS s (org_id, created_by, prompt, provider, model, routine_id, trigger)
			VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
			RETURNING `+sessionCols,
			p.OrgID, nullIfEmpty(p.CreatedBy), p.Prompt, p.Provider, p.Model, nullIfEmpty(p.RoutineID), trigger))
		if pgCode(err) == "23503" {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		if len(repoIDs) == 0 {
			return nil
		}
		tag, err := tx.db.Exec(ctx, `
			INSERT INTO session_repos (session_id, repo_id, branch)
			SELECT $1, r.id, $2 FROM repos r
			WHERE r.id = ANY($3::uuid[]) AND r.org_id = $4 AND r.accessible`,
			out.ID, SessionBranch(out.ID), repoIDs, p.OrgID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != int64(len(repoIDs)) {
			return fmt.Errorf("%w: a repo is not an accessible repo of the org", ErrNotFound)
		}
		out.Repos, err = tx.ListSessionRepos(ctx, out.ID)
		return err
	})
	if err != nil {
		return Session{}, wrap(op, err)
	}
	return out, nil
}

// GetSession returns a session with its repos.
func (s *Store) GetSession(ctx context.Context, id string) (Session, error) {
	if !validID(id) {
		return Session{}, errNotFound("get session")
	}
	x, err := scanSession(s.db.QueryRow(ctx, `SELECT `+sessionCols+` FROM sessions s WHERE s.id = $1`, id))
	if err != nil {
		return Session{}, notFound("get session", err)
	}
	if x.Repos, err = s.ListSessionRepos(ctx, id); err != nil {
		return Session{}, err
	}
	return x, nil
}

// Session list paging limits.
const (
	DefaultSessionPageSize = 50
	MaxSessionPageSize     = 200
)

// ErrInvalidCursor is returned for a page cursor this package did not
// produce.
var ErrInvalidCursor = errors.New("store: invalid page cursor")

// ListSessionsParams selects a page of an org's sessions, newest first.
type ListSessionsParams struct {
	OrgID string
	// Limit is the page size; 0 means DefaultSessionPageSize, and it is
	// capped at MaxSessionPageSize.
	Limit int
	// Cursor is the previous page's NextCursor; "" starts at the newest.
	Cursor string
}

// SessionPage is one page of sessions.
type SessionPage struct {
	Sessions []Session
	// NextCursor fetches the next page; "" means this is the last.
	NextCursor string
}

// ListSessions returns a page of an org's sessions with their repos,
// newest first. Paging is by (created_at, id), so sessions created while
// paging do not shift later pages.
func (s *Store) ListSessions(ctx context.Context, p ListSessionsParams) (SessionPage, error) {
	const op = "list sessions"
	if !validID(p.OrgID) {
		return SessionPage{}, nil
	}
	limit := p.Limit
	if limit <= 0 {
		limit = DefaultSessionPageSize
	}
	limit = min(limit, MaxSessionPageSize)

	var beforeTS *time.Time
	var beforeID *string
	if p.Cursor != "" {
		ts, id, err := decodeCursor(p.Cursor)
		if err != nil {
			return SessionPage{}, err
		}
		beforeTS, beforeID = &ts, &id
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+sessionCols+` FROM sessions s
		WHERE s.org_id = $1
		  AND ($2::timestamptz IS NULL OR (s.created_at, s.id) < ($2::timestamptz, $3::uuid))
		ORDER BY s.created_at DESC, s.id DESC
		LIMIT $4`, p.OrgID, beforeTS, beforeID, limit+1)
	if err != nil {
		return SessionPage{}, wrap(op, err)
	}
	defer rows.Close()
	var page SessionPage
	for rows.Next() {
		x, err := scanSession(rows)
		if err != nil {
			return SessionPage{}, wrap(op, err)
		}
		page.Sessions = append(page.Sessions, x)
	}
	if err := rows.Err(); err != nil {
		return SessionPage{}, wrap(op, err)
	}
	if len(page.Sessions) > limit {
		page.Sessions = page.Sessions[:limit]
		last := page.Sessions[limit-1]
		page.NextCursor = encodeCursor(last.CreatedAt, last.ID)
	}
	if len(page.Sessions) == 0 {
		return page, nil
	}

	ids := make([]string, len(page.Sessions))
	for i, x := range page.Sessions {
		ids[i] = x.ID
	}
	repos, err := s.sessionRepos(ctx, ids)
	if err != nil {
		return SessionPage{}, err
	}
	for i := range page.Sessions {
		page.Sessions[i].Repos = repos[page.Sessions[i].ID]
	}
	return page, nil
}

func encodeCursor(ts time.Time, id string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(ts.UTC().Format(time.RFC3339Nano) + "|" + id))
}

func decodeCursor(c string) (time.Time, string, error) {
	b, err := base64.RawURLEncoding.DecodeString(c)
	if err != nil {
		return time.Time{}, "", ErrInvalidCursor
	}
	tsText, id, ok := strings.Cut(string(b), "|")
	if !ok || !validID(id) {
		return time.Time{}, "", ErrInvalidCursor
	}
	ts, err := time.Parse(time.RFC3339Nano, tsText)
	if err != nil {
		return time.Time{}, "", ErrInvalidCursor
	}
	return ts, id, nil
}

// SetSessionStatus sets a session's status and reason. A terminal status
// sets finished_at to now; any other status clears it.
func (s *Store) SetSessionStatus(ctx context.Context, id string, status events.SessionStatus, reason string) error {
	const op = "set session status"
	if !slices.Contains(events.Statuses, status) {
		return fmt.Errorf("store: %s: invalid status %q", op, status)
	}
	if !validID(id) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET
			status        = $2,
			status_reason = $3,
			finished_at   = CASE WHEN $4::bool THEN now() END,
			updated_at    = now()
		WHERE id = $1`, id, string(status), reason, status.Terminal())
	return affected(op, tag.RowsAffected(), err)
}

// SetSessionSandbox records the session's sandbox; "" clears it.
func (s *Store) SetSessionSandbox(ctx context.Context, id, sandboxID, provider string) error {
	const op = "set session sandbox"
	if !validID(id) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET sandbox_id = $2, sandbox_provider = $3, updated_at = now()
		WHERE id = $1`, id, nullIfEmpty(sandboxID), nullIfEmpty(provider))
	return affected(op, tag.RowsAffected(), err)
}

// RequestCancel flags the session for cancellation (Stop). The worker
// checks the flag between steps. Repeated calls keep the first time.
func (s *Store) RequestCancel(ctx context.Context, id string) error {
	const op = "request cancel"
	if !validID(id) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET cancel_requested_at = COALESCE(cancel_requested_at, now()), updated_at = now()
		WHERE id = $1`, id)
	return affected(op, tag.RowsAffected(), err)
}

// ClearCancelRequest clears the cancel flag, for a session resumed after
// a stop.
func (s *Store) ClearCancelRequest(ctx context.Context, id string) error {
	const op = "clear cancel request"
	if !validID(id) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET cancel_requested_at = NULL, updated_at = now()
		WHERE id = $1`, id)
	return affected(op, tag.RowsAffected(), err)
}

// IsCancelRequested reports whether Stop was requested for the session.
func (s *Store) IsCancelRequested(ctx context.Context, id string) (bool, error) {
	if !validID(id) {
		return false, errNotFound("is cancel requested")
	}
	var requested bool
	err := s.db.QueryRow(ctx,
		`SELECT cancel_requested_at IS NOT NULL FROM sessions WHERE id = $1`, id).Scan(&requested)
	return requested, notFound("is cancel requested", err)
}

// SetSessionModel changes a session's provider and model. It is allowed
// only before agent state exists: once the conversation has started the
// session is pinned, and this returns ErrConflict.
func (s *Store) SetSessionModel(ctx context.Context, id, provider, model string) error {
	const op = "set session model"
	if !validID(id) {
		return errNotFound(op)
	}
	if provider == "" || model == "" {
		return fmt.Errorf("store: %s: provider and model are required", op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET provider = $2, model = $3, updated_at = now()
		WHERE id = $1 AND NOT EXISTS (SELECT 1 FROM agent_state a WHERE a.session_id = $1)`,
		id, provider, model)
	if err != nil {
		return wrap(op, err)
	}
	if tag.RowsAffected() == 1 {
		return nil
	}
	if exists, err := s.sessionExists(ctx, id); err != nil {
		return wrap(op, err)
	} else if exists {
		return fmt.Errorf("store: %s: %w: the session already has agent state", op, ErrConflict)
	}
	return errNotFound(op)
}

func (s *Store) sessionExists(ctx context.Context, id string) (bool, error) {
	var exists bool
	err := s.db.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM sessions WHERE id = $1)`, id).Scan(&exists)
	return exists, err
}

const sessionRepoCols = `sr.session_id, sr.repo_id, r.full_name, r.installation_id, r.default_branch,
	sr.branch, sr.pr_number, sr.pr_url`

func scanSessionRepo(row interface{ Scan(...any) error }) (SessionRepo, error) {
	var x SessionRepo
	var prNumber *int32
	var prURL *string
	err := row.Scan(&x.SessionID, &x.RepoID, &x.FullName, &x.InstallationID, &x.DefaultBranch,
		&x.Branch, &prNumber, &prURL)
	if prNumber != nil {
		x.PRNumber = int(*prNumber)
	}
	x.PRURL = deref(prURL)
	return x, err
}

// ListSessionRepos returns a session's repos by full name.
func (s *Store) ListSessionRepos(ctx context.Context, sessionID string) ([]SessionRepo, error) {
	if !validID(sessionID) {
		return nil, nil
	}
	m, err := s.sessionRepos(ctx, []string{sessionID})
	return m[sessionID], err
}

func (s *Store) sessionRepos(ctx context.Context, sessionIDs []string) (map[string][]SessionRepo, error) {
	rows, err := s.db.Query(ctx, `
		SELECT `+sessionRepoCols+`
		FROM session_repos sr JOIN repos r ON r.id = sr.repo_id
		WHERE sr.session_id = ANY($1::uuid[])
		ORDER BY sr.session_id, r.full_name, r.id`, sessionIDs)
	if err != nil {
		return nil, wrap("list session repos", err)
	}
	defer rows.Close()
	out := make(map[string][]SessionRepo)
	for rows.Next() {
		x, err := scanSessionRepo(rows)
		if err != nil {
			return nil, wrap("list session repos", err)
		}
		out[x.SessionID] = append(out[x.SessionID], x)
	}
	return out, wrap("list session repos", rows.Err())
}

// SetSessionRepoBranch sets the branch the agent pushes for one repo of a
// session.
func (s *Store) SetSessionRepoBranch(ctx context.Context, sessionID, repoID, branch string) error {
	const op = "set session repo branch"
	if !validID(sessionID) || !validID(repoID) {
		return errNotFound(op)
	}
	if branch == "" {
		return fmt.Errorf("store: %s: empty branch", op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE session_repos SET branch = $3 WHERE session_id = $1 AND repo_id = $2`,
		sessionID, repoID, branch)
	return affected(op, tag.RowsAffected(), err)
}

// SetSessionRepoPR records the PR opened for one repo of a session.
func (s *Store) SetSessionRepoPR(ctx context.Context, sessionID, repoID string, number int, url string) error {
	const op = "set session repo pr"
	if !validID(sessionID) || !validID(repoID) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE session_repos SET pr_number = $3, pr_url = $4 WHERE session_id = $1 AND repo_id = $2`,
		sessionID, repoID, int32(number), url)
	return affected(op, tag.RowsAffected(), err)
}

// affected turns an UPDATE/DELETE result into nil, ErrNotFound (no rows)
// or the wrapped error.
func affected(op string, n int64, err error) error {
	if err != nil {
		return wrap(op, err)
	}
	if n == 0 {
		return errNotFound(op)
	}
	return nil
}
