package store

import (
	"context"

	"github.com/Pranay0083/keith/internal/events"
)

// Lookups the API, the webhook processor and the worker share.

// ActiveStatuses are the statuses that hold, or are about to hold, a
// sandbox: what the per-org cap counts. waiting_for_user is not in it,
// since its sandbox is parked.
var ActiveStatuses = []events.SessionStatus{
	events.StatusQueued, events.StatusProvisioning, events.StatusSettingUp, events.StatusRunning,
}

// LiveStatuses are every non-terminal status: ActiveStatuses plus
// waiting_for_user.
var LiveStatuses = append(append([]events.SessionStatus{}, ActiveStatuses...), events.StatusWaitingForUser)

func statusStrings(ss []events.SessionStatus) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = string(s)
	}
	return out
}

// GetSessionForUpdate returns a session (without its repos) and locks its
// row until the transaction ends, so status changes wait for the caller.
// Use it inside WithTx.
func (s *Store) GetSessionForUpdate(ctx context.Context, id string) (Session, error) {
	if !validID(id) {
		return Session{}, errNotFound("get session for update")
	}
	x, err := scanSession(s.db.QueryRow(ctx, `SELECT `+sessionCols+` FROM sessions s WHERE s.id = $1 FOR UPDATE`, id))
	return x, notFound("get session for update", err)
}

// LockOrg locks an org's row until the transaction ends, serialising
// session creation in the org so the active-session cap holds under
// concurrency. Use it inside WithTx.
func (s *Store) LockOrg(ctx context.Context, orgID string) error {
	if !validID(orgID) {
		return errNotFound("lock org")
	}
	var one int
	err := s.db.QueryRow(ctx, `SELECT 1 FROM orgs WHERE id = $1 FOR UPDATE`, orgID).Scan(&one)
	return notFound("lock org", err)
}

// CountActiveSessions counts an org's sessions in ActiveStatuses.
func (s *Store) CountActiveSessions(ctx context.Context, orgID string) (int, error) {
	if !validID(orgID) {
		return 0, nil
	}
	var n int
	err := s.db.QueryRow(ctx, `
		SELECT count(*) FROM sessions WHERE org_id = $1 AND status = ANY($2::text[])`,
		orgID, statusStrings(ActiveStatuses)).Scan(&n)
	return n, wrap("count active sessions", err)
}

// ListLiveSessionsByInstallation returns the sessions in LiveStatuses
// that work on any repo of a GitHub installation, oldest first, without
// their repos. The installation.deleted webhook stops them.
func (s *Store) ListLiveSessionsByInstallation(ctx context.Context, installationID int64) ([]Session, error) {
	return s.listSessions(ctx, "list live sessions by installation", `
		SELECT `+sessionCols+` FROM sessions s
		WHERE s.status = ANY($2::text[]) AND EXISTS (
			SELECT 1 FROM session_repos sr JOIN repos r ON r.id = sr.repo_id
			WHERE sr.session_id = s.id AND r.installation_id = $1)
		ORDER BY s.created_at, s.id`, installationID, statusStrings(LiveStatuses))
}

// ListLiveSessionsByRepos returns the sessions in LiveStatuses that work
// on any of the given repos, oldest first, without their repos.
func (s *Store) ListLiveSessionsByRepos(ctx context.Context, repoIDs []string) ([]Session, error) {
	if len(repoIDs) == 0 || !validIDs(repoIDs) {
		return nil, nil
	}
	return s.listSessions(ctx, "list live sessions by repos", `
		SELECT `+sessionCols+` FROM sessions s
		WHERE s.status = ANY($2::text[]) AND EXISTS (
			SELECT 1 FROM session_repos sr WHERE sr.session_id = s.id AND sr.repo_id = ANY($1::uuid[]))
		ORDER BY s.created_at, s.id`, repoIDs, statusStrings(LiveStatuses))
}

func (s *Store) listSessions(ctx context.Context, op, q string, args ...any) ([]Session, error) {
	rows, err := s.db.Query(ctx, q, args...)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []Session
	for rows.Next() {
		x, err := scanSession(rows)
		if err != nil {
			return nil, wrap(op, err)
		}
		out = append(out, x)
	}
	return out, wrap(op, rows.Err())
}

// GetRepoByFullName returns an org's repo by owner/name, compared
// case-insensitively as GitHub does. When several rows match (a repo
// renamed onto an old name), the accessible, most recently updated one
// wins.
func (s *Store) GetRepoByFullName(ctx context.Context, orgID, fullName string) (Repo, error) {
	if !validID(orgID) {
		return Repo{}, errNotFound("get repo by name")
	}
	r, err := scanRepo(s.db.QueryRow(ctx, `
		SELECT `+repoCols+` FROM repos r
		WHERE r.org_id = $1 AND lower(r.full_name) = lower($2)
		ORDER BY r.accessible DESC, r.updated_at DESC, r.id
		LIMIT 1`, orgID, fullName))
	return r, notFound("get repo by name", err)
}

// SetInstallationReposInaccessibleExcept marks every accessible repo of
// an installation inaccessible except the listed GitHub repo IDs (the
// ones a fresh listing still shows), and returns how many changed.
func (s *Store) SetInstallationReposInaccessibleExcept(ctx context.Context, installationID int64, keepGitHubRepoIDs []int64) (int64, error) {
	if keepGitHubRepoIDs == nil {
		keepGitHubRepoIDs = []int64{}
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE repos SET accessible = false, updated_at = now()
		WHERE installation_id = $1 AND accessible AND NOT (github_repo_id = ANY($2::bigint[]))`,
		installationID, keepGitHubRepoIDs)
	if err != nil {
		return 0, wrap("set installation repos inaccessible", err)
	}
	return tag.RowsAffected(), nil
}

// GetUsers returns the users with the given IDs, keyed by ID. Unknown and
// malformed IDs are left out.
func (s *Store) GetUsers(ctx context.Context, ids []string) (map[string]User, error) {
	out := make(map[string]User)
	valid := make([]string, 0, len(ids))
	for _, id := range ids {
		if validID(id) {
			valid = append(valid, id)
		}
	}
	if len(valid) == 0 {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `SELECT `+userCols+` FROM users WHERE id = ANY($1::uuid[])`, valid)
	if err != nil {
		return nil, wrap("get users", err)
	}
	defer rows.Close()
	for rows.Next() {
		u, err := scanUser(rows)
		if err != nil {
			return nil, wrap("get users", err)
		}
		out[u.ID] = u
	}
	return out, wrap("get users", rows.Err())
}

// OrgSecret is a secret's metadata with its repo's full name.
type OrgSecret struct {
	SecretMeta
	// RepoFullName is "" for an org secret.
	RepoFullName string
}

// ListOrgSecrets lists every secret of an org, org and repo secrets
// alike, without values: by name, then org secrets before repo secrets,
// then by repo full name.
func (s *Store) ListOrgSecrets(ctx context.Context, orgID string) ([]OrgSecret, error) {
	const op = "list org secrets"
	if !validID(orgID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT s.id, s.org_id, s.repo_id, s.name, s.created_by, s.created_at, s.updated_at, r.full_name
		FROM secrets s LEFT JOIN repos r ON r.id = s.repo_id
		WHERE s.org_id = $1
		ORDER BY s.name, s.repo_id IS NOT NULL, r.full_name, s.id`, orgID)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []OrgSecret
	for rows.Next() {
		var x OrgSecret
		var repoID, createdBy, fullName *string
		if err := rows.Scan(&x.ID, &x.OrgID, &repoID, &x.Name, &createdBy, &x.CreatedAt, &x.UpdatedAt, &fullName); err != nil {
			return nil, wrap(op, err)
		}
		x.RepoID, x.CreatedBy, x.RepoFullName = deref(repoID), deref(createdBy), deref(fullName)
		out = append(out, x)
	}
	return out, wrap(op, rows.Err())
}

// ListRepoProfiles returns the saved profiles of an org's repos, keyed by
// repo ID. Repos without a profile are absent.
func (s *Store) ListRepoProfiles(ctx context.Context, orgID string) (map[string]RepoProfile, error) {
	const op = "list repo profiles"
	out := make(map[string]RepoProfile)
	if !validID(orgID) {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT p.repo_id, p.install_cmd, p.build_cmd, p.test_cmd, p.run_cmd, p.knowledge,
			p.snapshot_id, p.snapshot_provider, p.snapshot_lock_hash, p.updated_at
		FROM repo_profiles p JOIN repos r ON r.id = p.repo_id
		WHERE r.org_id = $1`, orgID)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	for rows.Next() {
		p, err := scanProfile(rows)
		if err != nil {
			return nil, wrap(op, err)
		}
		out[p.RepoID] = p
	}
	return out, wrap(op, rows.Err())
}

// EnsurePersonalOrg gives a user who belongs to no org a new org named
// name, with them as owner, and reports whether it created one. The
// user's row is locked meanwhile, so two concurrent sign-ins create one
// org. Sign-in calls it, so every signed-in user has an org.
func (s *Store) EnsurePersonalOrg(ctx context.Context, userID, name string) (created bool, err error) {
	const op = "ensure personal org"
	if !validID(userID) {
		return false, errNotFound(op)
	}
	err = s.WithTx(ctx, func(tx *Store) error {
		var member bool
		err := tx.db.QueryRow(ctx, `
			SELECT EXISTS (SELECT 1 FROM org_members WHERE user_id = u.id)
			FROM users u WHERE u.id = $1 FOR UPDATE OF u`, userID).Scan(&member)
		if isNoRows(err) {
			return ErrNotFound
		}
		if err != nil || member {
			return err
		}
		if _, err := tx.CreateOrg(ctx, name, userID); err != nil {
			return err
		}
		created = true
		return nil
	})
	if err != nil {
		return false, wrap(op, err)
	}
	return created, nil
}
