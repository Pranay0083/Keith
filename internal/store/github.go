package store

import (
	"context"
	"fmt"
	"time"
)

// InstallationStatus is a GitHub App installation's state.
type InstallationStatus string

const (
	InstallationActive    InstallationStatus = "active"
	InstallationSuspended InstallationStatus = "suspended"
	InstallationDeleted   InstallationStatus = "deleted"
)

// Valid reports whether s is a defined status.
func (s InstallationStatus) Valid() bool {
	switch s {
	case InstallationActive, InstallationSuspended, InstallationDeleted:
		return true
	}
	return false
}

// Installation is one GitHub App installation, linked to one org.
type Installation struct {
	ID             string
	OrgID          string
	InstallationID int64
	AccountLogin   string
	Status         InstallationStatus
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

// UpsertInstallationParams describes an installation; Status "" means
// active.
type UpsertInstallationParams struct {
	OrgID          string
	InstallationID int64
	AccountLogin   string
	Status         InstallationStatus
}

const installationCols = `id, org_id, installation_id, account_login, status, created_at, updated_at`

func scanInstallation(row interface{ Scan(...any) error }) (Installation, error) {
	var i Installation
	err := row.Scan(&i.ID, &i.OrgID, &i.InstallationID, &i.AccountLogin, &i.Status, &i.CreatedAt, &i.UpdatedAt)
	return i, err
}

// UpsertInstallation records an installation for an org, or refreshes its
// account login and status. An installation stays with the org that
// first claimed it: upserting it for another org returns ErrConflict.
func (s *Store) UpsertInstallation(ctx context.Context, p UpsertInstallationParams) (Installation, error) {
	if p.Status == "" {
		p.Status = InstallationActive
	}
	if !p.Status.Valid() {
		return Installation{}, fmt.Errorf("store: upsert installation: invalid status %q", p.Status)
	}
	if !validID(p.OrgID) {
		return Installation{}, errNotFound("upsert installation: org")
	}
	inst, err := scanInstallation(s.db.QueryRow(ctx, `
		INSERT INTO github_installations (org_id, installation_id, account_login, status)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (installation_id) DO UPDATE SET
			account_login = EXCLUDED.account_login,
			status        = EXCLUDED.status,
			updated_at    = now()
		WHERE github_installations.org_id = EXCLUDED.org_id
		RETURNING `+installationCols,
		p.OrgID, p.InstallationID, p.AccountLogin, string(p.Status)))
	if isNoRows(err) { // the WHERE rejected the update: another org owns it
		return Installation{}, fmt.Errorf("store: upsert installation %d: %w: linked to another org", p.InstallationID, ErrConflict)
	}
	if pgCode(err) == "23503" {
		return Installation{}, errNotFound("upsert installation: org")
	}
	return inst, wrap("upsert installation", err)
}

// SetInstallationStatus updates an installation's status, from the
// installation webhooks.
func (s *Store) SetInstallationStatus(ctx context.Context, installationID int64, status InstallationStatus) error {
	if !status.Valid() {
		return fmt.Errorf("store: set installation status: invalid status %q", status)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE github_installations SET status = $2, updated_at = now()
		WHERE installation_id = $1`, installationID, string(status))
	if err != nil {
		return wrap("set installation status", err)
	}
	if tag.RowsAffected() == 0 {
		return errNotFound("set installation status")
	}
	return nil
}

// GetInstallation returns the installation with GitHub's installation ID.
func (s *Store) GetInstallation(ctx context.Context, installationID int64) (Installation, error) {
	inst, err := scanInstallation(s.db.QueryRow(ctx,
		`SELECT `+installationCols+` FROM github_installations WHERE installation_id = $1`, installationID))
	return inst, notFound("get installation", err)
}

// ListInstallations returns an org's installations, oldest first.
func (s *Store) ListInstallations(ctx context.Context, orgID string) ([]Installation, error) {
	if !validID(orgID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+installationCols+` FROM github_installations
		WHERE org_id = $1 ORDER BY created_at, id`, orgID)
	if err != nil {
		return nil, wrap("list installations", err)
	}
	defer rows.Close()
	var out []Installation
	for rows.Next() {
		inst, err := scanInstallation(rows)
		if err != nil {
			return nil, wrap("list installations", err)
		}
		out = append(out, inst)
	}
	return out, wrap("list installations", rows.Err())
}

// Repo is a GitHub repo the App can see (or could, when Accessible is
// false).
type Repo struct {
	ID             string
	OrgID          string
	InstallationID int64
	GitHubRepoID   int64
	// FullName is owner/name.
	FullName      string
	DefaultBranch string
	Private       bool
	Accessible    bool
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// UpsertRepoParams describes a repo as an installation sees it.
// DefaultBranch "" means main.
type UpsertRepoParams struct {
	InstallationID int64
	GitHubRepoID   int64
	FullName       string
	DefaultBranch  string
	Private        bool
}

const repoCols = `r.id, r.org_id, r.installation_id, r.github_repo_id, r.full_name, r.default_branch,
	r.private, r.accessible, r.created_at, r.updated_at`

func scanRepo(row interface{ Scan(...any) error }) (Repo, error) {
	var r Repo
	err := row.Scan(&r.ID, &r.OrgID, &r.InstallationID, &r.GitHubRepoID, &r.FullName, &r.DefaultBranch,
		&r.Private, &r.Accessible, &r.CreatedAt, &r.UpdatedAt)
	return r, err
}

// UpsertRepo records a repo under its installation, taking the org from
// the installation, and marks it accessible. A repo that moved to another
// installation (a transfer) moves with it. The installation must exist.
func (s *Store) UpsertRepo(ctx context.Context, p UpsertRepoParams) (Repo, error) {
	if p.DefaultBranch == "" {
		p.DefaultBranch = "main"
	}
	r, err := scanRepo(s.db.QueryRow(ctx, `
		INSERT INTO repos AS r (org_id, installation_id, github_repo_id, full_name, default_branch, private)
		SELECT i.org_id, i.installation_id, $2, $3, $4, $5
		FROM github_installations i WHERE i.installation_id = $1
		ON CONFLICT (github_repo_id) DO UPDATE SET
			org_id          = EXCLUDED.org_id,
			installation_id = EXCLUDED.installation_id,
			full_name       = EXCLUDED.full_name,
			default_branch  = EXCLUDED.default_branch,
			private         = EXCLUDED.private,
			accessible      = true,
			updated_at      = now()
		RETURNING `+repoCols,
		p.InstallationID, p.GitHubRepoID, p.FullName, p.DefaultBranch, p.Private))
	return r, notFound("upsert repo: installation", err)
}

// GetRepo returns a repo by ID.
func (s *Store) GetRepo(ctx context.Context, id string) (Repo, error) {
	if !validID(id) {
		return Repo{}, errNotFound("get repo")
	}
	r, err := scanRepo(s.db.QueryRow(ctx, `SELECT `+repoCols+` FROM repos r WHERE r.id = $1`, id))
	return r, notFound("get repo", err)
}

// ListRepos returns an org's repos by full name. Inaccessible repos are
// left out unless includeInaccessible is set.
func (s *Store) ListRepos(ctx context.Context, orgID string, includeInaccessible bool) ([]Repo, error) {
	if !validID(orgID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+repoCols+` FROM repos r
		WHERE r.org_id = $1 AND (r.accessible OR $2)
		ORDER BY r.full_name, r.id`, orgID, includeInaccessible)
	if err != nil {
		return nil, wrap("list repos", err)
	}
	defer rows.Close()
	var out []Repo
	for rows.Next() {
		r, err := scanRepo(rows)
		if err != nil {
			return nil, wrap("list repos", err)
		}
		out = append(out, r)
	}
	return out, wrap("list repos", rows.Err())
}

// SetReposInaccessible marks the given repos of an installation
// inaccessible (they were removed from it) and returns how many changed.
func (s *Store) SetReposInaccessible(ctx context.Context, installationID int64, githubRepoIDs []int64) (int64, error) {
	if len(githubRepoIDs) == 0 {
		return 0, nil
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE repos SET accessible = false, updated_at = now()
		WHERE installation_id = $1 AND github_repo_id = ANY($2) AND accessible`,
		installationID, githubRepoIDs)
	if err != nil {
		return 0, wrap("set repos inaccessible", err)
	}
	return tag.RowsAffected(), nil
}

// SetInstallationReposInaccessible marks every repo of an installation
// inaccessible (the App was uninstalled) and returns how many changed.
func (s *Store) SetInstallationReposInaccessible(ctx context.Context, installationID int64) (int64, error) {
	tag, err := s.db.Exec(ctx, `
		UPDATE repos SET accessible = false, updated_at = now()
		WHERE installation_id = $1 AND accessible`, installationID)
	if err != nil {
		return 0, wrap("set installation repos inaccessible", err)
	}
	return tag.RowsAffected(), nil
}
