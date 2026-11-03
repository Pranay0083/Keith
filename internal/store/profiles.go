package store

import (
	"context"
	"time"
)

// RepoProfile is a repo's onboarding answers plus its sandbox snapshot.
type RepoProfile struct {
	RepoID     string
	InstallCmd string
	BuildCmd   string
	TestCmd    string
	RunCmd     string
	// Knowledge is free-form notes for the agent's prompt.
	Knowledge string
	// Snapshot is the zero Snapshot when none has been taken.
	Snapshot  Snapshot
	UpdatedAt time.Time
}

// Snapshot identifies a sandbox snapshot taken after a repo's first
// successful setup. IDs are provider-specific; LockHash decides whether a
// session started from it must re-run the install command.
type Snapshot struct {
	ID       string
	Provider string
	LockHash string
}

// UpsertRepoProfileParams are the onboarding answers.
type UpsertRepoProfileParams struct {
	RepoID     string
	InstallCmd string
	BuildCmd   string
	TestCmd    string
	RunCmd     string
	Knowledge  string
}

const profileCols = `repo_id, install_cmd, build_cmd, test_cmd, run_cmd, knowledge,
	snapshot_id, snapshot_provider, snapshot_lock_hash, updated_at`

func scanProfile(row interface{ Scan(...any) error }) (RepoProfile, error) {
	var p RepoProfile
	var id, provider, hash *string
	err := row.Scan(&p.RepoID, &p.InstallCmd, &p.BuildCmd, &p.TestCmd, &p.RunCmd, &p.Knowledge,
		&id, &provider, &hash, &p.UpdatedAt)
	p.Snapshot = Snapshot{ID: deref(id), Provider: deref(provider), LockHash: deref(hash)}
	return p, err
}

// GetRepoProfile returns a repo's profile, or ErrNotFound before
// onboarding.
func (s *Store) GetRepoProfile(ctx context.Context, repoID string) (RepoProfile, error) {
	if !validID(repoID) {
		return RepoProfile{}, errNotFound("get repo profile")
	}
	p, err := scanProfile(s.db.QueryRow(ctx, `SELECT `+profileCols+` FROM repo_profiles WHERE repo_id = $1`, repoID))
	return p, notFound("get repo profile", err)
}

// UpsertRepoProfile saves the onboarding answers. Changing the install
// command clears the snapshot, since its dependencies were installed the
// old way.
func (s *Store) UpsertRepoProfile(ctx context.Context, p UpsertRepoProfileParams) (RepoProfile, error) {
	if !validID(p.RepoID) {
		return RepoProfile{}, errNotFound("upsert repo profile: repo")
	}
	prof, err := scanProfile(s.db.QueryRow(ctx, `
		INSERT INTO repo_profiles AS p (repo_id, install_cmd, build_cmd, test_cmd, run_cmd, knowledge)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (repo_id) DO UPDATE SET
			install_cmd = EXCLUDED.install_cmd,
			build_cmd   = EXCLUDED.build_cmd,
			test_cmd    = EXCLUDED.test_cmd,
			run_cmd     = EXCLUDED.run_cmd,
			knowledge   = EXCLUDED.knowledge,
			snapshot_id        = CASE WHEN p.install_cmd = EXCLUDED.install_cmd THEN p.snapshot_id END,
			snapshot_provider  = CASE WHEN p.install_cmd = EXCLUDED.install_cmd THEN p.snapshot_provider END,
			snapshot_lock_hash = CASE WHEN p.install_cmd = EXCLUDED.install_cmd THEN p.snapshot_lock_hash END,
			updated_at  = now()
		RETURNING `+profileCols,
		p.RepoID, p.InstallCmd, p.BuildCmd, p.TestCmd, p.RunCmd, p.Knowledge))
	if pgCode(err) == "23503" {
		return RepoProfile{}, errNotFound("upsert repo profile: repo")
	}
	return prof, wrap("upsert repo profile", err)
}

// SetRepoSnapshot records a repo's snapshot, creating an empty profile if
// there is none. The zero Snapshot clears it.
func (s *Store) SetRepoSnapshot(ctx context.Context, repoID string, snap Snapshot) error {
	if !validID(repoID) {
		return errNotFound("set repo snapshot: repo")
	}
	_, err := s.db.Exec(ctx, `
		INSERT INTO repo_profiles (repo_id, snapshot_id, snapshot_provider, snapshot_lock_hash)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (repo_id) DO UPDATE SET
			snapshot_id        = EXCLUDED.snapshot_id,
			snapshot_provider  = EXCLUDED.snapshot_provider,
			snapshot_lock_hash = EXCLUDED.snapshot_lock_hash,
			updated_at         = now()`,
		repoID, nullIfEmpty(snap.ID), nullIfEmpty(snap.Provider), nullIfEmpty(snap.LockHash))
	if pgCode(err) == "23503" {
		return errNotFound("set repo snapshot: repo")
	}
	return wrap("set repo snapshot", err)
}
