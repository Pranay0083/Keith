package store

import (
	"context"
	"fmt"
	"regexp"
	"time"
)

// SecretMeta is a secret without its value, for listing.
type SecretMeta struct {
	ID    string
	OrgID string
	// RepoID is "" for an org secret.
	RepoID    string
	Name      string
	CreatedBy string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// SecretCiphertext is a secret's sealed value, for the worker to open.
type SecretCiphertext struct {
	OrgID      string
	RepoID     string
	Name       string
	Ciphertext []byte
}

// AAD returns SecretAAD for the secret's row.
func (c SecretCiphertext) AAD() []byte { return SecretAAD(c.OrgID, c.RepoID, c.Name) }

// SecretAAD is the additional authenticated data a secret's value is
// sealed with (secrets.Keyring.Seal/Open), binding the ciphertext to its
// org, repo ("" for an org secret) and name so it cannot be moved to
// another row.
func SecretAAD(orgID, repoID, name string) []byte {
	return []byte("keith/secret/v1\x00org=" + orgID + "\x00repo=" + repoID + "\x00name=" + name)
}

// PutSecretParams writes a secret. Ciphertext must come from
// secrets.Keyring.Seal with SecretAAD(OrgID, RepoID, Name).
type PutSecretParams struct {
	OrgID string
	// RepoID "" makes an org secret.
	RepoID     string
	Name       string
	Ciphertext []byte
	CreatedBy  string
}

var envName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// ValidSecretName reports whether name is a valid environment variable
// name, as the secrets table requires.
func ValidSecretName(name string) bool { return envName.MatchString(name) }

const secretMetaCols = `id, org_id, repo_id, name, created_by, created_at, updated_at`

func scanSecretMeta(row interface{ Scan(...any) error }) (SecretMeta, error) {
	var m SecretMeta
	var repoID, createdBy *string
	err := row.Scan(&m.ID, &m.OrgID, &repoID, &m.Name, &createdBy, &m.CreatedAt, &m.UpdatedAt)
	m.RepoID, m.CreatedBy = deref(repoID), deref(createdBy)
	return m, err
}

// PutSecret creates or replaces a secret. A repo secret's repo must
// belong to the org, or it returns ErrNotFound.
func (s *Store) PutSecret(ctx context.Context, p PutSecretParams) (SecretMeta, error) {
	const op = "put secret"
	if !ValidSecretName(p.Name) {
		return SecretMeta{}, fmt.Errorf("store: %s: %q is not a valid environment variable name", op, p.Name)
	}
	if len(p.Ciphertext) == 0 {
		return SecretMeta{}, fmt.Errorf("store: %s: empty ciphertext", op)
	}
	if !validID(p.OrgID) || (p.RepoID != "" && !validID(p.RepoID)) || (p.CreatedBy != "" && !validID(p.CreatedBy)) {
		return SecretMeta{}, errNotFound(op)
	}
	// The two partial unique indexes need matching ON CONFLICT targets.
	q := `
		INSERT INTO secrets (org_id, repo_id, name, ciphertext, created_by)
		SELECT o.id, NULL, $3, $4, $5 FROM orgs o WHERE o.id = $1 AND $2::uuid IS NULL
		ON CONFLICT (org_id, name) WHERE repo_id IS NULL DO UPDATE SET
			ciphertext = EXCLUDED.ciphertext, updated_at = now()
		RETURNING ` + secretMetaCols
	if p.RepoID != "" {
		q = `
		INSERT INTO secrets (org_id, repo_id, name, ciphertext, created_by)
		SELECT r.org_id, r.id, $3, $4, $5 FROM repos r WHERE r.org_id = $1 AND r.id = $2
		ON CONFLICT (org_id, repo_id, name) WHERE repo_id IS NOT NULL DO UPDATE SET
			ciphertext = EXCLUDED.ciphertext, updated_at = now()
		RETURNING ` + secretMetaCols
	}
	m, err := scanSecretMeta(s.db.QueryRow(ctx, q,
		p.OrgID, nullIfEmpty(p.RepoID), p.Name, p.Ciphertext, nullIfEmpty(p.CreatedBy)))
	if pgCode(err) == "23503" { // created_by is not a user
		return SecretMeta{}, errNotFound(op)
	}
	return m, notFound(op, err)
}

// ListSecretNames lists an org's secrets (repoID "") or one repo's
// secrets, by name, without values.
func (s *Store) ListSecretNames(ctx context.Context, orgID, repoID string) ([]SecretMeta, error) {
	const op = "list secrets"
	if !validID(orgID) || (repoID != "" && !validID(repoID)) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+secretMetaCols+` FROM secrets
		WHERE org_id = $1 AND repo_id IS NOT DISTINCT FROM $2::uuid
		ORDER BY name`, orgID, nullIfEmpty(repoID))
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []SecretMeta
	for rows.Next() {
		m, err := scanSecretMeta(rows)
		if err != nil {
			return nil, wrap(op, err)
		}
		out = append(out, m)
	}
	return out, wrap(op, rows.Err())
}

// DeleteSecret deletes an org secret (repoID "") or a repo secret.
func (s *Store) DeleteSecret(ctx context.Context, orgID, repoID, name string) error {
	const op = "delete secret"
	if !validID(orgID) || (repoID != "" && !validID(repoID)) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		DELETE FROM secrets
		WHERE org_id = $1 AND repo_id IS NOT DISTINCT FROM $2::uuid AND name = $3`,
		orgID, nullIfEmpty(repoID), name)
	return affected(op, tag.RowsAffected(), err)
}

// LoadSecretCiphertexts returns the sealed values a session in the org
// gets: every org secret plus the secrets of the given repos. Rows come
// org secrets first, then by repo, then by name, so applying them in
// order lets a repo secret override an org secret of the same name.
// Repos outside the org contribute nothing.
func (s *Store) LoadSecretCiphertexts(ctx context.Context, orgID string, repoIDs ...string) ([]SecretCiphertext, error) {
	const op = "load secrets"
	if !validID(orgID) || !validIDs(repoIDs) {
		return nil, errNotFound(op)
	}
	if repoIDs == nil {
		repoIDs = []string{}
	}
	rows, err := s.db.Query(ctx, `
		SELECT org_id, repo_id, name, ciphertext FROM secrets
		WHERE org_id = $1 AND (repo_id IS NULL OR repo_id = ANY($2::uuid[]))
		ORDER BY repo_id NULLS FIRST, name`, orgID, repoIDs)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []SecretCiphertext
	for rows.Next() {
		var c SecretCiphertext
		var repoID *string
		if err := rows.Scan(&c.OrgID, &repoID, &c.Name, &c.Ciphertext); err != nil {
			return nil, wrap(op, err)
		}
		c.RepoID = deref(repoID)
		out = append(out, c)
	}
	return out, wrap(op, rows.Err())
}
