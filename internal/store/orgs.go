package store

import (
	"context"
	"fmt"
	"time"
)

// Org is the workspace that installs the GitHub App and pays.
type Org struct {
	ID   string
	Name string
	// DefaultLLMProvider is "" for the service default.
	DefaultLLMProvider string
	CreatedAt          time.Time
	UpdatedAt          time.Time
}

// Role is an org member's role. Owners outrank admins, who outrank
// members.
type Role string

const (
	RoleOwner  Role = "owner"
	RoleAdmin  Role = "admin"
	RoleMember Role = "member"
)

func (r Role) rank() int {
	switch r {
	case RoleOwner:
		return 3
	case RoleAdmin:
		return 2
	case RoleMember:
		return 1
	}
	return 0
}

// Valid reports whether r is one of the defined roles.
func (r Role) Valid() bool { return r.rank() > 0 }

// AtLeast reports whether r grants everything min does.
func (r Role) AtLeast(min Role) bool { return r.Valid() && r.rank() >= min.rank() }

// OrgMembership is an org as seen by one member.
type OrgMembership struct {
	Org  Org
	Role Role
}

const orgCols = `o.id, o.name, o.default_llm_provider, o.created_at, o.updated_at`

func scanOrg(row interface{ Scan(...any) error }, extra ...any) (Org, error) {
	var o Org
	var provider *string
	err := row.Scan(append([]any{&o.ID, &o.Name, &provider, &o.CreatedAt, &o.UpdatedAt}, extra...)...)
	o.DefaultLLMProvider = deref(provider)
	return o, err
}

// CreateOrg creates an org with ownerID as its owner, in one transaction.
func (s *Store) CreateOrg(ctx context.Context, name, ownerID string) (Org, error) {
	if !validID(ownerID) {
		return Org{}, errNotFound("create org: owner")
	}
	var org Org
	err := s.WithTx(ctx, func(tx *Store) error {
		var err error
		org, err = scanOrg(tx.db.QueryRow(ctx,
			`INSERT INTO orgs AS o (name) VALUES ($1) RETURNING `+orgCols, name))
		if err != nil {
			return err
		}
		_, err = tx.db.Exec(ctx,
			`INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, 'owner')`, org.ID, ownerID)
		if pgCode(err) == "23503" { // foreign_key_violation: no such user
			return ErrNotFound
		}
		return err
	})
	return org, wrap("create org", err)
}

// GetOrg returns an org by ID.
func (s *Store) GetOrg(ctx context.Context, id string) (Org, error) {
	if !validID(id) {
		return Org{}, errNotFound("get org")
	}
	o, err := scanOrg(s.db.QueryRow(ctx, `SELECT `+orgCols+` FROM orgs o WHERE o.id = $1`, id))
	return o, notFound("get org", err)
}

// SetOrgDefaultLLMProvider sets the provider new sessions in the org use;
// "" means the service default.
func (s *Store) SetOrgDefaultLLMProvider(ctx context.Context, orgID, provider string) error {
	if !validID(orgID) {
		return errNotFound("set org provider")
	}
	tag, err := s.db.Exec(ctx,
		`UPDATE orgs SET default_llm_provider = $2, updated_at = now() WHERE id = $1`,
		orgID, nullIfEmpty(provider))
	if err != nil {
		return wrap("set org provider", err)
	}
	if tag.RowsAffected() == 0 {
		return errNotFound("set org provider")
	}
	return nil
}

// ListOrgsForUser returns the orgs userID belongs to, by name.
func (s *Store) ListOrgsForUser(ctx context.Context, userID string) ([]OrgMembership, error) {
	if !validID(userID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+orgCols+`, m.role
		FROM org_members m JOIN orgs o ON o.id = m.org_id
		WHERE m.user_id = $1
		ORDER BY o.name, o.id`, userID)
	if err != nil {
		return nil, wrap("list orgs", err)
	}
	defer rows.Close()
	var out []OrgMembership
	for rows.Next() {
		var m OrgMembership
		if m.Org, err = scanOrg(rows, &m.Role); err != nil {
			return nil, wrap("list orgs", err)
		}
		out = append(out, m)
	}
	return out, wrap("list orgs", rows.Err())
}

// AddOrgMember adds userID to the org with role, or changes the role of
// an existing member.
func (s *Store) AddOrgMember(ctx context.Context, orgID, userID string, role Role) error {
	if !role.Valid() {
		return fmt.Errorf("store: add org member: invalid role %q", role)
	}
	if !validID(orgID) || !validID(userID) {
		return errNotFound("add org member")
	}
	_, err := s.db.Exec(ctx, `
		INSERT INTO org_members (org_id, user_id, role) VALUES ($1, $2, $3)
		ON CONFLICT (org_id, user_id) DO UPDATE SET role = EXCLUDED.role`,
		orgID, userID, string(role))
	if pgCode(err) == "23503" {
		return errNotFound("add org member")
	}
	return wrap("add org member", err)
}

// RemoveOrgMember removes userID from the org.
func (s *Store) RemoveOrgMember(ctx context.Context, orgID, userID string) error {
	if !validID(orgID) || !validID(userID) {
		return errNotFound("remove org member")
	}
	tag, err := s.db.Exec(ctx, `DELETE FROM org_members WHERE org_id = $1 AND user_id = $2`, orgID, userID)
	if err != nil {
		return wrap("remove org member", err)
	}
	if tag.RowsAffected() == 0 {
		return errNotFound("remove org member")
	}
	return nil
}

// GetOrgRole returns userID's role in the org, or ErrNotFound if they are
// not a member.
func (s *Store) GetOrgRole(ctx context.Context, orgID, userID string) (Role, error) {
	if !validID(orgID) || !validID(userID) {
		return "", errNotFound("get org role")
	}
	var role Role
	err := s.db.QueryRow(ctx,
		`SELECT role FROM org_members WHERE org_id = $1 AND user_id = $2`, orgID, userID).Scan(&role)
	return role, notFound("get org role", err)
}

// HasOrgRole reports whether userID is a member of the org with at least
// role min. Non-members get false and no error.
func (s *Store) HasOrgRole(ctx context.Context, orgID, userID string, min Role) (bool, error) {
	role, err := s.GetOrgRole(ctx, orgID, userID)
	if isNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return role.AtLeast(min), nil
}
