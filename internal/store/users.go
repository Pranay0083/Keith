package store

import (
	"context"
	"time"
)

// User is a person signed in through GitHub OAuth.
type User struct {
	ID           string
	GitHubUserID int64
	GitHubLogin  string
	// Email is "" when GitHub does not expose one.
	Email     string
	Name      string
	AvatarURL string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// UpsertUserParams is the GitHub profile seen at sign-in.
type UpsertUserParams struct {
	GitHubUserID int64
	GitHubLogin  string
	Email        string
	Name         string
	AvatarURL    string
}

const userCols = `id, github_user_id, github_login, email, name, avatar_url, created_at, updated_at`

func scanUser(row interface{ Scan(...any) error }) (User, error) {
	var u User
	var email *string
	err := row.Scan(&u.ID, &u.GitHubUserID, &u.GitHubLogin, &email, &u.Name, &u.AvatarURL, &u.CreatedAt, &u.UpdatedAt)
	u.Email = deref(email)
	return u, err
}

// UpsertUser creates the user for a GitHub account, or refreshes its
// profile fields if it exists. It is keyed by github_user_id, which never
// changes, unlike the login.
func (s *Store) UpsertUser(ctx context.Context, p UpsertUserParams) (User, error) {
	row := s.db.QueryRow(ctx, `
		INSERT INTO users (github_user_id, github_login, email, name, avatar_url)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (github_user_id) DO UPDATE SET
			github_login = EXCLUDED.github_login,
			email        = EXCLUDED.email,
			name         = EXCLUDED.name,
			avatar_url   = EXCLUDED.avatar_url,
			updated_at   = now()
		RETURNING `+userCols,
		p.GitHubUserID, p.GitHubLogin, nullIfEmpty(p.Email), p.Name, p.AvatarURL)
	u, err := scanUser(row)
	return u, wrap("upsert user", err)
}

// GetUser returns a user by ID.
func (s *Store) GetUser(ctx context.Context, id string) (User, error) {
	if !validID(id) {
		return User{}, errNotFound("get user")
	}
	u, err := scanUser(s.db.QueryRow(ctx, `SELECT `+userCols+` FROM users WHERE id = $1`, id))
	return u, notFound("get user", err)
}

// GetUserByGitHubID returns the user for a GitHub account.
func (s *Store) GetUserByGitHubID(ctx context.Context, githubUserID int64) (User, error) {
	u, err := scanUser(s.db.QueryRow(ctx, `SELECT `+userCols+` FROM users WHERE github_user_id = $1`, githubUserID))
	return u, notFound("get user by github id", err)
}
