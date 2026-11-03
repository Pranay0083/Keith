// Package storetest gives integration tests a migrated scratch Postgres
// database and a few fixture rows.
//
// Tests that use it run only with KEITH_IT=1. DATABASE_URL names a server
// the tests may create databases on; it defaults to the docker-compose.yml
// database:
//
//	docker compose up -d --wait
//	KEITH_IT=1 go test ./...
package storetest

import (
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"math/rand/v2"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/Pranay0083/keith/internal/store"
)

// DefaultDatabaseURL is the docker-compose.yml database.
const DefaultDatabaseURL = "postgres://keith:keith@localhost:5432/keith?sslmode=disable"

// Enabled reports whether integration tests should run (KEITH_IT=1).
func Enabled() bool { return os.Getenv("KEITH_IT") == "1" }

// DatabaseURL returns DATABASE_URL, or DefaultDatabaseURL when unset.
func DatabaseURL() string {
	if u := os.Getenv("DATABASE_URL"); u != "" {
		return u
	}
	return DefaultDatabaseURL
}

// Skip skips t unless KEITH_IT=1.
func Skip(t testing.TB) {
	t.Helper()
	if !Enabled() {
		t.Skip("integration test: set KEITH_IT=1 (and DATABASE_URL/REDIS_URL, or run docker compose up)")
	}
}

// New creates an empty scratch database next to DatabaseURL, migrates it
// and returns a Store on it. The database is dropped when t ends. It skips
// t unless KEITH_IT=1.
func New(t testing.TB) *store.Store {
	t.Helper()
	Skip(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	dbURL := ScratchDatabase(t)
	s, err := store.Open(ctx, dbURL)
	if err != nil {
		t.Fatalf("open scratch database: %v", err)
	}
	t.Cleanup(s.Close)
	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return s
}

// ScratchDatabase creates an empty database next to DatabaseURL, drops it
// when t ends, and returns its URL. It does not migrate.
func ScratchDatabase(t testing.TB) string {
	t.Helper()
	Skip(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	base := DatabaseURL()
	admin, err := pgx.Connect(ctx, base)
	if err != nil {
		t.Fatalf("connect to DATABASE_URL: %v", err)
	}
	defer admin.Close(ctx)

	suffix := make([]byte, 6)
	_, _ = crand.Read(suffix)
	name := "keith_it_" + hex.EncodeToString(suffix)
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Fatalf("create scratch database: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		c, err := pgx.Connect(ctx, base)
		if err != nil {
			t.Logf("drop scratch database %s: %v", name, err)
			return
		}
		defer c.Close(ctx)
		if _, err := c.Exec(ctx, "DROP DATABASE IF EXISTS "+name+" WITH (FORCE)"); err != nil {
			t.Logf("drop scratch database %s: %v", name, err)
		}
	})

	u, err := url.Parse(base)
	if err != nil {
		t.Fatalf("DATABASE_URL is not a URL")
	}
	u.Path = "/" + name
	return u.String()
}

// Fixture is a user who owns an org with one installation, one repo and
// one queued session on that repo.
type Fixture struct {
	User         store.User
	Org          store.Org
	Installation store.Installation
	Repo         store.Repo
	Session      store.Session
}

// Seed inserts a Fixture with random GitHub IDs.
func Seed(t testing.TB, s *store.Store) Fixture {
	t.Helper()
	ctx := context.Background()
	var f Fixture
	var err error
	if f.User, err = s.UpsertUser(ctx, store.UpsertUserParams{
		GitHubUserID: rand.Int64N(1 << 40), GitHubLogin: "octocat", Name: "Octo Cat",
	}); err != nil {
		t.Fatalf("seed user: %v", err)
	}
	if f.Org, err = s.CreateOrg(ctx, "Acme", f.User.ID); err != nil {
		t.Fatalf("seed org: %v", err)
	}
	if f.Installation, err = s.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: f.Org.ID, InstallationID: rand.Int64N(1 << 40), AccountLogin: "acme",
	}); err != nil {
		t.Fatalf("seed installation: %v", err)
	}
	if f.Repo, err = s.UpsertRepo(ctx, store.UpsertRepoParams{
		InstallationID: f.Installation.InstallationID, GitHubRepoID: rand.Int64N(1 << 40),
		FullName: "acme/api", DefaultBranch: "main", Private: true,
	}); err != nil {
		t.Fatalf("seed repo: %v", err)
	}
	if f.Session, err = s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: f.Org.ID, CreatedBy: f.User.ID, Prompt: "fix the tests",
		Provider: "anthropic", Model: "claude-opus-5-5", RepoIDs: []string{f.Repo.ID},
	}); err != nil {
		t.Fatalf("seed session: %v", err)
	}
	return f
}
