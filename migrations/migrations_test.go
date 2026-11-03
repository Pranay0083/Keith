package migrations

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"io/fs"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

var v1Tables = []string{
	"agent_state", "events", "github_installations", "org_members", "orgs", "repo_profiles",
	"repos", "secrets", "session_repos", "sessions", "users",
}

// allTables is every table after the latest migration: v1Tables plus
// session_inputs (00003).
var allTables = []string{
	"agent_state", "events", "github_installations", "org_members", "orgs", "repo_profiles",
	"repos", "secrets", "session_inputs", "session_repos", "sessions", "users",
}

func TestV1SchemaDeclaresElevenTables(t *testing.T) {
	b, err := fs.ReadFile(FS, "00001_v1_schema.sql")
	if err != nil {
		t.Fatal(err)
	}
	sql := string(b)
	if !strings.Contains(sql, "-- +goose Up") || !strings.Contains(sql, "-- +goose Down") {
		t.Fatal("missing goose Up/Down annotations")
	}
	var got []string
	for _, m := range regexp.MustCompile(`(?m)^CREATE TABLE (\w+)`).FindAllStringSubmatch(sql, -1) {
		got = append(got, m[1])
	}
	sort.Strings(got)
	if strings.Join(got, ",") != strings.Join(v1Tables, ",") {
		t.Errorf("tables = %v, want %v", got, v1Tables)
	}
}

// TestMigrationsApply runs every migration up, down and up again against
// a scratch database created next to DATABASE_URL (which must allow
// CREATE DATABASE). It runs only with KEITH_IT=1, for example:
//
//	KEITH_IT=1 DATABASE_URL=postgres://keith:keith@localhost:5432/keith?sslmode=disable go test ./migrations
func TestMigrationsApply(t *testing.T) {
	if os.Getenv("KEITH_IT") != "1" {
		t.Skip("set KEITH_IT=1 and DATABASE_URL to run")
	}
	baseURL := os.Getenv("DATABASE_URL")
	if baseURL == "" {
		t.Fatal("KEITH_IT=1 needs DATABASE_URL")
	}
	ctx := context.Background()

	admin, err := sql.Open("pgx", baseURL)
	if err != nil {
		t.Fatal(err)
	}
	// Registered before the drop, so it runs after it (cleanups are LIFO);
	// a deferred Close would run first and leak the scratch database.
	t.Cleanup(func() { admin.Close() })

	suffix := make([]byte, 6)
	rand.Read(suffix)
	dbName := "keith_migrations_" + hex.EncodeToString(suffix)
	if _, err := admin.ExecContext(ctx, "CREATE DATABASE "+dbName); err != nil {
		t.Fatalf("create scratch database: %v", err)
	}
	t.Cleanup(func() {
		admin.ExecContext(context.Background(), "DROP DATABASE IF EXISTS "+dbName+" WITH (FORCE)")
	})

	u, err := url.Parse(baseURL)
	if err != nil {
		t.Fatalf("DATABASE_URL must be a URL: %v", err)
	}
	u.Path = "/" + dbName
	db, err := sql.Open("pgx", u.String())
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	p, err := goose.NewProvider(goose.DialectPostgres, db, FS)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Up(ctx); err != nil {
		t.Fatalf("up: %v", err)
	}
	assertTables(t, db, allTables)
	if _, err := p.DownTo(ctx, 0); err != nil {
		t.Fatalf("down: %v", err)
	}
	assertTables(t, db, nil)
	if _, err := p.Up(ctx); err != nil {
		t.Fatalf("up again: %v", err)
	}
	assertTables(t, db, allTables)
}

func assertTables(t *testing.T, db *sql.DB, want []string) {
	t.Helper()
	rows, err := db.Query(`SELECT table_name FROM information_schema.tables
		WHERE table_schema = 'public' AND table_name <> 'goose_db_version' ORDER BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var got []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		got = append(got, name)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("tables = %v, want %v", got, want)
	}
}
