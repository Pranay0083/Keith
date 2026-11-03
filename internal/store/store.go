// Package store is the Postgres data layer: typed models and hand-written
// pgx queries for every V1 table except events, whose reads and writes go
// through events.Store (see Events and internal/events/pgevents).
//
// Conventions:
//   - IDs are uuid strings. A malformed ID never matches a row, so lookups
//     return ErrNotFound for it rather than a driver error.
//   - Nullable text columns map to plain strings; "" means NULL.
//   - Every method takes a context and runs on the pool, or on the
//     transaction when the Store came from WithTx.
//
// Tenancy is enforced in the queries where a row names another row: a
// session's repos and a repo secret's repo must belong to the same org.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
	"github.com/pressly/goose/v3/lock"

	"github.com/Pranay0083/keith/internal/events/pgevents"
	"github.com/Pranay0083/keith/migrations"
)

var (
	// ErrNotFound means the row does not exist, or is outside the org the
	// query was scoped to.
	ErrNotFound = errors.New("store: not found")
	// ErrConflict means the write would break an invariant held by
	// existing rows, for example linking an installation to a second org.
	ErrConflict = errors.New("store: conflict")
	// ErrProviderMismatch means agent state was saved for a provider other
	// than the session's.
	ErrProviderMismatch = errors.New("store: provider does not match the session")
)

// DBTX is what queries run on: a *pgxpool.Pool or a pgx.Tx. Begin starts a
// transaction on a pool and a savepoint on a transaction.
type DBTX interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
	Begin(ctx context.Context) (pgx.Tx, error)
}

// Store runs queries against Postgres. A Store from Open owns its pool; a
// Store passed to a WithTx callback is bound to that transaction.
type Store struct {
	pool *pgxpool.Pool
	db   DBTX
}

// Open connects to databaseURL and checks the connection.
func Open(ctx context.Context, databaseURL string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		// The parse error can echo the URL, password included.
		return nil, errors.New("store: DATABASE_URL is not a valid Postgres connection string")
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("store: open pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("store: ping: %w", err)
	}
	return &Store{pool: pool, db: pool}, nil
}

// New wraps an existing pool. Close closes it.
func New(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool, db: pool}
}

// Close closes the pool. It is a no-op on a transaction-bound Store.
func (s *Store) Close() {
	if s.pool != nil && s.db == DBTX(s.pool) {
		s.pool.Close()
	}
}

// Pool returns the underlying pool.
func (s *Store) Pool() *pgxpool.Pool { return s.pool }

// DB returns what this Store's queries run on: the pool, or the current
// transaction inside WithTx.
func (s *Store) DB() DBTX { return s.db }

// Events returns an events.Store that writes on the same pool or
// transaction as s. Inside WithTx, publish the appended events only after
// the transaction commits: a rolled-back seq is reused by the next append,
// and a client that saw the published copy would drop the real one.
func (s *Store) Events() *pgevents.Store { return pgevents.New(s.db) }

// Migrate applies every pending migration in migrations.FS. A Postgres
// advisory lock serialises concurrent callers (the api and the worker
// both migrate on start).
func (s *Store) Migrate(ctx context.Context) error {
	if s.pool == nil {
		return errors.New("store: Migrate needs a pool-backed Store")
	}
	db := stdlib.OpenDBFromPool(s.pool)
	defer db.Close()
	return migrate(ctx, db)
}

func migrate(ctx context.Context, db *sql.DB) error {
	locker, err := lock.NewPostgresSessionLocker()
	if err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	p, err := goose.NewProvider(goose.DialectPostgres, db, migrations.FS, goose.WithSessionLocker(locker))
	if err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	if _, err := p.Up(ctx); err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	return nil
}

// WithTx runs fn in a transaction and commits if fn returns nil. Inside
// fn, use tx for every query. Nested calls use savepoints.
func (s *Store) WithTx(ctx context.Context, fn func(tx *Store) error) (err error) {
	t, err := s.db.Begin(ctx)
	if err != nil {
		return fmt.Errorf("store: begin: %w", err)
	}
	defer func() {
		if p := recover(); p != nil {
			_ = t.Rollback(context.WithoutCancel(ctx))
			panic(p)
		}
		if err != nil {
			_ = t.Rollback(context.WithoutCancel(ctx))
		}
	}()
	if err = fn(&Store{pool: s.pool, db: t}); err != nil {
		return err
	}
	if err = t.Commit(ctx); err != nil {
		return fmt.Errorf("store: commit: %w", err)
	}
	return nil
}

// validID reports whether id is a well-formed uuid in the hyphenated
// 36-character form, in either case. uuid.Validate alone also accepts
// forms Postgres rejects (urn:uuid:…), which would surface as a driver
// error instead of ErrNotFound.
func validID(id string) bool { return len(id) == 36 && uuid.Validate(id) == nil }

// validIDs reports whether every id is a well-formed uuid.
func validIDs(ids []string) bool {
	for _, id := range ids {
		if !validID(id) {
			return false
		}
	}
	return true
}

// notFound maps pgx.ErrNoRows to ErrNotFound and wraps other errors with
// the operation name.
func notFound(op string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("store: %s: %w", op, ErrNotFound)
	}
	return fmt.Errorf("store: %s: %w", op, err)
}

// errNotFound is ErrNotFound with the operation name.
func errNotFound(op string) error { return fmt.Errorf("store: %s: %w", op, ErrNotFound) }

// wrap prefixes err with the operation name.
func wrap(op string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("store: %s: %w", op, err)
}

// pgCode returns the SQLSTATE of a Postgres error, or "".
func pgCode(err error) string {
	var pe *pgconn.PgError
	if errors.As(err, &pe) {
		return pe.Code
	}
	return ""
}

// nullIfEmpty maps "" to SQL NULL.
func nullIfEmpty(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// deref maps SQL NULL to "".
func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// isNoRows reports whether err is pgx.ErrNoRows.
func isNoRows(err error) bool { return errors.Is(err, pgx.ErrNoRows) }

// isNotFound reports whether err is ErrNotFound.
func isNotFound(err error) bool { return errors.Is(err, ErrNotFound) }
