// Package pgevents is the Postgres events.Store, on the events table of
// migrations/00001_v1_schema.sql.
//
// Append assigns seq as MAX(seq)+1 for the session inside the INSERT. The
// session's lease holder is normally the only writer, so conflicts are
// rare; when two appends do race, the loser hits the (session_id, seq)
// primary key and retries with the next seq. Seqs are therefore unique
// and gap-free per session even without the lease.
//
// Payloads are stored as jsonb. The returned Event carries the payload as
// Postgres stored it (compacted), so an event published live and the same
// event read back by After are byte-identical.
package pgevents

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/Pranay0083/keith/internal/events"
)

// DB is what the queries run on: a *pgxpool.Pool, *pgx.Conn or pgx.Tx.
// Inside a transaction each append attempt runs in a savepoint, so a lost
// race does not abort the caller's transaction.
type DB interface {
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	Begin(ctx context.Context) (pgx.Tx, error)
}

// DefaultMaxAttempts is how many times Append tries before giving up on a
// seq conflict.
const DefaultMaxAttempts = 100

// ErrContention is returned when Append keeps losing the seq race.
var ErrContention = errors.New("pgevents: too many concurrent appends to one session")

// Store is an events.Store on Postgres.
type Store struct {
	db DB
	// Now returns event timestamps; nil means time.Now. Postgres keeps
	// microseconds, so the stored ts is truncated.
	Now func() time.Time
	// MaxAttempts bounds retries on a seq conflict; 0 means
	// DefaultMaxAttempts.
	MaxAttempts int
}

var _ events.Store = (*Store)(nil)

// New returns a Store that runs on db.
func New(db DB) *Store { return &Store{db: db} }

const appendSQL = `
INSERT INTO events (session_id, seq, type, payload, ts)
SELECT $1, COALESCE(MAX(seq), 0) + 1, $2, $3::jsonb, $4
FROM events
WHERE session_id = $1
RETURNING seq, ts, payload::text`

// Append implements events.Store.
func (s *Store) Append(ctx context.Context, sessionID, typ string, payload any) (events.Event, error) {
	if events.IsEphemeral(typ) {
		return events.Event{}, events.ErrEphemeral
	}
	if typ == "" {
		return events.Event{}, errors.New("pgevents: empty event type")
	}
	raw, err := events.MarshalPayload(payload)
	if err != nil {
		return events.Event{}, err
	}
	raw = sanitizeNUL(raw)

	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	ts := now().UTC().Truncate(time.Microsecond)

	_, inTx := s.db.(pgx.Tx)
	attempts := s.MaxAttempts
	if attempts <= 0 {
		attempts = DefaultMaxAttempts
	}
	for i := 0; i < attempts; i++ {
		ev, err := s.appendOnce(ctx, inTx, sessionID, typ, raw, ts)
		if err == nil {
			return ev, nil
		}
		if !isSeqConflict(err) {
			return events.Event{}, fmt.Errorf("pgevents: append %s: %w", typ, err)
		}
		// Lost the race for this seq; back off briefly and take the next.
		if err := sleep(ctx, backoff(i)); err != nil {
			return events.Event{}, err
		}
	}
	return events.Event{}, fmt.Errorf("pgevents: append %s: %w", typ, ErrContention)
}

func (s *Store) appendOnce(ctx context.Context, inTx bool, sessionID, typ string, raw json.RawMessage, ts time.Time) (events.Event, error) {
	db := s.db
	var sp pgx.Tx
	if inTx {
		var err error
		if sp, err = s.db.Begin(ctx); err != nil { // savepoint
			return events.Event{}, err
		}
		db = sp
	}
	ev := events.Event{SessionID: sessionID, Type: typ}
	var stored string
	err := db.QueryRow(ctx, appendSQL, sessionID, typ, string(raw), ts).Scan(&ev.Seq, &ev.TS, &stored)
	if sp != nil {
		if err != nil {
			_ = sp.Rollback(context.WithoutCancel(ctx))
			return events.Event{}, err
		}
		if err := sp.Commit(ctx); err != nil {
			return events.Event{}, err
		}
	}
	if err != nil {
		return events.Event{}, err
	}
	ev.TS = ev.TS.UTC()
	ev.Payload = compact(stored)
	return ev, nil
}

const afterSQL = `
SELECT seq, ts, type, payload::text
FROM events
WHERE session_id = $1 AND seq > $2
ORDER BY seq
LIMIT $3`

// After implements events.Store.
func (s *Store) After(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]events.Event, error) {
	var lim *int
	if limit > 0 {
		lim = &limit
	}
	rows, err := s.db.Query(ctx, afterSQL, sessionID, afterSeq, lim)
	if err != nil {
		return nil, fmt.Errorf("pgevents: after: %w", err)
	}
	defer rows.Close()
	var out []events.Event
	for rows.Next() {
		ev := events.Event{SessionID: sessionID}
		var payload string
		if err := rows.Scan(&ev.Seq, &ev.TS, &ev.Type, &payload); err != nil {
			return nil, fmt.Errorf("pgevents: after: %w", err)
		}
		ev.TS = ev.TS.UTC()
		ev.Payload = compact(payload)
		out = append(out, ev)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgevents: after: %w", err)
	}
	return out, nil
}

// isSeqConflict reports whether err is a unique violation on the events
// primary key, which is how a lost seq race surfaces.
func isSeqConflict(err error) bool {
	var pe *pgconn.PgError
	return errors.As(err, &pe) && pe.Code == "23505" && pe.TableName == "events"
}

func backoff(attempt int) time.Duration {
	d := time.Duration(1+min(attempt, 10)) * time.Millisecond
	return d/2 + rand.N(d)
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// compact removes the whitespace jsonb's text form adds, so stored and
// published payloads are byte-identical.
func compact(s string) json.RawMessage {
	var b bytes.Buffer
	if err := json.Compact(&b, []byte(s)); err != nil {
		return json.RawMessage(s)
	}
	return b.Bytes()
}

// sanitizeNUL replaces the JSON escape \u0000 with �. jsonb rejects
// NUL in strings, and terminal output can contain it. Outside strings
// valid JSON has no backslashes, so a plain scan of escapes is enough.
func sanitizeNUL(raw json.RawMessage) json.RawMessage {
	if !bytes.Contains(raw, []byte(`\u0000`)) {
		return raw
	}
	out := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if c != '\\' || i+1 >= len(raw) {
			out = append(out, c)
			continue
		}
		if bytes.HasPrefix(raw[i+1:], []byte("u0000")) {
			out = append(out, `�`...)
			i += 5
			continue
		}
		out = append(out, c, raw[i+1])
		i++
	}
	return out
}
