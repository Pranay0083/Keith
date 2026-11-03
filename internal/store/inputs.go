package store

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/google/uuid"
)

// InputKind is what a session input is.
type InputKind string

const (
	// InputMessage steers a running session or follows up on a finished
	// one (POST /messages).
	InputMessage InputKind = "message"
	// InputAnswer answers the open question.asked (POST /answers).
	InputAnswer InputKind = "answer"
)

// Valid reports whether k is a defined kind.
func (k InputKind) Valid() bool { return k == InputMessage || k == InputAnswer }

// Input is a message or answer a person sent to a session, waiting for the
// worker (session_inputs). The API writes it; the worker reads the
// pending ones with ListPendingInputs, hands them to the model in order,
// emits one message.user per input with MessageID = ID (and QuestionID
// for answers), then marks them consumed with MarkInputsConsumed.
type Input struct {
	// ID is "m_" + a time-ordered ID (NewInputID): the message_id the API
	// returned.
	ID        string
	SessionID string
	Kind      InputKind
	// QuestionID is set for answers only.
	QuestionID string
	// UserID is the author, or "".
	UserID    string
	Text      string
	CreatedAt time.Time
	// ConsumedAt is nil while the input is pending.
	ConsumedAt *time.Time
}

// CreateInputParams describes a new input.
type CreateInputParams struct {
	// ID "" means NewInputID().
	ID        string
	SessionID string
	Kind      InputKind
	// QuestionID is required for answers and must be "" for messages.
	QuestionID string
	UserID     string
	Text       string
}

var inputIDRe = regexp.MustCompile(`^m_[0-9A-Za-z]+$`)

// ValidInputID reports whether id has the shape NewInputID produces.
func ValidInputID(id string) bool { return len(id) <= 64 && inputIDRe.MatchString(id) }

// crockford is the Crockford base32 alphabet ULIDs use.
const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// NewInputID returns "m_" followed by 26 Crockford base32 characters
// encoding a UUIDv7: a millisecond timestamp then random bits, so IDs
// sort by creation time, like a ULID ("m_01J…").
func NewInputID() string {
	u, err := uuid.NewV7()
	if err != nil { // only if crypto/rand fails
		_, _ = rand.Read(u[:])
	}
	// 128 bits -> 26 base32 digits (130 bits): two zero bits in front
	// make the stream split evenly. Only the low bits of acc matter, so
	// its overflow is harmless.
	var out [26]byte
	var acc uint64
	bits, i := uint(2), 0
	for _, b := range u {
		acc = acc<<8 | uint64(b)
		bits += 8
		for bits >= 5 {
			bits -= 5
			out[i] = crockford[(acc>>bits)&31]
			i++
		}
	}
	return "m_" + string(out[:])
}

const inputCols = `id, session_id, kind, question_id, user_id, text, created_at, consumed_at`

func scanInput(row interface{ Scan(...any) error }) (Input, error) {
	var x Input
	var questionID, userID *string
	err := row.Scan(&x.ID, &x.SessionID, &x.Kind, &questionID, &userID, &x.Text, &x.CreatedAt, &x.ConsumedAt)
	x.QuestionID, x.UserID = deref(questionID), deref(userID)
	return x, err
}

// CreateInput stores a pending input for the worker. It does not check
// the session's status; callers that must (the API's /messages and
// /answers) lock the session with GetSessionForUpdate in the same
// transaction first.
func (s *Store) CreateInput(ctx context.Context, p CreateInputParams) (Input, error) {
	const op = "create input"
	if !p.Kind.Valid() {
		return Input{}, fmt.Errorf("store: %s: invalid kind %q", op, p.Kind)
	}
	if (p.Kind == InputAnswer) != (p.QuestionID != "") {
		return Input{}, fmt.Errorf("store: %s: an answer needs a question ID and a message must not have one", op)
	}
	if p.ID == "" {
		p.ID = NewInputID()
	}
	if !ValidInputID(p.ID) {
		return Input{}, fmt.Errorf("store: %s: invalid input ID %q", op, p.ID)
	}
	if !validID(p.SessionID) || (p.UserID != "" && !validID(p.UserID)) {
		return Input{}, errNotFound(op)
	}
	x, err := scanInput(s.db.QueryRow(ctx, `
		INSERT INTO session_inputs (id, session_id, kind, question_id, user_id, text)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING `+inputCols,
		p.ID, p.SessionID, string(p.Kind), nullIfEmpty(p.QuestionID), nullIfEmpty(p.UserID), p.Text))
	if pgCode(err) == "23503" { // no such session or user
		return Input{}, errNotFound(op)
	}
	if pgCode(err) == "23505" {
		return Input{}, fmt.Errorf("store: %s: %w: input ID %s exists", op, ErrConflict, p.ID)
	}
	return x, wrap(op, err)
}

// ListPendingInputs returns a session's unconsumed inputs, oldest first.
// The worker calls it before each model turn and on resume.
func (s *Store) ListPendingInputs(ctx context.Context, sessionID string) ([]Input, error) {
	const op = "list pending inputs"
	if !validID(sessionID) {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
		SELECT `+inputCols+` FROM session_inputs
		WHERE session_id = $1 AND consumed_at IS NULL
		ORDER BY created_at, id`, sessionID)
	if err != nil {
		return nil, wrap(op, err)
	}
	defer rows.Close()
	var out []Input
	for rows.Next() {
		x, err := scanInput(rows)
		if err != nil {
			return nil, wrap(op, err)
		}
		out = append(out, x)
	}
	return out, wrap(op, rows.Err())
}

// GetInput returns one input of a session.
func (s *Store) GetInput(ctx context.Context, sessionID, id string) (Input, error) {
	if !validID(sessionID) || !ValidInputID(id) {
		return Input{}, errNotFound("get input")
	}
	x, err := scanInput(s.db.QueryRow(ctx,
		`SELECT `+inputCols+` FROM session_inputs WHERE session_id = $1 AND id = $2`, sessionID, id))
	return x, notFound("get input", err)
}

// MarkInputsConsumed marks inputs of a session consumed and returns how
// many changed. Already-consumed and unknown IDs are skipped, so the call
// is idempotent. The worker calls it after it has emitted message.user
// for each input and saved agent state with them in the conversation.
func (s *Store) MarkInputsConsumed(ctx context.Context, sessionID string, ids []string) (int64, error) {
	const op = "mark inputs consumed"
	if len(ids) == 0 {
		return 0, nil
	}
	if !validID(sessionID) {
		return 0, errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE session_inputs SET consumed_at = now()
		WHERE session_id = $1 AND id = ANY($2::text[]) AND consumed_at IS NULL`, sessionID, ids)
	if err != nil {
		return 0, wrap(op, err)
	}
	return tag.RowsAffected(), nil
}

// SetOpenQuestion records the question.asked waiting for an answer. The
// worker calls it before it emits question.asked and moves the session to
// waiting_for_user; /answers accepts only this question ID.
func (s *Store) SetOpenQuestion(ctx context.Context, sessionID, questionID string) error {
	const op = "set open question"
	if questionID == "" {
		return errors.New("store: set open question: empty question ID")
	}
	if !validID(sessionID) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET open_question_id = $2, updated_at = now() WHERE id = $1`, sessionID, questionID)
	return affected(op, tag.RowsAffected(), err)
}

// ClearOpenQuestion clears the open question. The API clears it when it
// stores an answer (so a second answer is a question_mismatch); the
// worker clears it when it abandons a question (cancel).
func (s *Store) ClearOpenQuestion(ctx context.Context, sessionID string) error {
	const op = "clear open question"
	if !validID(sessionID) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET open_question_id = NULL, updated_at = now() WHERE id = $1`, sessionID)
	return affected(op, tag.RowsAffected(), err)
}

// DeleteInput deletes a pending input. The API uses it to undo an input
// whose resume_session could not be enqueued, so the user can retry.
func (s *Store) DeleteInput(ctx context.Context, sessionID, id string) error {
	const op = "delete input"
	if !validID(sessionID) || !ValidInputID(id) {
		return errNotFound(op)
	}
	tag, err := s.db.Exec(ctx, `
		DELETE FROM session_inputs WHERE session_id = $1 AND id = $2 AND consumed_at IS NULL`, sessionID, id)
	return affected(op, tag.RowsAffected(), err)
}

// ReopenQuestion sets the open question back to questionID if none is
// open and the session is still waiting_for_user, undoing the API's
// ClearOpenQuestion for an answer it had to delete. It reports whether
// it changed anything.
func (s *Store) ReopenQuestion(ctx context.Context, sessionID, questionID string) (bool, error) {
	if !validID(sessionID) || questionID == "" {
		return false, errNotFound("reopen question")
	}
	tag, err := s.db.Exec(ctx, `
		UPDATE sessions SET open_question_id = $2, updated_at = now()
		WHERE id = $1 AND open_question_id IS NULL AND status = 'waiting_for_user'`, sessionID, questionID)
	if err != nil {
		return false, wrap("reopen question", err)
	}
	return tag.RowsAffected() == 1, nil
}
