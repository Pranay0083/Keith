package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// AgentState is the model's conversation for a session, saved so a paused
// session resumes exactly.
type AgentState struct {
	SessionID string
	// Provider must equal the session's provider: thinking signatures only
	// replay on the provider that produced them.
	Provider string
	// LLMMessages is the normalized []llm.Message as JSON; nil saves [].
	LLMMessages json.RawMessage
	// RawMessages is the provider's raw responses as a JSON array; nil
	// saves [].
	RawMessages json.RawMessage
	Summary     string
	UpdatedAt   time.Time
}

// SaveAgentState writes a session's agent state, replacing what was
// there. It returns ErrNotFound if the session does not exist and
// ErrProviderMismatch if st.Provider is not the session's provider.
func (s *Store) SaveAgentState(ctx context.Context, st AgentState) error {
	const op = "save agent state"
	if !validID(st.SessionID) {
		return errNotFound(op)
	}
	llm, raw := jsonOr(st.LLMMessages, "[]"), jsonOr(st.RawMessages, "[]")
	tag, err := s.db.Exec(ctx, `
		INSERT INTO agent_state AS a (session_id, provider, llm_messages, raw_messages, summary)
		SELECT s.id, $2, $3::jsonb, $4::jsonb, $5 FROM sessions s
		WHERE s.id = $1 AND s.provider = $2
		ON CONFLICT (session_id) DO UPDATE SET
			llm_messages = EXCLUDED.llm_messages,
			raw_messages = EXCLUDED.raw_messages,
			summary      = EXCLUDED.summary,
			updated_at   = now()
		WHERE a.provider = EXCLUDED.provider`,
		st.SessionID, st.Provider, llm, raw, st.Summary)
	if err != nil {
		return wrap(op, err)
	}
	if tag.RowsAffected() == 1 {
		return nil
	}
	exists, err := s.sessionExists(ctx, st.SessionID)
	if err != nil {
		return wrap(op, err)
	}
	if !exists {
		return errNotFound(op)
	}
	return fmt.Errorf("store: %s: %w", op, ErrProviderMismatch)
}

// LoadAgentState returns a session's agent state, or ErrNotFound if none
// was saved.
func (s *Store) LoadAgentState(ctx context.Context, sessionID string) (AgentState, error) {
	if !validID(sessionID) {
		return AgentState{}, errNotFound("load agent state")
	}
	st := AgentState{SessionID: sessionID}
	var llm, raw []byte
	err := s.db.QueryRow(ctx, `
		SELECT provider, llm_messages, raw_messages, summary, updated_at
		FROM agent_state WHERE session_id = $1`, sessionID).
		Scan(&st.Provider, &llm, &raw, &st.Summary, &st.UpdatedAt)
	if err != nil {
		return AgentState{}, notFound("load agent state", err)
	}
	st.LLMMessages, st.RawMessages = llm, raw
	return st, nil
}

// jsonOr returns raw as a string for a jsonb parameter, or def when raw is
// empty.
func jsonOr(raw json.RawMessage, def string) string {
	if len(raw) == 0 {
		return def
	}
	return string(raw)
}
