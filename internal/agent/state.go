package agent

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/Pranay0083/keith/internal/llm"
)

// State is a session's agent state. The worker loads it before
// Runner.Run and saves it after (and on every Checkpoint), so a paused,
// stopped or finished session resumes exactly where it left off.
//
// Conversation history is strictly append-only: System is built once at
// session start and never changes, and Messages only ever grows. Opus 5.5
// binds thinking blocks to the exact prefix (system, tools, every earlier
// message), so an edit, trim or system prompt change would make the next
// request fail. Work that is not yet part of the conversation (tool
// results held back by a pause, user messages not yet sent) lives in
// Aux.Pending until the loop sends it.
//
// Storage (wave D), via MarshalColumns and UnmarshalState:
//
//	agent_state.llm_messages  = State.Messages as a JSON array of llm.Message
//	agent_state.raw_messages  = {"system": State.System, "aux": State.Aux}
//	agent_state.provider      = the session's provider (llm.Provider.Name)
//
// raw_messages holds System alongside Aux because agent_state has no
// column of its own for it; rebuilding the prompt on resume would change it
// whenever a repo profile changed. The column's default '[]' decodes as an
// empty Aux and an empty System.
//
// State is also plain JSON (json.Marshal) for tests and debugging.
type State struct {
	// System is the system prompt (BuildSystemPrompt), fixed for the
	// session's life.
	System string `json:"system"`
	// Messages is the conversation sent to the model, append-only.
	Messages []llm.Message `json:"messages"`
	// Aux is everything else the loop keeps between runs.
	Aux Aux `json:"aux"`
}

// Aux is the loop's bookkeeping, persisted with the conversation.
type Aux struct {
	// Raw holds each model call's raw provider response (llm.Response.Raw)
	// in order, for debugging and audits. Calls without one are skipped.
	Raw []RawResponse `json:"raw,omitempty"`
	// Edits is the per-file edit history undo_edit uses.
	Edits EditHistory `json:"edits,omitzero"`
	// Repeat is repeat detection's counters.
	Repeat RepeatState `json:"repeat,omitzero"`
	// Pending is the next user message, not yet sent: tool results
	// (including those held back by an ask_user pause), user messages and
	// loop notes. Nil when there is nothing to send.
	Pending *PendingTurn `json:"pending,omitempty"`
	// ContinueTurn is set after a pause_turn response: the next call sends
	// the conversation unchanged so the model continues its turn.
	ContinueTurn bool `json:"continue_turn,omitempty"`
	// Usage is the session's cumulative token usage (the token budget
	// counts it).
	Usage llm.Usage `json:"usage"`
	// Steps counts every model call of the session.
	Steps int `json:"steps"`
	// PRs maps a repo's full name to the pull request open_pr opened for
	// it this session, so pr.opened is emitted once per repo.
	PRs map[string]PR `json:"prs,omitempty"`
	// Delivered lists the IDs of the most recent user inputs handed to
	// the model, so an input delivered twice (a worker crash between
	// polling and acking) is sent only once.
	Delivered []string `json:"delivered,omitempty"`
}

// RawResponse is one model call's raw provider response.
type RawResponse struct {
	// Step is Aux.Steps after the call (1 for the session's first call).
	Step int             `json:"step"`
	Raw  json.RawMessage `json:"raw"`
}

// PendingTurn is a user message being assembled. tool_result blocks go
// first, in tool_use order, then text blocks (user messages and loop
// notes), as the providers require.
type PendingTurn struct {
	// Results are tool_result blocks in the order of the assistant
	// message's tool_use blocks. While Question is open, the result for
	// Question.ToolUseID is a placeholder replaced by the user's answer.
	Results []llm.Block `json:"results,omitempty"`
	// Texts are text blocks sent after the results.
	Texts []llm.Block `json:"texts,omitempty"`
	// Question is the open question the session is waiting on, if any.
	Question *Question `json:"question,omitempty"`
}

func (p *PendingTurn) empty() bool {
	return p == nil || len(p.Results) == 0 && len(p.Texts) == 0
}

// Question is a question to the user that pauses the session until it is
// answered.
type Question struct {
	// ID is the question_id in question.asked and in the answer's
	// message.user. For ask_user it is the tool_use ID; for an automatic
	// "stuck" pause it is "stuck_" + the failing call's tool_use ID.
	ID string `json:"id"`
	// ToolUseID is the ask_user call the answer becomes the tool_result
	// of; empty for an automatic pause, whose answer is sent as text.
	ToolUseID string   `json:"tool_use_id,omitempty"`
	Text      string   `json:"text"`
	Choices   []string `json:"choices,omitempty"`
	// Auto marks a pause the loop raised itself (repeat detection).
	Auto bool `json:"auto,omitempty"`
}

// NewState returns the state for a new session with the given system
// prompt and an empty conversation.
func NewState(system string) *State {
	return &State{System: system, Messages: []llm.Message{}}
}

// OpenQuestion returns the question the session is waiting on, or nil.
// The worker keeps sessions.open_question_id equal to its ID: set it when
// Run returns Paused, clear it when Run returns anything else.
func (s *State) OpenQuestion() *Question {
	if s.Aux.Pending == nil || s.Aux.Pending.Question == nil {
		return nil
	}
	q := *s.Aux.Pending.Question
	return &q
}

// LastAssistantText returns the text of the last assistant message, or "".
func (s *State) LastAssistantText() string {
	for i := len(s.Messages) - 1; i >= 0; i-- {
		if s.Messages[i].Role == llm.RoleAssistant {
			return s.Messages[i].Text()
		}
	}
	return ""
}

func (s *State) pending() *PendingTurn {
	if s.Aux.Pending == nil {
		s.Aux.Pending = &PendingTurn{}
	}
	return s.Aux.Pending
}

// auxColumn is the JSON shape of agent_state.raw_messages.
type auxColumn struct {
	System string `json:"system"`
	Aux    Aux    `json:"aux"`
}

// MarshalColumns encodes the state for agent_state: messages for
// llm_messages and aux for raw_messages.
func (s *State) MarshalColumns() (messages, aux json.RawMessage, err error) {
	msgs := s.Messages
	if msgs == nil {
		msgs = []llm.Message{}
	}
	messages, err = json.Marshal(msgs)
	if err != nil {
		return nil, nil, fmt.Errorf("agent: marshal messages: %w", err)
	}
	aux, err = json.Marshal(auxColumn{System: s.System, Aux: s.Aux})
	if err != nil {
		return nil, nil, fmt.Errorf("agent: marshal aux: %w", err)
	}
	return messages, aux, nil
}

// UnmarshalState decodes the agent_state columns written by
// MarshalColumns. Empty input, null and the columns' default '[]' decode
// as empty.
func UnmarshalState(messages, aux json.RawMessage) (*State, error) {
	st := &State{Messages: []llm.Message{}}
	if !isEmptyJSON(messages) {
		if err := json.Unmarshal(messages, &st.Messages); err != nil {
			return nil, fmt.Errorf("agent: unmarshal messages: %w", err)
		}
	}
	if !isEmptyJSON(aux) {
		var col auxColumn
		dec := json.NewDecoder(bytes.NewReader(aux))
		if err := dec.Decode(&col); err != nil {
			return nil, fmt.Errorf("agent: unmarshal aux: %w", err)
		}
		st.System, st.Aux = col.System, col.Aux
	}
	return st, nil
}

func isEmptyJSON(b json.RawMessage) bool {
	t := bytes.TrimSpace(b)
	return len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte("[]"))
}

// maxDelivered caps Aux.Delivered.
const maxDelivered = 256

func (a *Aux) delivered(id string) bool {
	for _, d := range a.Delivered {
		if d == id {
			return true
		}
	}
	return false
}

func (a *Aux) markDelivered(id string) {
	if id == "" {
		return
	}
	a.Delivered = append(a.Delivered, id)
	if n := len(a.Delivered); n > maxDelivered {
		a.Delivered = append([]string(nil), a.Delivered[n-maxDelivered:]...)
	}
}
