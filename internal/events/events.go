// Package events is the session event log: the typed events a session
// emits, and the Store and Bus that persist and fan them out.
//
// Each session has one append-only log, and that log is the source of
// truth for the UI. The rules:
//
//  1. One writer per session. The worker holding the session's lease is
//     the only code that appends, so Store.Append can assign seq without
//     races between workers.
//  2. Store, then publish. An event is inserted, then published to the
//     Bus. Ephemeral events (message.delta) are published only; the
//     finished message is stored as message.assistant. Emitter enforces
//     this order.
//  3. Replay on connect. A client sends {session_id, after_seq}. The
//     server subscribes to the Bus first, then sends stored events after
//     after_seq from Store.After, then forwards live events, dropping any
//     seq it already sent. Bus delivery is best-effort; Store is not.
//
// The payload structs are converted to TypeScript for the web app, so they
// stay plain: exported fields with json tags, no interfaces, no embedded
// pointers. json.RawMessage fields map to `unknown` in TypeScript.
package events

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Event is one entry in a session's log, exactly as stored and sent to
// clients.
type Event struct {
	SessionID string `json:"session_id"`
	// Seq starts at 1 and increases by 1 per stored event in a session.
	// Ephemeral events have Seq 0.
	Seq     int64           `json:"seq"`
	TS      time.Time       `json:"ts"`
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// Event types. Each has a payload struct below; NewPayload maps between
// them.
const (
	TypeSessionStatus    = "session.status"
	TypeMessageUser      = "message.user"
	TypeMessageAssistant = "message.assistant"
	TypeMessageDelta     = "message.delta"
	TypeToolCallStarted  = "tool_call.started"
	TypeToolCallOutput   = "tool_call.output"
	TypeToolCallFinished = "tool_call.finished"
	TypeShellCreated     = "shell.created"
	TypeFileChanged      = "file.changed"
	TypeQuestionAsked    = "question.asked"
	TypePROpened         = "pr.opened"
	TypeError            = "error"
)

// Types lists every event type.
var Types = []string{
	TypeSessionStatus,
	TypeMessageUser,
	TypeMessageAssistant,
	TypeMessageDelta,
	TypeToolCallStarted,
	TypeToolCallOutput,
	TypeToolCallFinished,
	TypeShellCreated,
	TypeFileChanged,
	TypeQuestionAsked,
	TypePROpened,
	TypeError,
}

// IsEphemeral reports whether events of this type are published without
// being stored. Only message.delta is.
func IsEphemeral(typ string) bool {
	return typ == TypeMessageDelta
}

// NewPayload returns a pointer to a zero payload struct for typ, ready to
// json.Unmarshal into, and false for an unknown type.
func NewPayload(typ string) (any, bool) {
	switch typ {
	case TypeSessionStatus:
		return &SessionStatusPayload{}, true
	case TypeMessageUser:
		return &MessageUserPayload{}, true
	case TypeMessageAssistant:
		return &MessageAssistantPayload{}, true
	case TypeMessageDelta:
		return &MessageDeltaPayload{}, true
	case TypeToolCallStarted:
		return &ToolCallStartedPayload{}, true
	case TypeToolCallOutput:
		return &ToolCallOutputPayload{}, true
	case TypeToolCallFinished:
		return &ToolCallFinishedPayload{}, true
	case TypeShellCreated:
		return &ShellCreatedPayload{}, true
	case TypeFileChanged:
		return &FileChangedPayload{}, true
	case TypeQuestionAsked:
		return &QuestionAskedPayload{}, true
	case TypePROpened:
		return &PROpenedPayload{}, true
	case TypeError:
		return &ErrorPayload{}, true
	}
	return nil, false
}

// Decode unmarshals e's payload into the struct for its type and returns a
// pointer to it (for example *ToolCallStartedPayload).
func Decode(e Event) (any, error) {
	p, ok := NewPayload(e.Type)
	if !ok {
		return nil, fmt.Errorf("events: unknown event type %q", e.Type)
	}
	if err := json.Unmarshal(e.Payload, p); err != nil {
		return nil, fmt.Errorf("events: decode %s payload: %w", e.Type, err)
	}
	return p, nil
}

// SessionStatus is a session's lifecycle state. It is also the value of
// sessions.status.
type SessionStatus string

const (
	StatusQueued         SessionStatus = "queued"
	StatusProvisioning   SessionStatus = "provisioning"
	StatusSettingUp      SessionStatus = "setting_up"
	StatusRunning        SessionStatus = "running"
	StatusWaitingForUser SessionStatus = "waiting_for_user"
	StatusCompleted      SessionStatus = "completed"
	StatusFailed         SessionStatus = "failed"
	StatusCancelled      SessionStatus = "cancelled"
	StatusNoChanges      SessionStatus = "no_changes"
)

// Statuses lists every session status.
var Statuses = []SessionStatus{
	StatusQueued,
	StatusProvisioning,
	StatusSettingUp,
	StatusRunning,
	StatusWaitingForUser,
	StatusCompleted,
	StatusFailed,
	StatusCancelled,
	StatusNoChanges,
}

// Terminal reports whether no more work happens in this status without a
// new user message.
func (s SessionStatus) Terminal() bool {
	switch s {
	case StatusCompleted, StatusFailed, StatusCancelled, StatusNoChanges:
		return true
	}
	return false
}

// SessionStatusPayload is the payload of session.status.
type SessionStatusPayload struct {
	Status SessionStatus `json:"status"`
	// Reason explains failed, cancelled and no_changes.
	Reason string `json:"reason,omitempty"`
}

// MessageUserPayload is the payload of message.user: a chat message from a
// person, including the session's first prompt and answers to questions.
type MessageUserPayload struct {
	MessageID string `json:"message_id"`
	// UserID is the author's users.id; empty for messages from a trigger.
	UserID string `json:"user_id,omitempty"`
	Text   string `json:"text"`
	// QuestionID is set when this message answers a question.asked.
	QuestionID string `json:"question_id,omitempty"`
}

// MessageAssistantPayload is the payload of message.assistant: a complete
// chat message from the agent (its final text, or a message_user note).
type MessageAssistantPayload struct {
	MessageID string `json:"message_id"`
	Text      string `json:"text"`
}

// MessageDeltaPayload is the payload of message.delta: a streamed fragment
// of the assistant message with the same MessageID. Ephemeral.
type MessageDeltaPayload struct {
	MessageID string `json:"message_id"`
	Text      string `json:"text"`
}

// ToolCallStartedPayload is the payload of tool_call.started.
type ToolCallStartedPayload struct {
	// CallID is the model's tool_use ID.
	CallID string `json:"call_id"`
	Tool   string `json:"tool"`
	// Input is the tool's arguments, with secrets redacted.
	Input json.RawMessage `json:"input"`
}

// ToolCallOutputPayload is the payload of tool_call.output: a chunk of
// terminal output from a running tool call.
type ToolCallOutputPayload struct {
	CallID string `json:"call_id"`
	// ShellID names the shell the output belongs to, if any.
	ShellID string `json:"shell_id,omitempty"`
	// Chunk is raw terminal output (may contain ANSI escapes), with
	// secrets redacted. Invalid UTF-8 is replaced.
	Chunk string `json:"chunk"`
}

// ToolCallFinishedPayload is the payload of tool_call.finished.
type ToolCallFinishedPayload struct {
	CallID  string `json:"call_id"`
	Tool    string `json:"tool"`
	IsError bool   `json:"is_error"`
	// ExitCode is set for shell commands; omitted means 0 or not
	// applicable.
	ExitCode   int   `json:"exit_code,omitempty"`
	TimedOut   bool  `json:"timed_out,omitempty"`
	DurationMS int64 `json:"duration_ms"`
	// Summary is a one-line result for the timeline.
	Summary string `json:"summary,omitempty"`
}

// ShellCreatedPayload is the payload of shell.created. The UI opens a
// terminal tab for it.
type ShellCreatedPayload struct {
	ShellID string `json:"shell_id"`
	CallID  string `json:"call_id,omitempty"`
}

// FileAction is what happened to a file.
type FileAction string

const (
	FileCreated  FileAction = "created"
	FileModified FileAction = "modified"
	FileDeleted  FileAction = "deleted"
)

// FileChangedPayload is the payload of file.changed. The UI refreshes the
// diff view.
type FileChangedPayload struct {
	CallID string `json:"call_id,omitempty"`
	// Repo is the repo's full name (owner/name), if the file is in one.
	Repo string `json:"repo,omitempty"`
	// Path is the file's absolute path in the sandbox.
	Path   string     `json:"path"`
	Action FileAction `json:"action"`
}

// QuestionAskedPayload is the payload of question.asked. The UI turns the
// input box into an answer form.
type QuestionAskedPayload struct {
	// QuestionID is the ask_user tool call's ID; the answer refers to it.
	QuestionID string   `json:"question_id"`
	Question   string   `json:"question"`
	Choices    []string `json:"choices,omitempty"`
}

// PROpenedPayload is the payload of pr.opened.
type PROpenedPayload struct {
	Repo   string `json:"repo"`
	Number int    `json:"number"`
	URL    string `json:"url"`
	Title  string `json:"title"`
	Branch string `json:"branch"`
}

// ErrorPayload is the payload of error.
type ErrorPayload struct {
	// Code is a stable machine-readable reason, for example
	// "sandbox_provision_failed" or "llm_rate_limited".
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
	// Retryable says whether the UI should offer a retry.
	Retryable bool `json:"retryable,omitempty"`
}

// ErrEphemeral is returned by Store.Append for an ephemeral event type.
var ErrEphemeral = errors.New("events: ephemeral event types are not stored")

// Store persists session events.
type Store interface {
	// Append stores an event with the session's next seq and the current
	// time, and returns it. payload is marshaled with encoding/json (a
	// json.RawMessage is stored as-is). The caller must hold the session's
	// lease. It returns ErrEphemeral for ephemeral types.
	Append(ctx context.Context, sessionID, typ string, payload any) (Event, error)

	// After returns stored events with seq > afterSeq in seq order, at
	// most limit of them (limit <= 0 means all).
	After(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]Event, error)
}

// Bus fans events out to live subscribers.
type Bus interface {
	// Publish delivers e to the session's current subscribers.
	Publish(ctx context.Context, e Event) error

	// Subscribe returns a channel of events published for the session
	// after the call. The returned func ends the subscription and closes
	// the channel; cancelling ctx does the same. Delivery is best-effort:
	// a subscriber that falls behind may miss events and must recover by
	// replaying from the Store.
	Subscribe(ctx context.Context, sessionID string) (<-chan Event, func(), error)
}

// ErrPublish wraps a Bus failure after the event was stored. The event is
// durable; live clients will see it on their next replay.
var ErrPublish = errors.New("events: publish failed after store")

// Emitter writes events in the order the log rules require: store, then
// publish (publish only for ephemeral types).
type Emitter struct {
	Store Store
	Bus   Bus
	// Now returns the timestamp for ephemeral events; nil means time.Now.
	Now func() time.Time
}

// Emit stores (unless ephemeral) and publishes one event and returns it.
// If only the publish fails, it returns the stored event and an error
// wrapping ErrPublish.
func (em *Emitter) Emit(ctx context.Context, sessionID, typ string, payload any) (Event, error) {
	var ev Event
	if IsEphemeral(typ) {
		raw, err := MarshalPayload(payload)
		if err != nil {
			return Event{}, err
		}
		now := time.Now
		if em.Now != nil {
			now = em.Now
		}
		ev = Event{SessionID: sessionID, TS: now().UTC(), Type: typ, Payload: raw}
	} else {
		var err error
		ev, err = em.Store.Append(ctx, sessionID, typ, payload)
		if err != nil {
			return Event{}, err
		}
	}
	if err := em.Bus.Publish(ctx, ev); err != nil {
		if IsEphemeral(typ) {
			return ev, err
		}
		return ev, fmt.Errorf("%w: %w", ErrPublish, err)
	}
	return ev, nil
}

// MarshalPayload encodes a payload the way Store implementations must:
// nil becomes {}, a json.RawMessage is kept as-is, anything else goes
// through encoding/json.
func MarshalPayload(payload any) (json.RawMessage, error) {
	switch p := payload.(type) {
	case nil:
		return json.RawMessage(`{}`), nil
	case json.RawMessage:
		if len(p) == 0 {
			return json.RawMessage(`{}`), nil
		}
		return p, nil
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("events: marshal payload: %w", err)
	}
	return b, nil
}
