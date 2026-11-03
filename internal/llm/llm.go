// Package llm is the provider-agnostic model interface the agent loop runs
// on. Providers (internal/llm/anthropic, internal/llm/gemini) translate
// these types to and from their SDKs; nothing outside a provider package
// sees SDK types.
//
// A session is pinned to one provider for its whole life. Thinking
// signatures and other ProviderRaw data are only valid for the provider
// that produced them, so a conversation saved in agent_state can only be
// resumed by the same provider (sessions.provider, agent_state.provider).
//
// Message and Block are persisted as JSON in agent_state.llm_messages, so
// their JSON shape is part of the storage format: add fields, never rename
// them.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Role is who authored a message. The system prompt is not a message; it
// travels in Request.System.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// BlockType says which fields of a Block are meaningful.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockToolUse    BlockType = "tool_use"
	BlockToolResult BlockType = "tool_result"
	BlockImage      BlockType = "image"
	BlockThinking   BlockType = "thinking"
)

// Block is one piece of message content. Which fields are set depends on
// Type:
//
//	text:        Text
//	tool_use:    ID, Name, Input (assistant messages only)
//	tool_result: ToolUseID, IsError, Content (user messages only; Content
//	             holds text and image blocks)
//	image:       MediaType, Data
//	thinking:    Text, Signature (assistant messages only)
//
// ProviderRaw may be set on any block. It is opaque data the provider needs
// back on the next request (for example a redacted thinking payload or a
// Gemini thought signature on a function call). Code outside the provider
// must keep it and echo it back unchanged, and must never edit, merge or
// drop thinking blocks or blocks that carry ProviderRaw.
type Block struct {
	Type BlockType `json:"type"`

	// Text is the content of a text or thinking block.
	Text string `json:"text,omitempty"`

	// ID, Name and Input describe a tool_use block. Input is the tool's
	// arguments as a JSON object.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// ToolUseID, IsError and Content describe a tool_result block.
	ToolUseID string  `json:"tool_use_id,omitempty"`
	IsError   bool    `json:"is_error,omitempty"`
	Content   []Block `json:"content,omitempty"`

	// MediaType (for example "image/png") and Data (raw bytes, base64 in
	// JSON) describe an image block.
	MediaType string `json:"media_type,omitempty"`
	Data      []byte `json:"data,omitempty"`

	// Signature is the provider's signature over a thinking block.
	Signature string `json:"signature,omitempty"`

	// ProviderRaw is opaque provider data; see the type comment.
	ProviderRaw json.RawMessage `json:"provider_raw,omitempty"`
}

// TextBlock returns a text block.
func TextBlock(text string) Block {
	return Block{Type: BlockText, Text: text}
}

// ToolUseBlock returns a tool_use block. A nil or empty input becomes {}.
func ToolUseBlock(id, name string, input json.RawMessage) Block {
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	return Block{Type: BlockToolUse, ID: id, Name: name, Input: input}
}

// ToolResultBlock returns a tool_result block answering the tool_use with
// the given ID.
func ToolResultBlock(toolUseID string, isError bool, content ...Block) Block {
	return Block{Type: BlockToolResult, ToolUseID: toolUseID, IsError: isError, Content: content}
}

// ImageBlock returns an image block.
func ImageBlock(mediaType string, data []byte) Block {
	return Block{Type: BlockImage, MediaType: mediaType, Data: data}
}

// Message is one turn of the conversation.
type Message struct {
	Role    Role    `json:"role"`
	Content []Block `json:"content"`
}

// UserText returns a user message holding one text block.
func UserText(text string) Message {
	return Message{Role: RoleUser, Content: []Block{TextBlock(text)}}
}

// Text joins the message's text blocks with newlines. Thinking blocks are
// not included.
func (m Message) Text() string {
	var parts []string
	for _, b := range m.Content {
		if b.Type == BlockText && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// ToolUses returns the message's tool_use blocks in order.
func (m Message) ToolUses() []Block {
	var out []Block
	for _, b := range m.Content {
		if b.Type == BlockToolUse {
			out = append(out, b)
		}
	}
	return out
}

// ToolSpec describes a tool to the model. InputSchema is a JSON Schema
// object. Strict asks the provider to guarantee the arguments match the
// schema, where it supports that.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      bool            `json:"strict,omitempty"`
}

// Effort levels understood by every provider. Providers map them to their
// own settings and may accept other values; empty means the provider's
// default.
const (
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
)

// Request is one model call.
type Request struct {
	// Model is the provider's model ID, for example "claude-opus-5-5".
	Model string `json:"model"`
	// System is the system prompt.
	System   string     `json:"system,omitempty"`
	Messages []Message  `json:"messages"`
	Tools    []ToolSpec `json:"tools,omitempty"`
	// MaxTokens caps the output tokens of this call; 0 means the
	// provider's default.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Effort is one of the Effort constants, or empty.
	Effort string `json:"effort,omitempty"`
	// CacheSystemAndTools asks the provider to place a prompt-cache
	// breakpoint after the system prompt and tool list, which stay the
	// same on every turn of a session.
	CacheSystemAndTools bool `json:"cache_system_and_tools,omitempty"`
}

// StopReason is why the model stopped. Providers map their own reasons to
// the closest of these (for example a stop sequence is end_turn, a safety
// block is refusal).
type StopReason string

const (
	// StopEndTurn: the model finished its turn.
	StopEndTurn StopReason = "end_turn"
	// StopToolUse: the message ends with tool_use blocks; run them and
	// send the results back.
	StopToolUse StopReason = "tool_use"
	// StopMaxTokens: the output hit MaxTokens.
	StopMaxTokens StopReason = "max_tokens"
	// StopRefusal: the model declined to continue.
	StopRefusal StopReason = "refusal"
	// StopPauseTurn: the provider paused a long turn; send the
	// conversation back unchanged to let it continue.
	StopPauseTurn StopReason = "pause_turn"
)

// Usage is the token accounting for one call.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

// Add adds o to u, for keeping per-session totals.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
}

// Response is the model's complete answer to one Request.
type Response struct {
	// Message is the assistant message, ready to append to the
	// conversation as-is.
	Message    Message    `json:"message"`
	StopReason StopReason `json:"stop_reason"`
	Usage      Usage      `json:"usage"`
	// Raw is the provider's full response body, kept for agent_state and
	// debugging. Callers must not parse it.
	Raw json.RawMessage `json:"raw,omitempty"`
}

// StreamEventType says what a StreamEvent carries.
type StreamEventType string

const (
	// StreamTextDelta carries a fragment of assistant text in Text.
	StreamTextDelta StreamEventType = "text_delta"
	// StreamThinkingDelta carries a fragment of thinking text in Text.
	StreamThinkingDelta StreamEventType = "thinking_delta"
	// StreamToolUseStart announces a tool call (ToolUseID, ToolName)
	// before its arguments finish streaming.
	StreamToolUseStart StreamEventType = "tool_use_start"
)

// StreamEvent is a live update while a response is generated. It is for
// display only; the Response returned by Generate is authoritative.
type StreamEvent struct {
	Type      StreamEventType `json:"type"`
	Text      string          `json:"text,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	ToolName  string          `json:"tool_name,omitempty"`
}

// Provider is a model backend.
type Provider interface {
	// Name is the provider's stable ID as stored in sessions.provider and
	// agent_state.provider, for example "anthropic" or "gemini".
	Name() string

	// Generate runs one model call and returns the complete response.
	// If onEvent is non-nil it is called synchronously, from the calling
	// goroutine, for each streamed fragment before Generate returns.
	//
	// Errors from the provider's API are returned as *Error so callers
	// can decide whether to retry without knowing the provider. If ctx is
	// cancelled Generate returns ctx.Err() (possibly wrapped).
	Generate(ctx context.Context, req Request, onEvent func(StreamEvent)) (*Response, error)
}

// ErrorKind classifies a provider error.
type ErrorKind string

const (
	// ErrRateLimited: too many requests; retry after RetryAfter.
	ErrRateLimited ErrorKind = "rate_limited"
	// ErrOverloaded: the provider is temporarily overloaded; retry.
	ErrOverloaded ErrorKind = "overloaded"
	// ErrServer: a 5xx or transport failure; retry.
	ErrServer ErrorKind = "server"
	// ErrContextTooLong: the request exceeds the model's context window;
	// compact the conversation before retrying.
	ErrContextTooLong ErrorKind = "context_too_long"
	// ErrInvalidRequest: the request is malformed; do not retry.
	ErrInvalidRequest ErrorKind = "invalid_request"
	// ErrAuth: the API key is missing, invalid or lacks permission.
	ErrAuth ErrorKind = "auth"
)

// Error is a classified provider failure.
type Error struct {
	Provider   string
	Kind       ErrorKind
	StatusCode int           // HTTP status, 0 if none
	Message    string        // human-readable, never contains the API key
	RetryAfter time.Duration // provider's hint, 0 if none
	Err        error         // underlying SDK error, if any
}

func (e *Error) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("llm %s: %s (status %d): %s", e.Provider, e.Kind, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("llm %s: %s: %s", e.Provider, e.Kind, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// Retryable reports whether the same request may succeed if sent again
// after a backoff.
func (e *Error) Retryable() bool {
	switch e.Kind {
	case ErrRateLimited, ErrOverloaded, ErrServer:
		return true
	}
	return false
}

// IsRetryable reports whether err is an *Error that is Retryable.
func IsRetryable(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Retryable()
}
