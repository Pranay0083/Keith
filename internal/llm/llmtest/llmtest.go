// Package llmtest provides a scripted llm.Provider for tests.
//
// Queue the responses a test expects the model to give, run the code under
// test, then inspect the requests it sent:
//
//	p := llmtest.New(
//		llmtest.ToolCall("call_1", "read_file", map[string]any{"path": "README.md"}),
//		llmtest.Text("Done."),
//	)
//	// ... run the agent loop with p ...
//	reqs := p.Requests()
package llmtest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/Pranay0083/keith/internal/llm"
)

// ErrExhausted is returned by Generate when no scripted step is left.
var ErrExhausted = errors.New("llmtest: no scripted response left")

// StepFunc computes a response from the request, for tests that need to
// look at what was sent before answering.
type StepFunc func(req llm.Request) (*llm.Response, error)

// Provider is a scripted llm.Provider. It is safe for concurrent use.
type Provider struct {
	name string

	mu       sync.Mutex
	steps    []StepFunc
	requests []llm.Request
}

var _ llm.Provider = (*Provider)(nil)

// New returns a Provider named "fake" that answers with responses in
// order.
func New(responses ...*llm.Response) *Provider {
	p := &Provider{name: "fake"}
	p.Push(responses...)
	return p
}

// WithName sets the name returned by Name and returns p.
func (p *Provider) WithName(name string) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.name = name
	return p
}

// Push queues responses.
func (p *Provider) Push(responses ...*llm.Response) {
	for _, r := range responses {
		r := r
		p.PushFunc(func(llm.Request) (*llm.Response, error) { return r, nil })
	}
}

// PushError queues a failed call.
func (p *Provider) PushError(err error) {
	p.PushFunc(func(llm.Request) (*llm.Response, error) { return nil, err })
}

// PushFunc queues a step computed from the request.
func (p *Provider) PushFunc(f StepFunc) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.steps = append(p.steps, f)
}

// Requests returns copies of every request Generate received, in order.
func (p *Provider) Requests() []llm.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]llm.Request, len(p.requests))
	copy(out, p.requests)
	return out
}

// Remaining returns how many scripted steps are left.
func (p *Provider) Remaining() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.steps)
}

// Name implements llm.Provider.
func (p *Provider) Name() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.name
}

// Generate implements llm.Provider. It records req, runs the next scripted
// step and, if onEvent is set, streams the response's text, thinking and
// tool_use blocks as events before returning it.
func (p *Provider) Generate(ctx context.Context, req llm.Request, onEvent func(llm.StreamEvent)) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	p.mu.Lock()
	p.requests = append(p.requests, cloneRequest(req))
	if len(p.steps) == 0 {
		p.mu.Unlock()
		return nil, ErrExhausted
	}
	step := p.steps[0]
	p.steps = p.steps[1:]
	p.mu.Unlock()

	resp, err := step(req)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("llmtest: step returned a nil response")
	}
	if onEvent != nil {
		for _, b := range resp.Message.Content {
			switch b.Type {
			case llm.BlockText:
				onEvent(llm.StreamEvent{Type: llm.StreamTextDelta, Text: b.Text})
			case llm.BlockThinking:
				onEvent(llm.StreamEvent{Type: llm.StreamThinkingDelta, Text: b.Text})
			case llm.BlockToolUse:
				onEvent(llm.StreamEvent{Type: llm.StreamToolUseStart, ToolUseID: b.ID, ToolName: b.Name})
			}
		}
	}
	return resp, nil
}

// Text returns an end_turn response holding one text block.
func Text(text string) *llm.Response {
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{llm.TextBlock(text)}},
		StopReason: llm.StopEndTurn,
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

// ToolCall returns a tool_use response calling one tool. input is
// marshaled to JSON; it may be a json.RawMessage, a struct or a map.
// It panics if input cannot be marshaled.
func ToolCall(id, name string, input any) *llm.Response {
	return ToolCalls(llm.ToolUseBlock(id, name, mustJSON(input)))
}

// ToolCalls returns a tool_use response holding the given blocks, which
// may mix text and tool_use blocks.
func ToolCalls(blocks ...llm.Block) *llm.Response {
	return &llm.Response{
		Message:    llm.Message{Role: llm.RoleAssistant, Content: blocks},
		StopReason: llm.StopToolUse,
		Usage:      llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

func mustJSON(v any) json.RawMessage {
	if raw, ok := v.(json.RawMessage); ok {
		return raw
	}
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("llmtest: marshal tool input: %v", err))
	}
	return b
}

// cloneRequest copies the slices a caller is likely to mutate after the
// call (the conversation grows in place), so recorded requests stay as
// they were sent.
func cloneRequest(req llm.Request) llm.Request {
	msgs := make([]llm.Message, len(req.Messages))
	for i, m := range req.Messages {
		m.Content = append([]llm.Block(nil), m.Content...)
		msgs[i] = m
	}
	req.Messages = msgs
	req.Tools = append([]llm.ToolSpec(nil), req.Tools...)
	return req
}
