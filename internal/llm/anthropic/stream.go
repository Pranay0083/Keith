package anthropic

import (
	"context"
	"encoding/json"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/Pranay0083/keith/internal/llm"
)

var _ llm.Provider = (*Provider)(nil)

// Generate implements llm.Provider. It always streams: text and thinking
// deltas and the start of each tool call go to onEvent as they arrive,
// and the complete message is assembled from the same events.
func (p *Provider) Generate(ctx context.Context, req llm.Request, onEvent func(llm.StreamEvent)) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	params, err := buildParams(req, p.cfg)
	if err != nil {
		return nil, err
	}

	stream := p.client.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	var (
		msg     sdk.Message
		stopped bool
	)
	for stream.Next() {
		ev := stream.Current()
		if err := msg.Accumulate(ev); err != nil {
			return nil, protocolError(err)
		}
		switch ev.Type {
		case "message_stop":
			stopped = true
		case "content_block_start":
			if onEvent != nil && ev.ContentBlock.Type == "tool_use" {
				onEvent(llm.StreamEvent{
					Type:      llm.StreamToolUseStart,
					ToolUseID: ev.ContentBlock.ID,
					ToolName:  ev.ContentBlock.Name,
				})
			}
		case "content_block_delta":
			if onEvent == nil {
				continue
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					onEvent(llm.StreamEvent{Type: llm.StreamTextDelta, Text: ev.Delta.Text})
				}
			case "thinking_delta":
				if ev.Delta.Thinking != "" {
					onEvent(llm.StreamEvent{Type: llm.StreamThinkingDelta, Text: ev.Delta.Thinking})
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, mapError(ctx, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !stopped {
		// The connection closed cleanly but early: the message is
		// incomplete, so sending the request again is the only fix.
		return nil, &llm.Error{Provider: Name, Kind: llm.ErrServer, Message: "stream ended before message_stop"}
	}

	out, err := messageFromAPI(&msg)
	if err != nil {
		return nil, protocolError(err)
	}
	raw := json.RawMessage(msg.RawJSON())
	if !json.Valid(raw) {
		raw = nil
	}
	return &llm.Response{
		Message:    out,
		StopReason: stopReason(msg.StopReason, out),
		Usage:      usageFromAPI(msg.Usage),
		Raw:        raw,
	}, nil
}

// protocolError reports a stream the provider could not make sense of.
// It is classed as a server error so the caller retries.
func protocolError(err error) error {
	return &llm.Error{
		Provider: Name,
		Kind:     llm.ErrServer,
		Message:  fmt.Sprintf("malformed stream: %v", err),
		Err:      err,
	}
}
