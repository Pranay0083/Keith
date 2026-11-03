package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/Pranay0083/keith/internal/llm"
)

// redactedThinking is the wire shape of a redacted_thinking block. The
// provider stores it, marshaled, in Block.ProviderRaw of a thinking block
// and sends it back verbatim.
type redactedThinking struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

// buildParams converts req into the SDK's request parameters.
func buildParams(req llm.Request, cfg settings) (sdk.MessageNewParams, error) {
	if req.Model == "" {
		return sdk.MessageNewParams{}, invalidRequest("request has no model")
	}
	if len(req.Messages) == 0 {
		return sdk.MessageNewParams{}, invalidRequest("request has no messages")
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = cfg.defaultMaxTokens
	}

	adaptive := sdk.ThinkingConfigAdaptiveParam{
		Display: sdk.ThinkingConfigAdaptiveDisplay(cfg.thinkingDisplay),
	}
	p := sdk.MessageNewParams{
		Model:        sdk.Model(req.Model),
		MaxTokens:    int64(maxTokens),
		Thinking:     sdk.ThinkingConfigParamUnion{OfAdaptive: &adaptive},
		OutputConfig: sdk.OutputConfigParam{Effort: effort(req.Effort)},
	}

	if req.System != "" {
		block := sdk.TextBlockParam{Text: req.System}
		if req.CacheSystemAndTools {
			block.CacheControl = sdk.NewCacheControlEphemeralParam()
		}
		p.System = []sdk.TextBlockParam{block}
	}

	for i, spec := range req.Tools {
		tool, err := toolParam(spec)
		if err != nil {
			return sdk.MessageNewParams{}, err
		}
		if req.CacheSystemAndTools && i == len(req.Tools)-1 {
			tool.CacheControl = sdk.NewCacheControlEphemeralParam()
		}
		p.Tools = append(p.Tools, sdk.ToolUnionParam{OfTool: &tool})
	}

	if req.CacheSystemAndTools && cfg.cacheConversation {
		// Automatic caching: the API puts one more breakpoint on the last
		// cacheable block and moves it forward as the conversation grows,
		// so each turn reads the previous turn's prefix from cache.
		p.CacheControl = sdk.NewCacheControlEphemeralParam()
	}

	for i, m := range req.Messages {
		mp, err := messageParam(m)
		if err != nil {
			return sdk.MessageNewParams{}, fmt.Errorf("message %d: %w", i, err)
		}
		p.Messages = append(p.Messages, mp)
	}
	return p, nil
}

// effort maps llm effort levels to output_config.effort. Empty means high:
// claude-opus-5-5 defaults to medium, which is low for coding work. Other
// values (xhigh, max) pass through for the API to accept or reject.
func effort(e string) sdk.OutputConfigEffort {
	e = strings.ToLower(strings.TrimSpace(e))
	if e == "" {
		return sdk.OutputConfigEffortHigh
	}
	return sdk.OutputConfigEffort(e)
}

func toolParam(spec llm.ToolSpec) (sdk.ToolParam, error) {
	if spec.Name == "" {
		return sdk.ToolParam{}, invalidRequest("tool has no name")
	}
	schema := spec.InputSchema
	if len(schema) == 0 {
		schema = json.RawMessage(`{"type":"object"}`)
	}
	var probe map[string]any
	if err := json.Unmarshal(schema, &probe); err != nil {
		return sdk.ToolParam{}, invalidRequest(fmt.Sprintf("tool %q: input schema is not a JSON object: %v", spec.Name, err))
	}

	t := sdk.ToolParam{
		Name: spec.Name,
		// The schema is sent exactly as given rather than rebuilt from
		// the SDK's typed fields, so keywords the SDK doesn't model
		// (additionalProperties, $defs, ...) survive.
		InputSchema: param.Override[sdk.ToolInputSchemaParam](schema),
	}
	if spec.Description != "" {
		t.Description = param.NewOpt(spec.Description)
	}
	if spec.Strict {
		t.Strict = param.NewOpt(true)
	}
	return t, nil
}

func messageParam(m llm.Message) (sdk.MessageParam, error) {
	var role sdk.MessageParamRole
	switch m.Role {
	case llm.RoleUser:
		role = sdk.MessageParamRoleUser
	case llm.RoleAssistant:
		role = sdk.MessageParamRoleAssistant
	default:
		return sdk.MessageParam{}, invalidRequest(fmt.Sprintf("unknown role %q", m.Role))
	}
	mp := sdk.MessageParam{Role: role}
	for j, b := range m.Content {
		cb, err := contentBlockParam(b)
		if err != nil {
			return sdk.MessageParam{}, fmt.Errorf("block %d: %w", j, err)
		}
		mp.Content = append(mp.Content, cb)
	}
	return mp, nil
}

func contentBlockParam(b llm.Block) (sdk.ContentBlockParamUnion, error) {
	switch b.Type {
	case llm.BlockText:
		return sdk.ContentBlockParamUnion{OfText: &sdk.TextBlockParam{Text: b.Text}}, nil

	case llm.BlockImage:
		img, err := imageParam(b)
		if err != nil {
			return sdk.ContentBlockParamUnion{}, err
		}
		return sdk.ContentBlockParamUnion{OfImage: img}, nil

	case llm.BlockToolUse:
		input := b.Input
		if len(input) == 0 {
			input = json.RawMessage(`{}`)
		}
		return sdk.ContentBlockParamUnion{OfToolUse: &sdk.ToolUseBlockParam{
			ID:    b.ID,
			Name:  b.Name,
			Input: input,
		}}, nil

	case llm.BlockToolResult:
		tr := &sdk.ToolResultBlockParam{ToolUseID: b.ToolUseID}
		if b.IsError {
			tr.IsError = param.NewOpt(true)
		}
		for k, c := range b.Content {
			switch c.Type {
			case llm.BlockText:
				tr.Content = append(tr.Content, sdk.ToolResultBlockParamContentUnion{
					OfText: &sdk.TextBlockParam{Text: c.Text},
				})
			case llm.BlockImage:
				img, err := imageParam(c)
				if err != nil {
					return sdk.ContentBlockParamUnion{}, fmt.Errorf("tool_result content %d: %w", k, err)
				}
				tr.Content = append(tr.Content, sdk.ToolResultBlockParamContentUnion{OfImage: img})
			default:
				return sdk.ContentBlockParamUnion{}, invalidRequest(fmt.Sprintf("tool_result content %d: unsupported block type %q", k, c.Type))
			}
		}
		return sdk.ContentBlockParamUnion{OfToolResult: tr}, nil

	case llm.BlockThinking:
		if len(b.ProviderRaw) > 0 {
			// A redacted_thinking block (or any block this provider kept
			// raw): send back the exact bytes the API produced.
			return rawBlock(b.ProviderRaw)
		}
		return sdk.ContentBlockParamUnion{OfThinking: &sdk.ThinkingBlockParam{
			Thinking:  b.Text,
			Signature: b.Signature,
		}}, nil

	default:
		if len(b.ProviderRaw) > 0 {
			// A block type the contract doesn't model, preserved from an
			// earlier response by blockFromAPI.
			return rawBlock(b.ProviderRaw)
		}
		return sdk.ContentBlockParamUnion{}, invalidRequest(fmt.Sprintf("unsupported block type %q", b.Type))
	}
}

func rawBlock(raw json.RawMessage) (sdk.ContentBlockParamUnion, error) {
	var probe map[string]any
	if err := json.Unmarshal(raw, &probe); err != nil {
		return sdk.ContentBlockParamUnion{}, invalidRequest(fmt.Sprintf("provider_raw is not a JSON object: %v", err))
	}
	return param.Override[sdk.ContentBlockParamUnion](raw), nil
}

func imageParam(b llm.Block) (*sdk.ImageBlockParam, error) {
	if b.MediaType == "" {
		return nil, invalidRequest("image block has no media type")
	}
	return &sdk.ImageBlockParam{Source: sdk.ImageBlockParamSourceUnion{
		OfBase64: &sdk.Base64ImageSourceParam{
			Data:      base64.StdEncoding.EncodeToString(b.Data),
			MediaType: sdk.Base64ImageSourceMediaType(b.MediaType),
		},
	}}, nil
}

// messageFromAPI converts a complete API message into the assistant
// message the agent appends to the conversation.
func messageFromAPI(m *sdk.Message) (llm.Message, error) {
	out := llm.Message{Role: llm.RoleAssistant, Content: make([]llm.Block, 0, len(m.Content))}
	for i := range m.Content {
		b, err := blockFromAPI(&m.Content[i])
		if err != nil {
			return llm.Message{}, fmt.Errorf("content block %d: %w", i, err)
		}
		out.Content = append(out.Content, b)
	}
	return out, nil
}

func blockFromAPI(cb *sdk.ContentBlockUnion) (llm.Block, error) {
	switch cb.Type {
	case "text":
		return llm.TextBlock(cb.Text), nil

	case "thinking":
		return llm.Block{Type: llm.BlockThinking, Text: cb.Thinking, Signature: cb.Signature}, nil

	case "redacted_thinking":
		raw, err := json.Marshal(redactedThinking{Type: "redacted_thinking", Data: cb.Data})
		if err != nil {
			return llm.Block{}, err
		}
		return llm.Block{Type: llm.BlockThinking, ProviderRaw: raw}, nil

	case "tool_use":
		input := json.RawMessage(cb.Input)
		if !json.Valid(input) {
			// Only possible if the stream was cut inside the arguments
			// (stop_reason max_tokens). Keep the block well-formed so the
			// conversation can still be stored and sent back.
			input = nil
		}
		return llm.ToolUseBlock(cb.ID, cb.Name, input), nil

	default:
		// Not part of the llm contract (we declare no server tools, so
		// this should not happen). Keep it verbatim so it is never
		// dropped and can be echoed back on the next turn.
		raw := json.RawMessage(cb.RawJSON())
		if !json.Valid(raw) {
			return llm.Block{}, fmt.Errorf("unsupported block type %q without valid raw JSON", cb.Type)
		}
		return llm.Block{Type: llm.BlockType(cb.Type), ProviderRaw: raw}, nil
	}
}

// stopReason maps the API's stop reason to the llm contract's.
func stopReason(r sdk.StopReason, msg llm.Message) llm.StopReason {
	switch r {
	case sdk.StopReasonEndTurn, sdk.StopReasonStopSequence:
		return llm.StopEndTurn
	case sdk.StopReasonToolUse:
		return llm.StopToolUse
	case sdk.StopReasonMaxTokens, sdk.StopReasonModelContextWindowExceeded:
		return llm.StopMaxTokens
	case sdk.StopReasonRefusal:
		return llm.StopRefusal
	case sdk.StopReasonPauseTurn:
		return llm.StopPauseTurn
	}
	if len(msg.ToolUses()) > 0 {
		return llm.StopToolUse
	}
	return llm.StopEndTurn
}

func usageFromAPI(u sdk.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      int(u.InputTokens),
		OutputTokens:     int(u.OutputTokens),
		CacheReadTokens:  int(u.CacheReadInputTokens),
		CacheWriteTokens: int(u.CacheCreationInputTokens),
	}
}
