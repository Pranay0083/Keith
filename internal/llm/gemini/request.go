package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"github.com/Pranay0083/keith/internal/llm"
)

// Thinking budgets for Gemini 1.x/2.x models; see the package doc.
const (
	budgetLow    = 1024
	budgetMedium = 8192
	budgetHigh   = 24576
)

func invalidRequest(format string, args ...any) *llm.Error {
	return &llm.Error{Provider: Name, Kind: llm.ErrInvalidRequest, Message: fmt.Sprintf(format, args...)}
}

// legacyModel reports whether model is a Gemini 1.x or 2.x model, which
// take a thinking budget instead of a thinking level and do not accept
// media inside function responses.
func legacyModel(model string) bool {
	m := strings.ToLower(strings.TrimPrefix(model, "models/"))
	return strings.HasPrefix(m, "gemini-1") || strings.HasPrefix(m, "gemini-2")
}

func (p *Provider) buildRequest(req llm.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	if strings.TrimSpace(req.Model) == "" {
		return nil, nil, invalidRequest("model is required")
	}
	tc, err := thinkingConfig(req.Model, req.Effort)
	if err != nil {
		return nil, nil, err
	}
	cfg := &genai.GenerateContentConfig{ThinkingConfig: tc}
	if req.System != "" {
		cfg.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: req.System}}}
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.defaultMaxTokens
	}
	if maxTokens > 0 {
		cfg.MaxOutputTokens = clampInt32(maxTokens)
	}
	if cfg.Tools, err = tools(req.Tools); err != nil {
		return nil, nil, err
	}

	c := &converter{calls: map[string]callInfo{}, legacy: legacyModel(req.Model)}
	contents, err := c.contents(req.Messages)
	if err != nil {
		return nil, nil, err
	}
	return contents, cfg, nil
}

func thinkingConfig(model, effort string) (*genai.ThinkingConfig, error) {
	tc := &genai.ThinkingConfig{IncludeThoughts: true}
	if effort == "" {
		return tc, nil
	}
	if !legacyModel(model) {
		tc.ThinkingLevel = genai.ThinkingLevel(strings.ToUpper(effort))
		return tc, nil
	}
	var budget int32
	switch strings.ToLower(effort) {
	case llm.EffortLow:
		budget = budgetLow
	case llm.EffortMedium:
		budget = budgetMedium
	case llm.EffortHigh:
		budget = budgetHigh
	default:
		return nil, invalidRequest("effort %q is not supported for model %q (use low, medium or high)", effort, model)
	}
	tc.ThinkingBudget = &budget
	return tc, nil
}

func tools(specs []llm.ToolSpec) ([]*genai.Tool, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	decls := make([]*genai.FunctionDeclaration, 0, len(specs))
	for _, t := range specs {
		d := &genai.FunctionDeclaration{Name: t.Name, Description: t.Description}
		if len(t.InputSchema) > 0 {
			var schema any
			if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
				return nil, invalidRequest("tool %q: input schema is not valid JSON: %v", t.Name, err)
			}
			d.ParametersJsonSchema = schema
		}
		decls = append(decls, d)
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}, nil
}

// callInfo is what a tool_result needs from its tool_use.
type callInfo struct {
	name      string
	synthetic bool
}

// converter turns llm messages into genai contents. It remembers the
// tool_use blocks it has seen so later tool_results can name their
// function.
type converter struct {
	calls  map[string]callInfo
	legacy bool
}

func (c *converter) contents(msgs []llm.Message) ([]*genai.Content, error) {
	out := make([]*genai.Content, 0, len(msgs))
	for i, m := range msgs {
		var role string
		switch m.Role {
		case llm.RoleUser:
			role = genai.RoleUser
		case llm.RoleAssistant:
			role = genai.RoleModel
		default:
			return nil, invalidRequest("message %d: unknown role %q", i, m.Role)
		}
		parts, trailing, err := c.blockParts(m.Content)
		if err != nil {
			if e, ok := err.(*llm.Error); ok {
				e.Message = fmt.Sprintf("message %d: %s", i, e.Message)
			}
			return nil, err
		}
		parts = append(parts, trailing...)
		if len(parts) == 0 {
			continue
		}
		out = append(out, &genai.Content{Role: role, Parts: parts})
	}
	return out, nil
}

// blockParts converts one message's blocks. trailing holds parts that
// must follow all the others (images from tool results on legacy models).
func (c *converter) blockParts(blocks []llm.Block) (parts, trailing []*genai.Part, err error) {
	for j, b := range blocks {
		switch b.Type {
		case llm.BlockText, llm.BlockThinking:
			ps, err := textParts(b)
			if err != nil {
				return nil, nil, blockErr(j, err)
			}
			parts = append(parts, ps...)
		case llm.BlockImage:
			parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: b.MediaType, Data: b.Data}})
		case llm.BlockToolUse:
			part, err := c.functionCall(b)
			if err != nil {
				return nil, nil, blockErr(j, err)
			}
			parts = append(parts, part)
		case llm.BlockToolResult:
			part, extra, err := c.functionResponse(b)
			if err != nil {
				return nil, nil, blockErr(j, err)
			}
			parts = append(parts, part)
			trailing = append(trailing, extra...)
		default:
			return nil, nil, blockErr(j, invalidRequest("unknown block type %q", b.Type))
		}
	}
	return parts, trailing, nil
}

func blockErr(j int, err error) error {
	if e, ok := err.(*llm.Error); ok {
		e.Message = fmt.Sprintf("block %d: %s", j, e.Message)
	}
	return err
}

func parseRaw(b llm.Block) (rawData, error) {
	var r rawData
	if len(b.ProviderRaw) == 0 {
		return r, nil
	}
	if err := json.Unmarshal(b.ProviderRaw, &r); err != nil {
		return r, invalidRequest("%s block has corrupt provider_raw: %v", b.Type, err)
	}
	return r, nil
}

// textParts echoes a text or thinking block: one part per recorded
// segment, or one part from Text.
func textParts(b llm.Block) ([]*genai.Part, error) {
	thought := b.Type == llm.BlockThinking
	r, err := parseRaw(b)
	if err != nil {
		return nil, err
	}
	if len(r.Parts) > 0 {
		out := make([]*genai.Part, 0, len(r.Parts))
		for _, s := range r.Parts {
			out = append(out, &genai.Part{Text: s.Text, Thought: thought, ThoughtSignature: s.ThoughtSignature})
		}
		return out, nil
	}
	part := &genai.Part{Text: b.Text, Thought: thought}
	if thought && b.Signature != "" {
		sig, err := base64.StdEncoding.DecodeString(b.Signature)
		if err != nil {
			return nil, invalidRequest("thinking block signature is not base64: %v", err)
		}
		part.ThoughtSignature = sig
	}
	if part.Text == "" && len(part.ThoughtSignature) == 0 {
		return nil, nil
	}
	return []*genai.Part{part}, nil
}

func (c *converter) functionCall(b llm.Block) (*genai.Part, error) {
	if b.ID == "" {
		return nil, invalidRequest("tool_use block has no id")
	}
	r, err := parseRaw(b)
	if err != nil {
		return nil, err
	}
	var args map[string]any
	if len(b.Input) > 0 {
		if err := json.Unmarshal(b.Input, &args); err != nil {
			return nil, invalidRequest("tool_use %q: input is not a JSON object: %v", b.ID, err)
		}
	}
	c.calls[b.ID] = callInfo{name: b.Name, synthetic: r.SyntheticID}
	fc := &genai.FunctionCall{Name: b.Name, Args: args}
	if !r.SyntheticID {
		fc.ID = b.ID
	}
	return &genai.Part{FunctionCall: fc, ThoughtSignature: r.ThoughtSignature}, nil
}

func (c *converter) functionResponse(b llm.Block) (*genai.Part, []*genai.Part, error) {
	call, ok := c.calls[b.ToolUseID]
	if !ok {
		return nil, nil, invalidRequest("tool_result %q has no matching tool_use earlier in the conversation", b.ToolUseID)
	}
	var texts []string
	var media []*genai.FunctionResponsePart
	var trailing []*genai.Part
	for k, cb := range b.Content {
		switch cb.Type {
		case llm.BlockText:
			texts = append(texts, cb.Text)
		case llm.BlockImage:
			if c.legacy {
				trailing = append(trailing, &genai.Part{InlineData: &genai.Blob{MIMEType: cb.MediaType, Data: cb.Data}})
			} else {
				media = append(media, &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{MIMEType: cb.MediaType, Data: cb.Data}})
			}
		default:
			return nil, nil, invalidRequest("tool_result %q: content block %d has unsupported type %q", b.ToolUseID, k, cb.Type)
		}
	}
	key := "output"
	if b.IsError {
		key = "error"
	}
	fr := &genai.FunctionResponse{
		Name:     call.name,
		Response: map[string]any{key: strings.Join(texts, "\n")},
		Parts:    media,
	}
	if !call.synthetic {
		fr.ID = b.ToolUseID
	}
	return &genai.Part{FunctionResponse: fr}, trailing, nil
}
