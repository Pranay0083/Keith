package gemini

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"github.com/Pranay0083/keith/internal/llm"
)

// Name is the provider's stable ID, as stored in sessions.provider.
const Name = "gemini"

// defaultBaseURL is set explicitly so that GOOGLE_GEMINI_BASE_URL and
// genai.SetDefaultBaseURLs cannot redirect requests (and the API key).
const defaultBaseURL = "https://generativelanguage.googleapis.com/"

// syntheticIDPrefix starts the IDs generated for function calls that
// Gemini sent without one.
const syntheticIDPrefix = "gemini_call_"

// Option configures a Provider.
type Option func(*options)

type options struct {
	baseURL          string
	httpClient       *http.Client
	defaultMaxTokens int
}

// WithBaseURL sends requests to u instead of the Gemini API, for tests.
func WithBaseURL(u string) Option {
	return func(o *options) {
		if u != "" {
			o.baseURL = u
		}
	}
}

// WithHTTPClient sets the HTTP client. The Provider uses a shallow copy
// whose transport also records Retry-After headers of error responses.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		if c != nil {
			o.httpClient = c
		}
	}
}

// WithDefaultMaxTokens sets maxOutputTokens for requests whose MaxTokens
// is 0. Zero or less leaves Gemini's own default.
func WithDefaultMaxTokens(n int) Option {
	return func(o *options) {
		if n > 0 {
			o.defaultMaxTokens = n
		}
	}
}

// Provider is an llm.Provider backed by the Gemini API.
type Provider struct {
	client           *genai.Client
	apiKey           string
	defaultMaxTokens int
}

var _ llm.Provider = (*Provider)(nil)

// New returns a Provider that authenticates with apiKey. It fails if
// apiKey is empty or the genai client cannot be created. Environment
// variables never override the key, the backend or the base URL.
func New(apiKey string, opts ...Option) (*Provider, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, errors.New("gemini: API key is required")
	}
	o := options{baseURL: defaultBaseURL}
	for _, opt := range opts {
		opt(&o)
	}
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:      apiKey,
		Backend:     genai.BackendGeminiAPI,
		HTTPClient:  withHeaderCapture(o.httpClient),
		HTTPOptions: genai.HTTPOptions{BaseURL: o.baseURL},
	})
	if err != nil {
		// genai's errors may print the whole ClientConfig, key included,
		// so only a redacted string leaves this function.
		return nil, fmt.Errorf("gemini: create client: %s", redact(err.Error(), apiKey))
	}
	return &Provider{client: client, apiKey: apiKey, defaultMaxTokens: o.defaultMaxTokens}, nil
}

// Name implements llm.Provider.
func (p *Provider) Name() string { return Name }

// Generate implements llm.Provider.
func (p *Provider) Generate(ctx context.Context, req llm.Request, onEvent func(llm.StreamEvent)) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	contents, cfg, err := p.buildRequest(req)
	if err != nil {
		return nil, err
	}

	capture := &headerCapture{}
	streamCtx := context.WithValue(ctx, headerCaptureKey{}, capture)

	asm := &assembler{onEvent: onEvent}
	for chunk, err := range p.client.Models.GenerateContentStream(streamCtx, req.Model, contents, cfg) {
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			return nil, p.mapError(err, capture.retryAfter())
		}
		if err := asm.add(chunk); err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return asm.finish()
}

// rawData is the JSON stored in Block.ProviderRaw for Gemini blocks. Its
// shape is part of the agent_state storage format: add fields, never
// rename them.
type rawData struct {
	// ThoughtSignature is a tool_use block's functionCall signature.
	ThoughtSignature []byte `json:"thought_signature,omitempty"`
	// SyntheticID marks a tool_use ID generated here because Gemini sent
	// none; it is not sent back as functionCall.id or functionResponse.id.
	SyntheticID bool `json:"synthetic_id,omitempty"`
	// Parts are a text or thinking block's original part boundaries, set
	// only when at least one part carried a signature.
	Parts []rawPart `json:"parts,omitempty"`
}

// rawPart is one segment of a text or thinking block.
type rawPart struct {
	Text             string `json:"text"`
	ThoughtSignature []byte `json:"thought_signature,omitempty"`
}

// segment is a rawPart being assembled from streamed parts.
type segment struct {
	text strings.Builder
	sig  []byte
}

// textRun is a text or thinking block being assembled.
type textRun struct {
	thought bool
	segs    []*segment
	signed  bool
}

// assembler builds the Response from streamed chunks and emits stream
// events as parts arrive.
type assembler struct {
	onEvent func(llm.StreamEvent)

	blocks []llm.Block
	run    *textRun

	chunks         int
	sawCandidate   bool
	finishReason   genai.FinishReason
	finishMessage  string
	usage          *genai.GenerateContentResponseUsageMetadata
	modelVersion   string
	responseID     string
	promptFeedback *genai.GenerateContentResponsePromptFeedback
}

func (a *assembler) emit(ev llm.StreamEvent) {
	if a.onEvent != nil {
		a.onEvent(ev)
	}
}

func (a *assembler) add(chunk *genai.GenerateContentResponse) error {
	if chunk == nil {
		return nil
	}
	a.chunks++
	if chunk.UsageMetadata != nil {
		a.usage = chunk.UsageMetadata
	}
	if chunk.ModelVersion != "" {
		a.modelVersion = chunk.ModelVersion
	}
	if chunk.ResponseID != "" {
		a.responseID = chunk.ResponseID
	}
	if chunk.PromptFeedback != nil {
		a.promptFeedback = chunk.PromptFeedback
	}
	if len(chunk.Candidates) == 0 || chunk.Candidates[0] == nil {
		return nil
	}
	cand := chunk.Candidates[0]
	a.sawCandidate = true
	if cand.FinishReason != "" {
		a.finishReason = cand.FinishReason
		a.finishMessage = cand.FinishMessage
	}
	if cand.Content == nil {
		return nil
	}
	for _, part := range cand.Content.Parts {
		if err := a.addPart(part); err != nil {
			return err
		}
	}
	return nil
}

func (a *assembler) addPart(part *genai.Part) error {
	switch {
	case part == nil:
		return nil
	case part.FunctionCall != nil:
		return a.addFunctionCall(part.FunctionCall, part.ThoughtSignature)
	case part.InlineData != nil || part.FileData != nil || part.ExecutableCode != nil ||
		part.CodeExecutionResult != nil || part.FunctionResponse != nil ||
		part.ToolCall != nil || part.ToolResponse != nil:
		// Not produced without server-side tools or image output, which
		// this package never enables; see the package doc.
		return nil
	case part.Text == "" && len(part.ThoughtSignature) == 0:
		return nil
	}
	a.addText(part.Thought, part.Text, part.ThoughtSignature)
	return nil
}

func (a *assembler) addText(thought bool, text string, sig []byte) {
	if a.run == nil || a.run.thought != thought {
		a.flush()
		a.run = &textRun{thought: thought}
	}
	r := a.run
	n := len(r.segs)
	switch {
	case len(sig) > 0:
		s := &segment{sig: append([]byte(nil), sig...)}
		s.text.WriteString(text)
		r.segs = append(r.segs, s)
		r.signed = true
	case n > 0 && len(r.segs[n-1].sig) == 0:
		r.segs[n-1].text.WriteString(text)
	default:
		s := &segment{}
		s.text.WriteString(text)
		r.segs = append(r.segs, s)
	}
	if text == "" {
		return
	}
	if thought {
		a.emit(llm.StreamEvent{Type: llm.StreamThinkingDelta, Text: text})
	} else {
		a.emit(llm.StreamEvent{Type: llm.StreamTextDelta, Text: text})
	}
}

// flush closes the current text or thinking run into a block.
func (a *assembler) flush() {
	r := a.run
	if r == nil {
		return
	}
	a.run = nil

	b := llm.Block{Type: llm.BlockText}
	if r.thought {
		b.Type = llm.BlockThinking
	}
	var full strings.Builder
	parts := make([]rawPart, 0, len(r.segs))
	for _, s := range r.segs {
		t := s.text.String()
		full.WriteString(t)
		parts = append(parts, rawPart{Text: t, ThoughtSignature: s.sig})
	}
	b.Text = full.String()
	if r.signed {
		b.ProviderRaw = mustMarshalRaw(rawData{Parts: parts})
		if r.thought {
			for _, p := range parts {
				if len(p.ThoughtSignature) > 0 {
					b.Signature = base64.StdEncoding.EncodeToString(p.ThoughtSignature)
					break
				}
			}
		}
	}
	a.blocks = append(a.blocks, b)
}

func (a *assembler) addFunctionCall(fc *genai.FunctionCall, sig []byte) error {
	a.flush()
	id, synthetic := fc.ID, false
	if id == "" {
		id, synthetic = newCallID(), true
	}
	var input json.RawMessage
	if fc.Args != nil {
		b, err := json.Marshal(fc.Args)
		if err != nil {
			return &llm.Error{Provider: Name, Kind: llm.ErrServer, Message: fmt.Sprintf("encode arguments of function call %q: %v", fc.Name, err), Err: err}
		}
		input = b
	}
	block := llm.ToolUseBlock(id, fc.Name, input)
	if len(sig) > 0 || synthetic {
		block.ProviderRaw = mustMarshalRaw(rawData{ThoughtSignature: append([]byte(nil), sig...), SyntheticID: synthetic})
	}
	a.blocks = append(a.blocks, block)
	a.emit(llm.StreamEvent{Type: llm.StreamToolUseStart, ToolUseID: id, ToolName: fc.Name})
	return nil
}

func (a *assembler) finish() (*llm.Response, error) {
	a.flush()
	if a.chunks == 0 {
		return nil, &llm.Error{Provider: Name, Kind: llm.ErrServer, Message: "empty response stream"}
	}
	content := a.blocks
	if content == nil {
		content = []llm.Block{}
	}
	msg := llm.Message{Role: llm.RoleAssistant, Content: content}

	stop, err := a.stopReason(msg)
	if err != nil {
		return nil, err
	}
	return &llm.Response{
		Message:    msg,
		StopReason: stop,
		Usage:      mapUsage(a.usage),
		Raw:        a.raw(msg),
	}, nil
}

func (a *assembler) stopReason(msg llm.Message) (llm.StopReason, error) {
	switch a.finishReason {
	case genai.FinishReasonMalformedFunctionCall, genai.FinishReasonUnexpectedToolCall:
		m := fmt.Sprintf("model produced an unusable function call (%s)", a.finishReason)
		if a.finishMessage != "" {
			m += ": " + a.finishMessage
		}
		return "", &llm.Error{Provider: Name, Kind: llm.ErrServer, Message: m}
	case genai.FinishReasonMaxTokens, genai.FinishReasonContinuation:
		return llm.StopMaxTokens, nil
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonLanguage,
		genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent, genai.FinishReasonSPII,
		genai.FinishReasonImageSafety, genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return llm.StopRefusal, nil
	}
	if !a.sawCandidate && a.promptFeedback != nil && a.promptFeedback.BlockReason != "" {
		return llm.StopRefusal, nil
	}
	if len(msg.ToolUses()) > 0 {
		return llm.StopToolUse, nil
	}
	return llm.StopEndTurn, nil
}

// raw synthesizes one GenerateContentResponse holding the assembled
// message as it will be echoed; see the package doc.
func (a *assembler) raw(msg llm.Message) json.RawMessage {
	resp := &genai.GenerateContentResponse{
		ModelVersion:   a.modelVersion,
		ResponseID:     a.responseID,
		UsageMetadata:  a.usage,
		PromptFeedback: a.promptFeedback,
	}
	if a.sawCandidate || len(msg.Content) > 0 {
		cand := &genai.Candidate{FinishReason: a.finishReason, FinishMessage: a.finishMessage}
		if parts, _, err := (&converter{calls: map[string]callInfo{}}).blockParts(msg.Content); err == nil && len(parts) > 0 {
			cand.Content = &genai.Content{Role: genai.RoleModel, Parts: parts}
		}
		resp.Candidates = []*genai.Candidate{cand}
	}
	b, err := json.Marshal(resp)
	if err != nil {
		return nil
	}
	return b
}

func mapUsage(u *genai.GenerateContentResponseUsageMetadata) llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	in := int(u.PromptTokenCount) - int(u.CachedContentTokenCount) + int(u.ToolUsePromptTokenCount)
	return llm.Usage{
		InputTokens:     max(in, 0),
		OutputTokens:    int(u.CandidatesTokenCount) + int(u.ThoughtsTokenCount),
		CacheReadTokens: int(u.CachedContentTokenCount),
	}
}

func mustMarshalRaw(r rawData) json.RawMessage {
	b, err := json.Marshal(r)
	if err != nil {
		// rawData holds only strings, bytes and bools.
		panic(fmt.Sprintf("gemini: marshal provider raw: %v", err))
	}
	return b
}

func newCallID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("gemini: random call id: %v", err))
	}
	return syntheticIDPrefix + hex.EncodeToString(b[:])
}

func clampInt32(n int) int32 {
	if n > math.MaxInt32 {
		return math.MaxInt32
	}
	return int32(n)
}
