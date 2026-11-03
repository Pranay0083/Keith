// Package anthropic is the llm.Provider for the Claude API, built on
// anthropic-sdk-go. SDK types never leave this package.
//
// Every request runs with adaptive thinking (display "summarized" by
// default, so thinking deltas carry readable text) and an explicit
// output_config.effort: Request.Effort as given, or "high" when empty
// (claude-opus-5-5's own default is medium). Responses are always
// streamed.
//
// Block mapping:
//
//	text        <-> text
//	image       <-> image with a base64 source
//	tool_use    <-> tool_use
//	tool_result  -> tool_result; its Content becomes nested text and
//	                image blocks
//	thinking    <-> thinking (Text, Signature), or redacted_thinking,
//	                which is kept whole in ProviderRaw. Both are sent back
//	                exactly as received.
//
// A response block of any other type (none are expected, since no server
// tools are declared) is kept with its API type and its raw JSON in
// ProviderRaw, and sent back verbatim.
//
// ToolSpec.InputSchema is sent unchanged and ToolSpec.Strict becomes the
// tool's strict flag. With Request.CacheSystemAndTools, cache_control
// breakpoints go on the system block and the last tool, and top-level
// automatic caching covers the growing conversation (see
// WithConversationCache).
package anthropic

import (
	"net/http"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Name is the provider ID stored in sessions.provider.
const Name = "anthropic"

// DefaultMaxTokens is the output cap used when Request.MaxTokens is 0. It
// leaves room for adaptive thinking, which counts toward max_tokens.
const DefaultMaxTokens = 64000

// Thinking display modes for WithThinkingDisplay.
const (
	ThinkingSummarized = "summarized"
	ThinkingOmitted    = "omitted"
)

// responseHeaderTimeout bounds the wait for response headers, as the SDK's
// own default client does. Streaming bodies are not limited by it.
const responseHeaderTimeout = 10 * time.Minute

type settings struct {
	baseURL           string
	httpClient        *http.Client
	maxRetries        int
	maxRetriesSet     bool
	defaultMaxTokens  int
	thinkingDisplay   string
	cacheConversation bool
}

// Option configures a Provider.
type Option func(*settings)

// WithBaseURL sends requests to u instead of https://api.anthropic.com.
func WithBaseURL(u string) Option { return func(s *settings) { s.baseURL = u } }

// WithHTTPClient sets the HTTP client.
func WithHTTPClient(c *http.Client) Option { return func(s *settings) { s.httpClient = c } }

// WithMaxRetries sets how many times the SDK retries 408, 409, 429, 5xx and
// connection errors before returning them (SDK default 2). Retries happen
// only before a stream starts.
func WithMaxRetries(n int) Option {
	return func(s *settings) { s.maxRetries, s.maxRetriesSet = n, true }
}

// WithDefaultMaxTokens sets the output cap used when Request.MaxTokens is 0.
func WithDefaultMaxTokens(n int) Option { return func(s *settings) { s.defaultMaxTokens = n } }

// WithThinkingDisplay sets thinking.display: ThinkingSummarized (default)
// or ThinkingOmitted, under which thinking blocks arrive with empty text.
// Thinking is billed the same either way.
func WithThinkingDisplay(d string) Option { return func(s *settings) { s.thinkingDisplay = d } }

// WithConversationCache turns top-level automatic caching on or off for
// requests with CacheSystemAndTools set (default on). With it, each turn
// of an agent loop reads the previous turn's conversation prefix from
// cache instead of paying full input price for it.
func WithConversationCache(on bool) Option { return func(s *settings) { s.cacheConversation = on } }

// Provider is the Claude API llm.Provider. It is safe for concurrent use.
type Provider struct {
	client sdk.Client
	cfg    settings
}

// New returns a Provider that authenticates with apiKey. It ignores the
// SDK's environment defaults (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
// ANTHROPIC_BASE_URL, profiles), so only the key and options given here
// are used.
func New(apiKey string, opts ...Option) *Provider {
	cfg := settings{
		defaultMaxTokens:  DefaultMaxTokens,
		thinkingDisplay:   ThinkingSummarized,
		cacheConversation: true,
	}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.defaultMaxTokens <= 0 {
		cfg.defaultMaxTokens = DefaultMaxTokens
	}
	if cfg.thinkingDisplay == "" {
		cfg.thinkingDisplay = ThinkingSummarized
	}

	hc := cfg.httpClient
	if hc == nil {
		hc = defaultHTTPClient()
	}
	ro := []option.RequestOption{
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(hc),
	}
	if cfg.baseURL != "" {
		ro = append(ro, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.maxRetriesSet {
		ro = append(ro, option.WithMaxRetries(cfg.maxRetries))
	}
	return &Provider{client: sdk.NewClient(ro...), cfg: cfg}
}

// Name implements llm.Provider.
func (p *Provider) Name() string { return Name }

func defaultHTTPClient() *http.Client {
	if t, ok := http.DefaultTransport.(*http.Transport); ok {
		t = t.Clone()
		t.ResponseHeaderTimeout = responseHeaderTimeout
		return &http.Client{Transport: t}
	}
	return &http.Client{}
}
