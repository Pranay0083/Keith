package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/llm"
)

// sse renders events in the API's text/event-stream shape.
func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(e), &head); err != nil {
			panic(fmt.Sprintf("bad event %s: %v", e, err))
		}
		fmt.Fprintf(&b, "event: %s\ndata: %s\n\n", head.Type, e)
	}
	return b.String()
}

// A recorded-shape stream: thinking with a signature, a redacted thinking
// block, text, then a tool call whose input arrives in two fragments.
var toolUseStream = sse(
	`{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":12,"cache_creation_input_tokens":100,"cache_read_input_tokens":2000,"output_tokens":1}}}`,
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`{"type":"ping"}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants "}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"the README."}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM"}}`,
	`{"type":"content_block_stop","index":0}`,
	`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix"}}`,
	`{"type":"content_block_stop","index":1}`,
	`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"I'll read"}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":" the file."}}`,
	`{"type":"content_block_stop","index":2}`,
	`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_01","name":"read_file","input":{}}}`,
	`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":""}}`,
	`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"READ"}}`,
	`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"ME.md\"}"}}`,
	`{"type":"content_block_stop","index":3}`,
	`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null,"stop_details":null},"usage":{"output_tokens":87}}`,
	`{"type":"message_stop"}`,
)

func textStream(text, stop string) string {
	return sse(
		`{"type":"message_start","message":{"id":"msg_02","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":50,"cache_creation_input_tokens":0,"cache_read_input_tokens":2100,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text),
		`{"type":"content_block_stop","index":0}`,
		fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%q,"stop_sequence":null},"usage":{"output_tokens":9}}`, stop),
		`{"type":"message_stop"}`,
	)
}

type captured struct {
	path   string
	header http.Header
	body   map[string]any
}

// fakeAPI serves one scripted reply per request and records what it got.
type fakeAPI struct {
	t       *testing.T
	srv     *httptest.Server
	mu      sync.Mutex
	replies []func(w http.ResponseWriter, r *http.Request)
	reqs    []captured
}

func newFakeAPI(t *testing.T, replies ...func(w http.ResponseWriter, r *http.Request)) *fakeAPI {
	f := &fakeAPI{t: t, replies: replies}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeAPI) serve(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		f.t.Errorf("request body is not JSON: %v\n%s", err, raw)
	}
	f.mu.Lock()
	f.reqs = append(f.reqs, captured{path: r.URL.Path, header: r.Header.Clone(), body: body})
	if len(f.replies) == 0 {
		f.mu.Unlock()
		f.t.Errorf("unexpected request %d", len(f.reqs))
		http.Error(w, "no reply scripted", http.StatusTeapot)
		return
	}
	reply := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	reply(w, r)
}

func (f *fakeAPI) requests() []captured {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]captured(nil), f.reqs...)
}

func (f *fakeAPI) provider(opts ...Option) *Provider {
	return New("test-key", append([]Option{WithBaseURL(f.srv.URL), WithMaxRetries(0)}, opts...)...)
}

func streamReply(body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("request-id", "req_test")
		io.WriteString(w, body)
	}
}

func errorReply(status int, header map[string]string, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("request-id", "req_err")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}
}

func errBody(typ, msg string) string {
	return fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q},"request_id":"req_err"}`, typ, msg)
}

func collect() (*[]llm.StreamEvent, func(llm.StreamEvent)) {
	var evs []llm.StreamEvent
	return &evs, func(e llm.StreamEvent) { evs = append(evs, e) }
}

func TestGenerateStreamsAndAssembles(t *testing.T) {
	// Environment credentials and base URL must not leak into a provider
	// built with an explicit key.
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")

	api := newFakeAPI(t, streamReply(toolUseStream))
	p := api.provider()
	if p.Name() != "anthropic" {
		t.Errorf("Name() = %q", p.Name())
	}

	req := baseRequest()
	req.CacheSystemAndTools = true
	evs, onEvent := collect()
	resp, err := p.Generate(context.Background(), req, onEvent)
	if err != nil {
		t.Fatal(err)
	}

	wantEvents := []llm.StreamEvent{
		{Type: llm.StreamThinkingDelta, Text: "The user wants "},
		{Type: llm.StreamThinkingDelta, Text: "the README."},
		{Type: llm.StreamTextDelta, Text: "I'll read"},
		{Type: llm.StreamTextDelta, Text: " the file."},
		{Type: llm.StreamToolUseStart, ToolUseID: "toolu_01", ToolName: "read_file"},
	}
	if !reflect.DeepEqual(*evs, wantEvents) {
		t.Errorf("events:\n got %+v\nwant %+v", *evs, wantEvents)
	}

	if resp.StopReason != llm.StopToolUse {
		t.Errorf("stop reason = %q", resp.StopReason)
	}
	wantUsage := llm.Usage{InputTokens: 12, OutputTokens: 87, CacheReadTokens: 2000, CacheWriteTokens: 100}
	if resp.Usage != wantUsage {
		t.Errorf("usage = %+v, want %+v", resp.Usage, wantUsage)
	}

	c := resp.Message.Content
	if resp.Message.Role != llm.RoleAssistant || len(c) != 4 {
		t.Fatalf("message = %+v", resp.Message)
	}
	if c[0].Type != llm.BlockThinking || c[0].Text != "The user wants the README." || c[0].Signature != "EqQBCgIYAhIM" {
		t.Errorf("thinking = %+v", c[0])
	}
	if c[1].Type != llm.BlockThinking {
		t.Errorf("redacted = %+v", c[1])
	}
	jsonEqual(t, mustDecode(t, c[1].ProviderRaw), `{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix"}`)
	if c[2].Type != llm.BlockText || c[2].Text != "I'll read the file." {
		t.Errorf("text = %+v", c[2])
	}
	if c[3].Type != llm.BlockToolUse || c[3].ID != "toolu_01" || c[3].Name != "read_file" {
		t.Errorf("tool_use = %+v", c[3])
	}
	jsonEqual(t, mustDecode(t, c[3].Input), `{"path":"README.md"}`)

	// Raw is the final message JSON.
	raw := mustDecode(t, resp.Raw)
	if at(t, raw, "id") != "msg_01" || at(t, raw, "stop_reason") != "tool_use" {
		t.Errorf("raw = %s", resp.Raw)
	}
	if got := at(t, raw, "content", 3, "input"); !reflect.DeepEqual(got, map[string]any{"path": "README.md"}) {
		t.Errorf("raw tool input = %v", got)
	}
	if got := at(t, raw, "usage", "output_tokens"); got != float64(87) {
		t.Errorf("raw output_tokens = %v", got)
	}

	// The outgoing request.
	reqs := api.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	r := reqs[0]
	if r.path != "/v1/messages" {
		t.Errorf("path = %q", r.path)
	}
	if got := r.header.Get("X-Api-Key"); got != "test-key" {
		t.Errorf("x-api-key = %q", got)
	}
	if got := r.header.Get("Authorization"); got != "" {
		t.Errorf("Authorization header leaked from env: %q", got)
	}
	if r.header.Get("Anthropic-Version") == "" {
		t.Errorf("anthropic-version header missing")
	}
	if at(t, r.body, "stream") != true {
		t.Errorf("request is not streaming")
	}
	if at(t, r.body, "model") != "claude-opus-5-5" {
		t.Errorf("model = %v", at(t, r.body, "model"))
	}
	jsonEqual(t, at(t, r.body, "thinking"), `{"type":"adaptive","display":"summarized"}`)
	jsonEqual(t, at(t, r.body, "output_config"), `{"effort":"high"}`)
	jsonEqual(t, at(t, r.body, "system", 0, "cache_control"), `{"type":"ephemeral"}`)
	jsonEqual(t, at(t, r.body, "tools", 1, "cache_control"), `{"type":"ephemeral"}`)
	if at(t, r.body, "tools", 0, "strict") != true {
		t.Errorf("strict not sent")
	}
}

func TestGenerateNilOnEvent(t *testing.T) {
	api := newFakeAPI(t, streamReply(textStream("Done.", "end_turn")))
	resp, err := api.provider().Generate(context.Background(), baseRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "Done." || resp.StopReason != llm.StopEndTurn {
		t.Errorf("resp = %+v", resp)
	}
}

func TestGenerateStopReasons(t *testing.T) {
	tests := []struct {
		api  string
		want llm.StopReason
	}{
		{"end_turn", llm.StopEndTurn},
		{"stop_sequence", llm.StopEndTurn},
		{"max_tokens", llm.StopMaxTokens},
		{"model_context_window_exceeded", llm.StopMaxTokens},
		{"refusal", llm.StopRefusal},
		{"pause_turn", llm.StopPauseTurn},
	}
	for _, tt := range tests {
		t.Run(tt.api, func(t *testing.T) {
			api := newFakeAPI(t, streamReply(textStream("x", tt.api)))
			resp, err := api.provider().Generate(context.Background(), baseRequest(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StopReason != tt.want {
				t.Errorf("stop reason = %q, want %q", resp.StopReason, tt.want)
			}
		})
	}
}

func TestGenerateErrors(t *testing.T) {
	tests := []struct {
		name       string
		reply      func(http.ResponseWriter, *http.Request)
		kind       llm.ErrorKind
		status     int
		retryAfter time.Duration
		msg        string
	}{
		{
			name:       "rate limited with retry-after seconds",
			reply:      errorReply(429, map[string]string{"retry-after": "17"}, errBody("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit")),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 17 * time.Second,
			msg:        "per-minute rate limit",
		},
		{
			name:       "rate limited with retry-after-ms",
			reply:      errorReply(429, map[string]string{"retry-after-ms": "1500", "retry-after": "2"}, errBody("rate_limit_error", "slow down")),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 1500 * time.Millisecond,
		},
		{
			name:   "overloaded",
			reply:  errorReply(529, nil, errBody("overloaded_error", "Overloaded")),
			kind:   llm.ErrOverloaded,
			status: 529,
		},
		{
			name:   "server error",
			reply:  errorReply(500, nil, errBody("api_error", "Internal server error")),
			kind:   llm.ErrServer,
			status: 500,
		},
		{
			name:   "bad gateway without a body",
			reply:  errorReply(502, nil, "<html>bad gateway</html>"),
			kind:   llm.ErrServer,
			status: 502,
		},
		{
			name:   "prompt too long",
			reply:  errorReply(400, nil, errBody("invalid_request_error", "prompt is too long: 1050000 tokens > 1000000 maximum")),
			kind:   llm.ErrContextTooLong,
			status: 400,
			msg:    "prompt is too long",
		},
		{
			name:   "request too large",
			reply:  errorReply(413, nil, errBody("request_too_large", "Request exceeds the maximum allowed number of bytes.")),
			kind:   llm.ErrContextTooLong,
			status: 413,
		},
		{
			name:   "unauthorized",
			reply:  errorReply(401, nil, errBody("authentication_error", "invalid x-api-key")),
			kind:   llm.ErrAuth,
			status: 401,
		},
		{
			name:   "forbidden",
			reply:  errorReply(403, nil, errBody("permission_error", "Your API key does not have permission to use the specified resource.")),
			kind:   llm.ErrAuth,
			status: 403,
		},
		{
			name:   "bad request",
			reply:  errorReply(400, nil, errBody("invalid_request_error", "messages: roles must alternate")),
			kind:   llm.ErrInvalidRequest,
			status: 400,
			msg:    "roles must alternate",
		},
		{
			name:   "not found",
			reply:  errorReply(404, nil, errBody("not_found_error", "model: claude-nope")),
			kind:   llm.ErrInvalidRequest,
			status: 404,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			api := newFakeAPI(t, tt.reply)
			_, err := api.provider().Generate(context.Background(), baseRequest(), nil)
			var le *llm.Error
			if !errors.As(err, &le) {
				t.Fatalf("err = %v (%T), want *llm.Error", err, err)
			}
			if le.Kind != tt.kind || le.StatusCode != tt.status || le.Provider != "anthropic" {
				t.Errorf("got kind %q status %d provider %q, want %q %d", le.Kind, le.StatusCode, le.Provider, tt.kind, tt.status)
			}
			if le.RetryAfter != tt.retryAfter {
				t.Errorf("RetryAfter = %v, want %v", le.RetryAfter, tt.retryAfter)
			}
			if tt.msg != "" && !strings.Contains(le.Message, tt.msg) {
				t.Errorf("Message = %q, want it to contain %q", le.Message, tt.msg)
			}
			if strings.Contains(err.Error(), "test-key") {
				t.Errorf("error leaks the API key: %v", err)
			}
			if le.Err == nil {
				t.Errorf("underlying SDK error not kept")
			}
			if n := len(api.requests()); n != 1 {
				t.Errorf("%d requests, want 1 (retries disabled)", n)
			}
		})
	}
}

func TestRetryAfterHTTPDate(t *testing.T) {
	now := time.Date(2026, 11, 3, 12, 0, 0, 0, time.UTC)
	h := http.Header{}
	h.Set("Retry-After", now.Add(30*time.Second).Format(http.TimeFormat))
	if got := retryAfter(h, now); got != 30*time.Second {
		t.Errorf("retryAfter = %v, want 30s", got)
	}
	h.Set("Retry-After", "soon")
	if got := retryAfter(h, now); got != 0 {
		t.Errorf("retryAfter(garbage) = %v, want 0", got)
	}
}

func TestGenerateErrorEventMidStream(t *testing.T) {
	body := sse(
		`{"type":"message_start","message":{"id":"msg_03","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
	) + "event: error\ndata: " + errBody("overloaded_error", "Overloaded") + "\n\n"
	api := newFakeAPI(t, streamReply(body))
	evs, onEvent := collect()
	_, err := api.provider().Generate(context.Background(), baseRequest(), onEvent)
	var le *llm.Error
	if !errors.As(err, &le) || le.Kind != llm.ErrOverloaded {
		t.Fatalf("err = %v, want overloaded *llm.Error", err)
	}
	if !le.Retryable() {
		t.Errorf("overloaded should be retryable")
	}
	if le.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 for an in-stream error", le.StatusCode)
	}
	if len(*evs) != 1 || (*evs)[0].Text != "Hel" {
		t.Errorf("events before the error = %+v", *evs)
	}
}

func TestGenerateTruncatedStream(t *testing.T) {
	body := sse(
		`{"type":"message_start","message":{"id":"msg_04","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
	)
	api := newFakeAPI(t, streamReply(body))
	_, err := api.provider().Generate(context.Background(), baseRequest(), nil)
	var le *llm.Error
	if !errors.As(err, &le) || le.Kind != llm.ErrServer || !le.Retryable() {
		t.Fatalf("err = %v, want retryable server *llm.Error", err)
	}
}

func TestGenerateTransportError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	_, err := New("test-key", WithBaseURL(url), WithMaxRetries(0)).Generate(context.Background(), baseRequest(), nil)
	var le *llm.Error
	if !errors.As(err, &le) || le.Kind != llm.ErrServer {
		t.Fatalf("err = %v, want server *llm.Error", err)
	}
}

func TestGenerateInvalidRequestNotSent(t *testing.T) {
	api := newFakeAPI(t)
	req := baseRequest()
	req.Model = ""
	_, err := api.provider().Generate(context.Background(), req, nil)
	var le *llm.Error
	if !errors.As(err, &le) || le.Kind != llm.ErrInvalidRequest {
		t.Fatalf("err = %v, want invalid_request", err)
	}
	if n := len(api.requests()); n != 0 {
		t.Errorf("%d requests sent for an invalid request", n)
	}
}

func TestGenerateContextCancelled(t *testing.T) {
	head := sse(
		`{"type":"message_start","message":{"id":"msg_05","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Working"}}`,
	)
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, head)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := api.provider().Generate(ctx, baseRequest(), func(e llm.StreamEvent) {
		if e.Type == llm.StreamTextDelta {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	// Already cancelled: nothing is sent.
	_, err = api.provider().Generate(ctx, baseRequest(), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if n := len(api.requests()); n != 1 {
		t.Errorf("%d requests, want 1", n)
	}
}

// TestMultiTurnToolRoundTrip sends the assistant's tool_use turn, stored
// as agent_state would store it, back with the user's tool_result, and
// checks the second request carries both exactly.
func TestMultiTurnToolRoundTrip(t *testing.T) {
	api := newFakeAPI(t,
		streamReply(toolUseStream),
		streamReply(textStream("The README says hello.", "end_turn")),
	)
	p := api.provider()

	req := baseRequest()
	req.CacheSystemAndTools = true
	req.Messages = []llm.Message{llm.UserText("Summarise the README.")}
	first, err := p.Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}

	// agent_state persists the conversation as JSON between turns.
	history := append(req.Messages, first.Message)
	stored, err := json.Marshal(history)
	if err != nil {
		t.Fatal(err)
	}
	var restored []llm.Message
	if err := json.Unmarshal(stored, &restored); err != nil {
		t.Fatal(err)
	}

	toolUse := restored[1].ToolUses()[0]
	req.Messages = append(restored, llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.ToolResultBlock(toolUse.ID, false,
			llm.TextBlock("# Keith\nhello"),
			llm.ImageBlock("image/png", pngBytes),
		),
	}})
	second, err := p.Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if second.StopReason != llm.StopEndTurn || second.Message.Text() != "The README says hello." {
		t.Errorf("second response = %+v", second)
	}
	if second.Usage.CacheReadTokens != 2100 {
		t.Errorf("cache read tokens = %d", second.Usage.CacheReadTokens)
	}

	reqs := api.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests", len(reqs))
	}
	jsonEqual(t, at(t, reqs[1].body, "messages"), `[
		{"role":"user","content":[{"type":"text","text":"Summarise the README."}]},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"The user wants the README.","signature":"EqQBCgIYAhIM"},
			{"type":"redacted_thinking","data":"EmwKAhgBEgy3va3pzix"},
			{"type":"text","text":"I'll read the file."},
			{"type":"tool_use","id":"toolu_01","name":"read_file","input":{"path":"README.md"}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_01","content":[
				{"type":"text","text":"# Keith\nhello"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"`+pngBase64+`"}}
			]}
		]}
	]`)
	// The system prompt and tools are byte-identical across turns, so the
	// cached prefix and the thinking blocks stay valid.
	if !reflect.DeepEqual(at(t, reqs[0].body, "system"), at(t, reqs[1].body, "system")) ||
		!reflect.DeepEqual(at(t, reqs[0].body, "tools"), at(t, reqs[1].body, "tools")) {
		t.Errorf("system or tools changed between turns")
	}
}
