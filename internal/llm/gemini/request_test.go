package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/Pranay0083/keith/internal/llm"
)

var okStream = sse(chunk(txt("ok")), finish("STOP", nil))

func TestRequestMapping(t *testing.T) {
	f := newFake(t, okStream)
	png := []byte("\x89PNG user image")
	jpg := []byte("\xff\xd8 tool image")
	schema := `{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"],"additionalProperties":false}`

	_, err := f.provider().Generate(context.Background(), llm.Request{
		Model:  "gemini-3-pro-preview",
		System: "You are Keith.",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.TextBlock("Fix the bug"),
				llm.ImageBlock("image/png", png),
			}},
			{Role: llm.RoleAssistant, Content: []llm.Block{
				{Type: llm.BlockThinking, Text: "hmm", Signature: base64.StdEncoding.EncodeToString([]byte("plain-sig"))},
				llm.TextBlock("Reading."),
				llm.ToolUseBlock("call_9", "read_file", json.RawMessage(`{"path":"main.go"}`)),
				llm.ToolUseBlock("call_10", "grep", json.RawMessage(`{"q":"x"}`)),
			}},
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.ToolResultBlock("call_9", false, llm.TextBlock("line1"), llm.TextBlock("line2"), llm.ImageBlock("image/jpeg", jpg)),
				llm.ToolResultBlock("call_10", true, llm.TextBlock("boom")),
			}},
		},
		Tools: []llm.ToolSpec{
			{Name: "read_file", Description: "Read a file", InputSchema: json.RawMessage(schema), Strict: true},
			{Name: "grep", Description: "Search", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)},
		},
		MaxTokens:           4096,
		Effort:              llm.EffortHigh,
		CacheSystemAndTools: true,
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	reqs := f.requests()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	r := reqs[0]
	if r.Method != http.MethodPost || r.Path != "/v1beta/models/gemini-3-pro-preview:streamGenerateContent" || r.RawQuery != "alt=sse" {
		t.Errorf("request = %s %s?%s", r.Method, r.Path, r.RawQuery)
	}
	if r.APIKey != testKey {
		t.Errorf("x-goog-api-key = %q, want the key given to New", r.APIKey)
	}

	assertJSON(t, "systemInstruction", r.Body["systemInstruction"], `{"role":"user","parts":[{"text":"You are Keith."}]}`)
	assertJSON(t, "contents", r.Body["contents"], `[
		{"role": "user", "parts": [
			{"text": "Fix the bug"},
			{"inlineData": {"mimeType": "image/png", "data": "`+base64.StdEncoding.EncodeToString(png)+`"}}
		]},
		{"role": "model", "parts": [
			{"text": "hmm", "thought": true, "thoughtSignature": "`+b64("plain-sig")+`"},
			{"text": "Reading."},
			{"functionCall": {"id": "call_9", "name": "read_file", "args": {"path": "main.go"}}},
			{"functionCall": {"id": "call_10", "name": "grep", "args": {"q": "x"}}}
		]},
		{"role": "user", "parts": [
			{"functionResponse": {
				"id": "call_9",
				"name": "read_file",
				"response": {"output": "line1\nline2"},
				"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": "`+base64.StdEncoding.EncodeToString(jpg)+`"}}]
			}},
			{"functionResponse": {"id": "call_10", "name": "grep", "response": {"error": "boom"}}}
		]}
	]`)
	assertJSON(t, "tools", r.Body["tools"], `[{"functionDeclarations": [
		{"name": "read_file", "description": "Read a file", "parametersJsonSchema": `+schema+`},
		{"name": "grep", "description": "Search", "parametersJsonSchema": {"type":"object","properties":{"q":{"type":"string"}}}}
	]}]`)
	assertJSON(t, "generationConfig", r.Body["generationConfig"], `{
		"maxOutputTokens": 4096,
		"thinkingConfig": {"includeThoughts": true, "thinkingLevel": "HIGH"}
	}`)
	for _, k := range []string{"cachedContent", "toolConfig", "safetySettings"} {
		if _, ok := r.Body[k]; ok {
			t.Errorf("unexpected %q in request", k)
		}
	}
	if strings.Contains(string(r.RawBody), "strict") {
		t.Errorf("Strict leaked into the request: %s", r.RawBody)
	}
}

func TestThinkingConfigAndMaxTokens(t *testing.T) {
	tests := []struct {
		model, effort string
		opts          []Option
		maxTokens     int
		wantConfig    string // generationConfig JSON
	}{
		{"gemini-3-pro-preview", "", nil, 0, `{"thinkingConfig":{"includeThoughts":true}}`},
		{"gemini-3-pro-preview", "low", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"LOW"}}`},
		{"gemini-3-flash", "medium", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"MEDIUM"}}`},
		{"gemini-3-pro-preview", "high", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"HIGH"}}`},
		{"gemini-3-flash", "minimal", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"MINIMAL"}}`},
		{"gemini-2.5-pro", "", nil, 0, `{"thinkingConfig":{"includeThoughts":true}}`},
		{"gemini-2.5-pro", "low", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":1024}}`},
		{"gemini-2.5-flash", "medium", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":8192}}`},
		{"models/gemini-2.5-flash", "high", nil, 0, `{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":24576}}`},
		{"gemini-3-pro-preview", "", []Option{WithDefaultMaxTokens(2048)}, 0, `{"maxOutputTokens":2048,"thinkingConfig":{"includeThoughts":true}}`},
		{"gemini-3-pro-preview", "", []Option{WithDefaultMaxTokens(2048)}, 512, `{"maxOutputTokens":512,"thinkingConfig":{"includeThoughts":true}}`},
	}
	for _, tc := range tests {
		t.Run(tc.model+"/"+tc.effort, func(t *testing.T) {
			f := newFake(t, okStream)
			_, err := f.provider(tc.opts...).Generate(context.Background(), llm.Request{
				Model:     tc.model,
				Messages:  []llm.Message{llm.UserText("hi")},
				Effort:    tc.effort,
				MaxTokens: tc.maxTokens,
			}, nil)
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			assertJSON(t, "generationConfig", f.lastBody()["generationConfig"], tc.wantConfig)
		})
	}
}

func TestLegacyModelSendsToolImagesAsTrailingParts(t *testing.T) {
	f := newFake(t, okStream)
	img := []byte("shot")
	_, err := f.provider().Generate(context.Background(), llm.Request{
		Model: "gemini-2.5-flash",
		Messages: []llm.Message{
			llm.UserText("look"),
			{Role: llm.RoleAssistant, Content: []llm.Block{llm.ToolUseBlock("c1", "screenshot", nil)}},
			{Role: llm.RoleUser, Content: []llm.Block{
				llm.ToolResultBlock("c1", false, llm.TextBlock("captured"), llm.ImageBlock("image/png", img)),
				llm.TextBlock("what do you see?"),
			}},
		},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	assertJSON(t, "tool result turn", get(t, f.lastBody(), "contents", 2), `{"role": "user", "parts": [
		{"functionResponse": {"id": "c1", "name": "screenshot", "response": {"output": "captured"}}},
		{"text": "what do you see?"},
		{"inlineData": {"mimeType": "image/png", "data": "`+base64.StdEncoding.EncodeToString(img)+`"}}
	]}`)
}

func TestEmptyMessagesAndBlocksAreSkipped(t *testing.T) {
	f := newFake(t, okStream)
	_, err := f.provider().Generate(context.Background(), llm.Request{
		Model: "gemini-3-pro-preview",
		Messages: []llm.Message{
			llm.UserText("first"),
			{Role: llm.RoleAssistant, Content: []llm.Block{}},
			{Role: llm.RoleUser, Content: []llm.Block{llm.TextBlock(""), llm.TextBlock("second")}},
		},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	assertJSON(t, "contents", f.lastBody()["contents"], `[
		{"role": "user", "parts": [{"text": "first"}]},
		{"role": "user", "parts": [{"text": "second"}]}
	]`)
}

func TestInvalidRequestsFailBeforeSending(t *testing.T) {
	user := llm.UserText("hi")
	tests := []struct {
		name string
		req  llm.Request
	}{
		{"no model", llm.Request{Messages: []llm.Message{user}}},
		{"unknown block type", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Block{{Type: "video"}}},
		}}},
		{"unknown role", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			{Role: "system", Content: []llm.Block{llm.TextBlock("x")}},
		}}},
		{"tool result without tool use", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			user,
			{Role: llm.RoleUser, Content: []llm.Block{llm.ToolResultBlock("missing", false, llm.TextBlock("x"))}},
		}}},
		{"tool use input not an object", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			user,
			{Role: llm.RoleAssistant, Content: []llm.Block{llm.ToolUseBlock("c1", "f", json.RawMessage(`[1,2]`))}},
		}}},
		{"corrupt provider raw", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			user,
			{Role: llm.RoleAssistant, Content: []llm.Block{{Type: llm.BlockText, Text: "x", ProviderRaw: json.RawMessage(`{"parts":"nope"}`)}}},
		}}},
		{"bad thinking signature", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{
			user,
			{Role: llm.RoleAssistant, Content: []llm.Block{{Type: llm.BlockThinking, Text: "x", Signature: "%%%"}}},
		}}},
		{"bad tool schema", llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{user},
			Tools: []llm.ToolSpec{{Name: "f", InputSchema: json.RawMessage(`{nope`)}}}},
		{"unsupported effort on 2.5", llm.Request{Model: "gemini-2.5-pro", Messages: []llm.Message{user}, Effort: "minimal"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, okStream)
			_, err := f.provider().Generate(context.Background(), tc.req, nil)
			e := asLLMError(t, err)
			if e.Kind != llm.ErrInvalidRequest || e.Retryable() {
				t.Errorf("error = %v, want invalid_request", e)
			}
			if n := len(f.requests()); n != 0 {
				t.Errorf("sent %d requests, want 0", n)
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestEnvironmentIsIgnored sets every variable genai reads and checks the
// request still goes to the Gemini API with the key given to New.
func TestEnvironmentIsIgnored(t *testing.T) {
	for k, v := range map[string]string{
		"GOOGLE_API_KEY":              "env-google-key",
		"GEMINI_API_KEY":              "env-gemini-key",
		"GOOGLE_GENAI_USE_VERTEXAI":   "true",
		"GOOGLE_GENAI_USE_ENTERPRISE": "true",
		"GOOGLE_CLOUD_PROJECT":        "env-project",
		"GOOGLE_CLOUD_LOCATION":       "us-central1",
		"GOOGLE_CLOUD_REGION":         "us-central1",
		"GOOGLE_GEMINI_BASE_URL":      "http://127.0.0.1:1/",
		"GOOGLE_VERTEX_BASE_URL":      "http://127.0.0.1:1/",
	} {
		t.Setenv(k, v)
	}

	var mu sync.Mutex
	var got *http.Request
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		mu.Lock()
		got = r
		mu.Unlock()
		body := "data: " + chunk(txt("ok")) + "\n\ndata: " + finish("STOP", nil) + "\n\n"
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(body)),
			Request:    r,
		}, nil
	})
	hc := &http.Client{Transport: transport}

	p, err := New(testKey, WithHTTPClient(hc))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Generate(context.Background(), llm.Request{
		Model:    "gemini-3-pro-preview",
		Messages: []llm.Message{llm.UserText("hi")},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Message.Text() != "ok" {
		t.Errorf("text = %q", resp.Message.Text())
	}

	mu.Lock()
	defer mu.Unlock()
	if got == nil {
		t.Fatal("no request was sent")
	}
	if got.URL.Scheme != "https" || got.URL.Host != "generativelanguage.googleapis.com" ||
		got.URL.Path != "/v1beta/models/gemini-3-pro-preview:streamGenerateContent" {
		t.Errorf("request URL = %s", got.URL)
	}
	if k := got.Header.Get("x-goog-api-key"); k != testKey {
		t.Errorf("x-goog-api-key = %q, want the key given to New", k)
	}
	if a := got.Header.Get("Authorization"); a != "" {
		t.Errorf("unexpected Authorization header %q", a)
	}
	if _, ok := hc.Transport.(roundTripFunc); !ok {
		t.Errorf("WithHTTPClient modified the caller's client: transport is %T", hc.Transport)
	}
}
