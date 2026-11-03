package anthropic

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/Pranay0083/keith/internal/llm"
)

var pngBytes = []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a}

const pngBase64 = "iVBORw0KGgo="

func defaultSettings() settings {
	return New("test-key").cfg
}

// wire marshals SDK params and decodes them into generic JSON, which is
// what the API sees.
func wire(t *testing.T, p sdk.MessageNewParams) map[string]any {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal params: %v\n%s", err, b)
	}
	return m
}

// at walks a decoded JSON value by object keys and array indexes.
func at(t *testing.T, v any, path ...any) any {
	t.Helper()
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := v.(map[string]any)
			if !ok {
				t.Fatalf("at %v: %T is not an object", path, v)
			}
			v, ok = m[k]
			if !ok {
				t.Fatalf("at %v: key %q missing in %v", path, k, m)
			}
		case int:
			a, ok := v.([]any)
			if !ok || k >= len(a) {
				t.Fatalf("at %v: index %d out of range in %v", path, k, v)
			}
			v = a[k]
		}
	}
	return v
}

func has(v any, key string) bool {
	m, ok := v.(map[string]any)
	if !ok {
		return false
	}
	_, ok = m[key]
	return ok
}

func jsonEqual(t *testing.T, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad want JSON: %v", err)
	}
	if !reflect.DeepEqual(got, w) {
		g, _ := json.Marshal(got)
		t.Errorf("JSON mismatch\n got: %s\nwant: %s", g, want)
	}
}

var readFileSchema = json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`)

func baseRequest() llm.Request {
	return llm.Request{
		Model:  "claude-opus-5-5",
		System: "You are Keith.",
		Tools: []llm.ToolSpec{
			{Name: "read_file", Description: "Read a file.", InputSchema: readFileSchema, Strict: true},
			{Name: "shell_exec", Description: "Run a command.", InputSchema: json.RawMessage(`{"type":"object","properties":{"cmd":{"type":"string"}}}`)},
		},
		Messages: []llm.Message{llm.UserText("hi")},
	}
}

func TestBuildParamsCacheBreakpoints(t *testing.T) {
	req := baseRequest()
	req.CacheSystemAndTools = true
	p, err := buildParams(req, defaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	w := wire(t, p)

	jsonEqual(t, at(t, w, "system"), `[{"type":"text","text":"You are Keith.","cache_control":{"type":"ephemeral"}}]`)
	if has(at(t, w, "tools", 0), "cache_control") {
		t.Errorf("first tool has cache_control; only the last tool should")
	}
	jsonEqual(t, at(t, w, "tools", 1, "cache_control"), `{"type":"ephemeral"}`)
	jsonEqual(t, at(t, w, "cache_control"), `{"type":"ephemeral"}`)

	// Without the flag there are no breakpoints anywhere.
	req.CacheSystemAndTools = false
	p, err = buildParams(req, defaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	w = wire(t, p)
	if has(at(t, w, "system", 0), "cache_control") || has(at(t, w, "tools", 1), "cache_control") || has(w, "cache_control") {
		t.Errorf("cache_control set without CacheSystemAndTools: %v", w)
	}
}

func TestBuildParamsConversationCacheOff(t *testing.T) {
	req := baseRequest()
	req.CacheSystemAndTools = true
	p, err := buildParams(req, New("k", WithConversationCache(false)).cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := wire(t, p)
	if has(w, "cache_control") {
		t.Errorf("top-level cache_control set with WithConversationCache(false)")
	}
	jsonEqual(t, at(t, w, "tools", 1, "cache_control"), `{"type":"ephemeral"}`)
}

func TestBuildParamsToolsStrictAndSchema(t *testing.T) {
	p, err := buildParams(baseRequest(), defaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	w := wire(t, p)
	jsonEqual(t, at(t, w, "tools", 0), `{
		"name": "read_file",
		"description": "Read a file.",
		"strict": true,
		"input_schema": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}
	}`)
	if has(at(t, w, "tools", 1), "strict") {
		t.Errorf("non-strict tool sent strict")
	}
	if has(at(t, w, "tools", 1), "eager_input_streaming") {
		t.Errorf("eager_input_streaming should stay off so strict validation holds")
	}
}

func TestBuildParamsThinkingAndEffort(t *testing.T) {
	tests := []struct {
		effort, want string
	}{
		{"", "high"},
		{llm.EffortLow, "low"},
		{llm.EffortMedium, "medium"},
		{llm.EffortHigh, "high"},
		{"xhigh", "xhigh"},
		{"MAX", "max"},
	}
	for _, tt := range tests {
		req := baseRequest()
		req.Effort = tt.effort
		p, err := buildParams(req, defaultSettings())
		if err != nil {
			t.Fatal(err)
		}
		w := wire(t, p)
		jsonEqual(t, at(t, w, "thinking"), `{"type":"adaptive","display":"summarized"}`)
		if got := at(t, w, "output_config", "effort"); got != tt.want {
			t.Errorf("effort %q: got %v, want %q", tt.effort, got, tt.want)
		}
		for _, banned := range []string{"temperature", "top_p", "top_k", "tool_choice"} {
			if has(w, banned) {
				t.Errorf("request sets %s, which claude-opus-5-5 rejects", banned)
			}
		}
	}

	p, err := buildParams(baseRequest(), New("k", WithThinkingDisplay(ThinkingOmitted)).cfg)
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, at(t, wire(t, p), "thinking"), `{"type":"adaptive","display":"omitted"}`)
}

func TestBuildParamsMaxTokens(t *testing.T) {
	req := baseRequest()
	p, _ := buildParams(req, defaultSettings())
	if got := at(t, wire(t, p), "max_tokens"); got != float64(DefaultMaxTokens) {
		t.Errorf("default max_tokens = %v, want %d", got, DefaultMaxTokens)
	}
	req.MaxTokens = 1234
	p, _ = buildParams(req, defaultSettings())
	if got := at(t, wire(t, p), "max_tokens"); got != float64(1234) {
		t.Errorf("max_tokens = %v, want 1234", got)
	}
	p, _ = buildParams(baseRequest(), New("k", WithDefaultMaxTokens(999)).cfg)
	if got := at(t, wire(t, p), "max_tokens"); got != float64(999) {
		t.Errorf("WithDefaultMaxTokens: max_tokens = %v, want 999", got)
	}
}

func TestBuildParamsBlocks(t *testing.T) {
	redacted := json.RawMessage(`{"type":"redacted_thinking","data":"EnCrYpTeD"}`)
	req := baseRequest()
	req.System = ""
	req.Tools = nil
	req.Messages = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Block{
			llm.TextBlock("look at this"),
			llm.ImageBlock("image/png", pngBytes),
		}},
		{Role: llm.RoleAssistant, Content: []llm.Block{
			{Type: llm.BlockThinking, Text: "I should read it.", Signature: "sig-abc"},
			{Type: llm.BlockThinking, ProviderRaw: redacted},
			llm.TextBlock("Reading."),
			llm.ToolUseBlock("toolu_1", "read_file", json.RawMessage(`{"path":"README.md"}`)),
			llm.ToolUseBlock("toolu_2", "list", nil),
		}},
		{Role: llm.RoleUser, Content: []llm.Block{
			llm.ToolResultBlock("toolu_1", false, llm.TextBlock("# Keith"), llm.ImageBlock("image/png", pngBytes)),
			llm.ToolResultBlock("toolu_2", true, llm.TextBlock("permission denied")),
		}},
	}
	p, err := buildParams(req, defaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	w := wire(t, p)
	if has(w, "system") || has(w, "tools") {
		t.Errorf("empty system/tools were sent: %v", w)
	}
	jsonEqual(t, at(t, w, "messages"), `[
		{"role":"user","content":[
			{"type":"text","text":"look at this"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"`+pngBase64+`"}}
		]},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"I should read it.","signature":"sig-abc"},
			{"type":"redacted_thinking","data":"EnCrYpTeD"},
			{"type":"text","text":"Reading."},
			{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"README.md"}},
			{"type":"tool_use","id":"toolu_2","name":"list","input":{}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_1","content":[
				{"type":"text","text":"# Keith"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"`+pngBase64+`"}}
			]},
			{"type":"tool_result","tool_use_id":"toolu_2","is_error":true,"content":[
				{"type":"text","text":"permission denied"}
			]}
		]}
	]`)
}

func TestBuildParamsErrors(t *testing.T) {
	tests := []struct {
		name string
		edit func(*llm.Request)
		want string
	}{
		{"no model", func(r *llm.Request) { r.Model = "" }, "no model"},
		{"no messages", func(r *llm.Request) { r.Messages = nil }, "no messages"},
		{"bad role", func(r *llm.Request) { r.Messages[0].Role = "system" }, "unknown role"},
		{"bad schema", func(r *llm.Request) { r.Tools[0].InputSchema = json.RawMessage(`[1]`) }, "not a JSON object"},
		{"unknown block", func(r *llm.Request) { r.Messages[0].Content[0].Type = "video" }, "unsupported block type"},
		{"bad tool_result content", func(r *llm.Request) {
			r.Messages[0].Content = []llm.Block{llm.ToolResultBlock("t", false, llm.ToolUseBlock("x", "y", nil))}
		}, "tool_result content 0"},
		{"image without media type", func(r *llm.Request) {
			r.Messages[0].Content = []llm.Block{llm.ImageBlock("", pngBytes)}
		}, "media type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := baseRequest()
			tt.edit(&req)
			_, err := buildParams(req, defaultSettings())
			var le *llm.Error
			if !errors.As(err, &le) || le.Kind != llm.ErrInvalidRequest {
				t.Fatalf("err = %v, want *llm.Error invalid_request", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

func apiMessage(t *testing.T, raw string) *sdk.Message {
	t.Helper()
	var m sdk.Message
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return &m
}

func TestMessageFromAPI(t *testing.T) {
	m := apiMessage(t, `{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5",
		"content":[
			{"type":"thinking","thinking":"plan","signature":"sig-1"},
			{"type":"redacted_thinking","data":"opaque"},
			{"type":"text","text":"Hello"},
			{"type":"tool_use","id":"toolu_9","name":"read_file","input":{"path":"a.go"}},
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}}
		],
		"stop_reason":"tool_use","stop_sequence":null,
		"usage":{"input_tokens":10,"output_tokens":20,"cache_read_input_tokens":30,"cache_creation_input_tokens":40}
	}`)
	got, err := messageFromAPI(m)
	if err != nil {
		t.Fatal(err)
	}
	if got.Role != llm.RoleAssistant || len(got.Content) != 5 {
		t.Fatalf("got %+v", got)
	}
	if b := got.Content[0]; b.Type != llm.BlockThinking || b.Text != "plan" || b.Signature != "sig-1" || b.ProviderRaw != nil {
		t.Errorf("thinking block = %+v", b)
	}
	if b := got.Content[1]; b.Type != llm.BlockThinking || b.Text != "" {
		t.Errorf("redacted block = %+v", b)
	} else {
		jsonEqual(t, mustDecode(t, b.ProviderRaw), `{"type":"redacted_thinking","data":"opaque"}`)
	}
	if b := got.Content[2]; !reflect.DeepEqual(b, llm.TextBlock("Hello")) {
		t.Errorf("text block = %+v", b)
	}
	if b := got.Content[3]; b.Type != llm.BlockToolUse || b.ID != "toolu_9" || b.Name != "read_file" {
		t.Errorf("tool_use block = %+v", b)
	} else {
		jsonEqual(t, mustDecode(t, b.Input), `{"path":"a.go"}`)
	}
	if b := got.Content[4]; b.Type != "server_tool_use" {
		t.Errorf("unknown block = %+v", b)
	} else {
		jsonEqual(t, mustDecode(t, b.ProviderRaw), `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}}`)
	}

	if u := usageFromAPI(m.Usage); u != (llm.Usage{InputTokens: 10, OutputTokens: 20, CacheReadTokens: 30, CacheWriteTokens: 40}) {
		t.Errorf("usage = %+v", u)
	}

	// The converted message goes back out unchanged, including the
	// redacted and unknown blocks.
	req := baseRequest()
	req.Messages = append(req.Messages, got, llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.ToolResultBlock("toolu_9", false, llm.TextBlock("package a")),
	}})
	p, err := buildParams(req, defaultSettings())
	if err != nil {
		t.Fatal(err)
	}
	jsonEqual(t, at(t, wire(t, p), "messages", 1, "content"), `[
		{"type":"thinking","thinking":"plan","signature":"sig-1"},
		{"type":"redacted_thinking","data":"opaque"},
		{"type":"text","text":"Hello"},
		{"type":"tool_use","id":"toolu_9","name":"read_file","input":{"path":"a.go"}},
		{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}}
	]`)
}

func TestMessageFromAPITruncatedToolInput(t *testing.T) {
	m := apiMessage(t, `{"id":"m","type":"message","role":"assistant","model":"x","content":[
		{"type":"tool_use","id":"toolu_1","name":"write_file","input":{}}
	],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":1}}`)
	m.Content[0].Input = json.RawMessage(`{"path":"a.go","body":"unterminated`)
	got, err := messageFromAPI(m)
	if err != nil {
		t.Fatal(err)
	}
	if in := string(got.Content[0].Input); in != `{}` {
		t.Errorf("truncated input = %s, want {}", in)
	}
	if _, err := json.Marshal(got); err != nil {
		t.Errorf("message must stay storable: %v", err)
	}
}

func TestStopReason(t *testing.T) {
	withTool := llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{llm.ToolUseBlock("t", "n", nil)}}
	textOnly := llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{llm.TextBlock("x")}}
	tests := []struct {
		in   sdk.StopReason
		msg  llm.Message
		want llm.StopReason
	}{
		{"end_turn", textOnly, llm.StopEndTurn},
		{"stop_sequence", textOnly, llm.StopEndTurn},
		{"tool_use", withTool, llm.StopToolUse},
		{"max_tokens", textOnly, llm.StopMaxTokens},
		{"model_context_window_exceeded", textOnly, llm.StopMaxTokens},
		{"refusal", textOnly, llm.StopRefusal},
		{"pause_turn", textOnly, llm.StopPauseTurn},
		{"", withTool, llm.StopToolUse},
		{"something_new", textOnly, llm.StopEndTurn},
	}
	for _, tt := range tests {
		if got := stopReason(tt.in, tt.msg); got != tt.want {
			t.Errorf("stopReason(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func mustDecode(t *testing.T, raw json.RawMessage) any {
	t.Helper()
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	return v
}
