package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/llm"
)

func TestNew(t *testing.T) {
	for _, key := range []string{"", "   "} {
		if _, err := New(key); err == nil {
			t.Errorf("New(%q) succeeded, want error", key)
		}
	}
	p, err := New(testKey)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Name() != "gemini" || Name != "gemini" {
		t.Errorf("Name() = %q, want gemini", p.Name())
	}
}

func TestGenerateAssemblesStreamAndEmitsEvents(t *testing.T) {
	usage := map[string]any{
		"promptTokenCount":        1200,
		"cachedContentTokenCount": 200,
		"candidatesTokenCount":    50,
		"thoughtsTokenCount":      30,
		"totalTokenCount":         1280,
	}
	f := newFake(t, sse(
		chunk(thought("Plan the ")),
		chunk(signed(thought("fix."), "sig-thought")),
		chunk(txt("Hello")),
		chunk(txt(" world")),
		chunk(signed(txt(""), "sig-text")),
		chunk(
			signed(call("call_1", "read_file", map[string]any{"path": "a.go"}), "sig-call"),
			call("", "list_dir", map[string]any{}),
		),
		finish("STOP", usage),
	))
	p := f.provider()

	var events []llm.StreamEvent
	resp, err := p.Generate(context.Background(), llm.Request{
		Model:    "gemini-3-pro-preview",
		Messages: []llm.Message{llm.UserText("go")},
	}, func(ev llm.StreamEvent) { events = append(events, ev) })
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	blocks := resp.Message.Content
	if resp.Message.Role != llm.RoleAssistant {
		t.Errorf("role = %q", resp.Message.Role)
	}
	if len(blocks) != 4 {
		t.Fatalf("got %d blocks, want 4: %+v", len(blocks), blocks)
	}

	// Thinking: two parts merged, segments kept because one is signed.
	th := blocks[0]
	if th.Type != llm.BlockThinking || th.Text != "Plan the fix." {
		t.Errorf("block 0 = %+v, want thinking %q", th, "Plan the fix.")
	}
	if want := base64.StdEncoding.EncodeToString([]byte("sig-thought")); th.Signature != want {
		t.Errorf("thinking Signature = %q, want %q", th.Signature, want)
	}
	if got, want := rawOf(t, th), (rawData{Parts: []rawPart{
		{Text: "Plan the "},
		{Text: "fix.", ThoughtSignature: []byte("sig-thought")},
	}}); !reflect.DeepEqual(got, want) {
		t.Errorf("thinking raw = %+v, want %+v", got, want)
	}

	// Text: unsigned parts merged into one segment, signed empty part kept.
	tx := blocks[1]
	if tx.Type != llm.BlockText || tx.Text != "Hello world" || tx.Signature != "" {
		t.Errorf("block 1 = %+v, want text %q", tx, "Hello world")
	}
	if got, want := rawOf(t, tx), (rawData{Parts: []rawPart{
		{Text: "Hello world"},
		{Text: "", ThoughtSignature: []byte("sig-text")},
	}}); !reflect.DeepEqual(got, want) {
		t.Errorf("text raw = %+v, want %+v", got, want)
	}
	if resp.Message.Text() != "Hello world" {
		t.Errorf("Message.Text() = %q", resp.Message.Text())
	}

	// Signed call with Gemini's ID.
	c1 := blocks[2]
	if c1.Type != llm.BlockToolUse || c1.ID != "call_1" || c1.Name != "read_file" {
		t.Errorf("block 2 = %+v", c1)
	}
	assertJSON(t, "call_1 input", c1.Input, `{"path":"a.go"}`)
	if got, want := rawOf(t, c1), (rawData{ThoughtSignature: []byte("sig-call")}); !reflect.DeepEqual(got, want) {
		t.Errorf("call_1 raw = %+v, want %+v", got, want)
	}

	// Unsigned call without ID: synthetic ID.
	c2 := blocks[3]
	if c2.Type != llm.BlockToolUse || c2.Name != "list_dir" || !strings.HasPrefix(c2.ID, "gemini_call_") {
		t.Errorf("block 3 = %+v", c2)
	}
	assertJSON(t, "list_dir input", c2.Input, `{}`)
	if got, want := rawOf(t, c2), (rawData{SyntheticID: true}); !reflect.DeepEqual(got, want) {
		t.Errorf("list_dir raw = %+v, want %+v", got, want)
	}

	if resp.StopReason != llm.StopToolUse {
		t.Errorf("StopReason = %q, want tool_use", resp.StopReason)
	}
	if want := (llm.Usage{InputTokens: 1000, OutputTokens: 80, CacheReadTokens: 200}); resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}

	wantEvents := []llm.StreamEvent{
		{Type: llm.StreamThinkingDelta, Text: "Plan the "},
		{Type: llm.StreamThinkingDelta, Text: "fix."},
		{Type: llm.StreamTextDelta, Text: "Hello"},
		{Type: llm.StreamTextDelta, Text: " world"},
		{Type: llm.StreamToolUseStart, ToolUseID: "call_1", ToolName: "read_file"},
		{Type: llm.StreamToolUseStart, ToolUseID: c2.ID, ToolName: "list_dir"},
	}
	if !reflect.DeepEqual(events, wantEvents) {
		t.Errorf("events =\n%+v\nwant\n%+v", events, wantEvents)
	}

	// Raw: one synthesized response with the parts as they will be echoed.
	var raw map[string]any
	if err := json.Unmarshal(resp.Raw, &raw); err != nil {
		t.Fatalf("Raw is not JSON: %v\n%s", err, resp.Raw)
	}
	if got := get(t, raw, "candidates", 0, "finishReason"); got != "STOP" {
		t.Errorf("Raw finishReason = %v", got)
	}
	assertJSON(t, "Raw content", get(t, raw, "candidates", 0, "content"), `{
		"role": "model",
		"parts": [
			{"text": "Plan the ", "thought": true},
			{"text": "fix.", "thought": true, "thoughtSignature": "`+b64("sig-thought")+`"},
			{"text": "Hello world"},
			{"thoughtSignature": "`+b64("sig-text")+`"},
			{"functionCall": {"id": "call_1", "name": "read_file", "args": {"path": "a.go"}}, "thoughtSignature": "`+b64("sig-call")+`"},
			{"functionCall": {"name": "list_dir"}}
		]
	}`)
	if get(t, raw, "usageMetadata", "promptTokenCount") != float64(1200) ||
		get(t, raw, "modelVersion") != "gemini-test-001" ||
		get(t, raw, "responseId") != "resp-abc" {
		t.Errorf("Raw metadata wrong: %s", resp.Raw)
	}
}

func TestUnsignedBlocksHaveNoProviderRaw(t *testing.T) {
	f := newFake(t, sse(
		chunk(thought("a"), thought("b")),
		chunk(txt("c")),
		chunk(txt("d")),
		finish("STOP", map[string]any{"promptTokenCount": 10, "candidatesTokenCount": 2}),
	))
	resp, err := f.provider().Generate(context.Background(), llm.Request{
		Model:    "gemini-3-flash",
		Messages: []llm.Message{llm.UserText("hi")},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	want := []llm.Block{
		{Type: llm.BlockThinking, Text: "ab"},
		{Type: llm.BlockText, Text: "cd"},
	}
	if !reflect.DeepEqual(resp.Message.Content, want) {
		t.Errorf("content = %+v, want %+v", resp.Message.Content, want)
	}
	if resp.StopReason != llm.StopEndTurn {
		t.Errorf("StopReason = %q", resp.StopReason)
	}
	if want := (llm.Usage{InputTokens: 10, OutputTokens: 2}); resp.Usage != want {
		t.Errorf("Usage = %+v", resp.Usage)
	}
}

func TestStopReasons(t *testing.T) {
	text := chunk(txt("partial"))
	tool := chunk(call("c1", "run", map[string]any{"cmd": "ls"}))
	blocked := mustJSON(map[string]any{
		"promptFeedback": map[string]any{"blockReason": "SAFETY"},
		"usageMetadata":  map[string]any{"promptTokenCount": 7},
		"modelVersion":   "gemini-test-001",
	})

	tests := []struct {
		name    string
		chunks  []string
		want    llm.StopReason
		wantErr llm.ErrorKind
	}{
		{"stop", []string{text, finish("STOP", nil)}, llm.StopEndTurn, ""},
		{"stop with tool call", []string{tool, finish("STOP", nil)}, llm.StopToolUse, ""},
		{"no finish reason", []string{text}, llm.StopEndTurn, ""},
		{"unspecified", []string{text, finish("FINISH_REASON_UNSPECIFIED", nil)}, llm.StopEndTurn, ""},
		{"other", []string{text, finish("OTHER", nil)}, llm.StopEndTurn, ""},
		{"other with tool call", []string{tool, finish("OTHER", nil)}, llm.StopToolUse, ""},
		{"max tokens", []string{text, finish("MAX_TOKENS", nil)}, llm.StopMaxTokens, ""},
		{"safety", []string{text, finish("SAFETY", nil)}, llm.StopRefusal, ""},
		{"recitation", []string{text, finish("RECITATION", nil)}, llm.StopRefusal, ""},
		{"blocklist", []string{finish("BLOCKLIST", nil)}, llm.StopRefusal, ""},
		{"prohibited", []string{finish("PROHIBITED_CONTENT", nil)}, llm.StopRefusal, ""},
		{"spii", []string{finish("SPII", nil)}, llm.StopRefusal, ""},
		{"image safety", []string{finish("IMAGE_SAFETY", nil)}, llm.StopRefusal, ""},
		{"prompt blocked", []string{blocked}, llm.StopRefusal, ""},
		{"malformed function call", []string{finish("MALFORMED_FUNCTION_CALL", nil)}, "", llm.ErrServer},
		{"unexpected tool call", []string{tool, finish("UNEXPECTED_TOOL_CALL", nil)}, "", llm.ErrServer},
		{"empty stream", nil, "", llm.ErrServer},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, sse(tc.chunks...))
			resp, err := f.provider().Generate(context.Background(), llm.Request{
				Model:    "gemini-3-pro-preview",
				Messages: []llm.Message{llm.UserText("hi")},
			}, nil)
			if tc.wantErr != "" {
				e := asLLMError(t, err)
				if e.Kind != tc.wantErr || !e.Retryable() {
					t.Errorf("error = %v, want retryable %s", e, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if resp.StopReason != tc.want {
				t.Errorf("StopReason = %q, want %q", resp.StopReason, tc.want)
			}
			if !json.Valid(resp.Raw) {
				t.Errorf("Raw is not valid JSON: %s", resp.Raw)
			}
			if resp.Message.Content == nil {
				t.Error("Message.Content is nil, want non-nil slice")
			}
		})
	}
}

func TestPromptBlockedRaw(t *testing.T) {
	f := newFake(t, sse(mustJSON(map[string]any{
		"promptFeedback": map[string]any{"blockReason": "PROHIBITED_CONTENT"},
		"usageMetadata":  map[string]any{"promptTokenCount": 7, "totalTokenCount": 7},
	})))
	resp, err := f.provider().Generate(context.Background(), llm.Request{
		Model:    "gemini-3-pro-preview",
		Messages: []llm.Message{llm.UserText("bad")},
	}, nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.StopReason != llm.StopRefusal || len(resp.Message.Content) != 0 {
		t.Errorf("got %q with %d blocks", resp.StopReason, len(resp.Message.Content))
	}
	if resp.Usage.InputTokens != 7 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
	assertJSON(t, "Raw", json.RawMessage(resp.Raw), `{
		"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
		"usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7}
	}`)
}

// TestMultiTurnRoundTrip runs a tool-use turn, stores the conversation as
// JSON the way agent_state does, and checks the next request echoes every
// signature and part boundary, names the function and omits synthetic IDs.
func TestMultiTurnRoundTrip(t *testing.T) {
	f := newFake(t,
		sse(
			chunk(thought("Need to read main.go.")),
			chunk(signed(call("", "read_file", map[string]any{"path": "main.go"}), "sig-fc")),
			chunk(txt("Checking.")),
			chunk(signed(txt(""), "sig-tail")),
			finish("STOP", map[string]any{"promptTokenCount": 20, "candidatesTokenCount": 5}),
		),
		sse(
			chunk(txt("Done.")),
			finish("STOP", map[string]any{"promptTokenCount": 40, "candidatesTokenCount": 2}),
		),
	)
	p := f.provider()
	ctx := context.Background()

	conv := []llm.Message{llm.UserText("fix it")}
	resp1, err := p.Generate(ctx, llm.Request{Model: "gemini-3-pro-preview", Messages: conv}, nil)
	if err != nil {
		t.Fatalf("Generate 1: %v", err)
	}
	if resp1.StopReason != llm.StopToolUse {
		t.Fatalf("StopReason = %q, want tool_use", resp1.StopReason)
	}
	uses := resp1.Message.ToolUses()
	if len(uses) != 1 || !strings.HasPrefix(uses[0].ID, syntheticIDPrefix) {
		t.Fatalf("tool uses = %+v", uses)
	}
	callID := uses[0].ID

	png := []byte("\x89PNG fake image")
	conv = append(conv, resp1.Message, llm.Message{Role: llm.RoleUser, Content: []llm.Block{
		llm.ToolResultBlock(callID, false, llm.TextBlock("package main"), llm.ImageBlock("image/png", png)),
	}})

	// Persist and reload, as agent_state.llm_messages does.
	stored, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	var reloaded []llm.Message
	if err := json.Unmarshal(stored, &reloaded); err != nil {
		t.Fatal(err)
	}

	resp2, err := p.Generate(ctx, llm.Request{Model: "gemini-3-pro-preview", Messages: reloaded}, nil)
	if err != nil {
		t.Fatalf("Generate 2: %v", err)
	}
	if resp2.Message.Text() != "Done." || resp2.StopReason != llm.StopEndTurn {
		t.Errorf("response 2 = %q / %q", resp2.Message.Text(), resp2.StopReason)
	}

	reqs := f.requests()
	if len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2", len(reqs))
	}
	assertJSON(t, "request 2 contents", reqs[1].Body["contents"], `[
		{"role": "user", "parts": [{"text": "fix it"}]},
		{"role": "model", "parts": [
			{"text": "Need to read main.go.", "thought": true},
			{"functionCall": {"name": "read_file", "args": {"path": "main.go"}}, "thoughtSignature": "`+b64("sig-fc")+`"},
			{"text": "Checking."},
			{"thoughtSignature": "`+b64("sig-tail")+`"}
		]},
		{"role": "user", "parts": [
			{"functionResponse": {
				"name": "read_file",
				"response": {"output": "package main"},
				"parts": [{"inlineData": {"mimeType": "image/png", "data": "`+base64.StdEncoding.EncodeToString(png)+`"}}]
			}}
		]}
	]`)
	if strings.Contains(string(reqs[1].RawBody), callID) {
		t.Errorf("synthetic id %q was sent to Gemini", callID)
	}
}
