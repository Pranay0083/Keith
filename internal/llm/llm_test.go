package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
)

func TestMessageJSONRoundTripKeepsProviderData(t *testing.T) {
	raw := json.RawMessage(`{"type":"redacted_thinking","data":"EuYBCkQ..."}`)
	in := []Message{
		UserText("fix the bug"),
		{Role: RoleAssistant, Content: []Block{
			{Type: BlockThinking, Text: "look at main.go", Signature: "sig-abc"},
			{Type: BlockThinking, ProviderRaw: raw},
			TextBlock("Reading it."),
			ToolUseBlock("call_1", "read_file", json.RawMessage(`{"path":"main.go"}`)),
		}},
		{Role: RoleUser, Content: []Block{
			ToolResultBlock("call_1", true, TextBlock("no such file"), ImageBlock("image/png", []byte{0x89, 'P', 'N', 'G'})),
		}},
	}

	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out []Message
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}

	th := out[1].Content[0]
	if th.Type != BlockThinking || th.Text != "look at main.go" || th.Signature != "sig-abc" {
		t.Errorf("thinking block = %+v", th)
	}
	if got := out[1].Content[1].ProviderRaw; !bytes.Equal(got, raw) {
		t.Errorf("ProviderRaw = %s, want %s unchanged", got, raw)
	}
	tu := out[1].Content[3]
	if tu.ID != "call_1" || tu.Name != "read_file" || string(tu.Input) != `{"path":"main.go"}` {
		t.Errorf("tool_use block = %+v", tu)
	}
	tr := out[2].Content[0]
	if tr.ToolUseID != "call_1" || !tr.IsError || len(tr.Content) != 2 {
		t.Fatalf("tool_result block = %+v", tr)
	}
	if img := tr.Content[1]; img.MediaType != "image/png" || !bytes.Equal(img.Data, []byte{0x89, 'P', 'N', 'G'}) {
		t.Errorf("image block = %+v", img)
	}
}

func TestToolUseBlockDefaultsInput(t *testing.T) {
	if got := string(ToolUseBlock("a", "b", nil).Input); got != "{}" {
		t.Errorf("Input = %q, want {}", got)
	}
}

func TestMessageHelpers(t *testing.T) {
	m := Message{Role: RoleAssistant, Content: []Block{
		{Type: BlockThinking, Text: "hidden"},
		TextBlock("one"),
		ToolUseBlock("c1", "search", nil),
		TextBlock("two"),
		ToolUseBlock("c2", "read_file", nil),
	}}
	if got := m.Text(); got != "one\ntwo" {
		t.Errorf("Text() = %q", got)
	}
	uses := m.ToolUses()
	if len(uses) != 2 || uses[0].ID != "c1" || uses[1].ID != "c2" {
		t.Errorf("ToolUses() = %+v", uses)
	}
}

func TestUsageAdd(t *testing.T) {
	u := Usage{InputTokens: 1, OutputTokens: 2, CacheReadTokens: 3, CacheWriteTokens: 4}
	u.Add(Usage{InputTokens: 10, OutputTokens: 20, CacheReadTokens: 30, CacheWriteTokens: 40})
	if u != (Usage{11, 22, 33, 44}) {
		t.Errorf("Add = %+v", u)
	}
}

func TestErrorRetryable(t *testing.T) {
	for kind, want := range map[ErrorKind]bool{
		ErrRateLimited:    true,
		ErrOverloaded:     true,
		ErrServer:         true,
		ErrContextTooLong: false,
		ErrInvalidRequest: false,
		ErrAuth:           false,
	} {
		err := fmt.Errorf("turn 3: %w", &Error{Provider: "x", Kind: kind, Message: "m"})
		if got := IsRetryable(err); got != want {
			t.Errorf("IsRetryable(%s) = %v, want %v", kind, got, want)
		}
	}
	if IsRetryable(errors.New("plain")) {
		t.Error("plain error is retryable")
	}

	cause := errors.New("sdk failure")
	e := &Error{Provider: "anthropic", Kind: ErrServer, StatusCode: 500, Message: "boom", Err: cause}
	if !errors.Is(e, cause) {
		t.Error("Error does not unwrap to its cause")
	}
	if e.Error() == "" {
		t.Error("empty error string")
	}
}
