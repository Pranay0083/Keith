package llmtest

import (
	"context"
	"errors"
	"testing"

	"github.com/Pranay0083/keith/internal/llm"
)

func TestScriptedResponsesInOrder(t *testing.T) {
	p := New(
		ToolCall("call_1", "read_file", map[string]any{"path": "a.go"}),
		Text("done"),
	)
	ctx := context.Background()
	conv := []llm.Message{llm.UserText("go")}

	var events []llm.StreamEvent
	r1, err := p.Generate(ctx, llm.Request{Model: "m", Messages: conv}, func(e llm.StreamEvent) { events = append(events, e) })
	if err != nil {
		t.Fatal(err)
	}
	if r1.StopReason != llm.StopToolUse || r1.Message.ToolUses()[0].Name != "read_file" {
		t.Fatalf("first response = %+v", r1)
	}
	if len(events) != 1 || events[0].Type != llm.StreamToolUseStart || events[0].ToolUseID != "call_1" {
		t.Errorf("events = %+v", events)
	}
	if string(r1.Message.ToolUses()[0].Input) != `{"path":"a.go"}` {
		t.Errorf("input = %s", r1.Message.ToolUses()[0].Input)
	}

	// The caller grows the conversation in place; the recording must not change.
	conv = append(conv, r1.Message)
	conv[0].Content[0].Text = "mutated"

	r2, err := p.Generate(ctx, llm.Request{Model: "m", Messages: conv}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r2.Message.Text() != "done" || r2.StopReason != llm.StopEndTurn {
		t.Fatalf("second response = %+v", r2)
	}

	reqs := p.Requests()
	if len(reqs) != 2 || len(reqs[0].Messages) != 1 || len(reqs[1].Messages) != 2 {
		t.Fatalf("requests = %+v", reqs)
	}
	if got := reqs[0].Messages[0].Content[0].Text; got != "go" {
		t.Errorf("recorded request changed: %q", got)
	}

	if _, err := p.Generate(ctx, llm.Request{}, nil); !errors.Is(err, ErrExhausted) {
		t.Errorf("err = %v, want ErrExhausted", err)
	}
}

func TestPushErrorAndFunc(t *testing.T) {
	boom := &llm.Error{Provider: "fake", Kind: llm.ErrOverloaded, Message: "busy"}
	p := New()
	p.PushError(boom)
	p.PushFunc(func(req llm.Request) (*llm.Response, error) {
		return Text("model was " + req.Model), nil
	})

	if _, err := p.Generate(context.Background(), llm.Request{}, nil); !llm.IsRetryable(err) {
		t.Errorf("err = %v, want retryable llm.Error", err)
	}
	r, err := p.Generate(context.Background(), llm.Request{Model: "x"}, nil)
	if err != nil || r.Message.Text() != "model was x" {
		t.Errorf("r = %+v, err = %v", r, err)
	}
	if p.Remaining() != 0 {
		t.Errorf("Remaining = %d", p.Remaining())
	}
}

func TestCancelledContext(t *testing.T) {
	p := New(Text("unused"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Generate(ctx, llm.Request{}, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
	if p.Remaining() != 1 {
		t.Error("cancelled call consumed a step")
	}
	if p.WithName("anthropic").Name() != "anthropic" {
		t.Error("WithName did not apply")
	}
}
