package worker

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/secrets"
)

// chunkyProvider streams its text in small deltas, as real providers do.
type chunkyProvider struct {
	*llmtest.Provider
	chunk int
}

func (c chunkyProvider) Generate(ctx context.Context, req llm.Request, on func(llm.StreamEvent)) (*llm.Response, error) {
	resp, err := c.Provider.Generate(ctx, req, nil)
	if err != nil || on == nil {
		return resp, err
	}
	for _, b := range resp.Message.Content {
		if b.Type == llm.BlockText {
			for i := 0; i < len(b.Text); i += c.chunk {
				on(llm.StreamEvent{Type: llm.StreamTextDelta, Text: b.Text[i:min(i+c.chunk, len(b.Text))]})
			}
		}
	}
	return resp, nil
}

// message.delta used to be redacted one delta at a time, so a secret
// split across deltas reached live viewers in clear. The live stream
// must carry the whole message, redacted, and nothing else.
func TestSecretSplitAcrossDeltasNeverGoesLive(t *testing.T) {
	e := newEnv(t)
	e.addSecret("API_KEY", testSecret, "")
	s := e.newSession()
	text := "The key in .env is " + testSecret + ", rotate it."
	e.llm.Push(llmtest.Text(text))
	w := e.worker()
	w.d.LLM = func(string) (llm.Provider, error) { return chunkyProvider{e.llm, 4}, nil }
	ch, unsub, err := e.bus.Subscribe(context.Background(), s.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer unsub()
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	var live strings.Builder
	timeout := time.After(time.Second)
loop:
	for {
		select {
		case ev := <-ch:
			switch ev.Type {
			case events.TypeMessageDelta:
				live.WriteString(payload[events.MessageDeltaPayload](t, ev).Text)
			case events.TypeSessionStatus:
				if payload[events.SessionStatusPayload](t, ev).Status.Terminal() {
					break loop
				}
			}
		case <-timeout:
			break loop
		}
	}
	e.noSecrets(s.ID, testSecret)
	if strings.Contains(live.String(), testSecret) {
		t.Errorf("the live message.delta stream carries the secret: %q", live.String())
	}
	if want := strings.ReplaceAll(text, testSecret, secrets.Redacted); live.String() != want {
		t.Errorf("live deltas = %q, want %q", live.String(), want)
	}
}
