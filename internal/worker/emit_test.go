package worker

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
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

// Secrets must not leave the session through what the worker or the
// agent writes to GitHub: the PR title and body, and the commit message.
func TestPRTitleBodyAndCommitAreRedacted(t *testing.T) {
	t.Run("worker ship", func(t *testing.T) {
		e := newEnv(t)
		e.addSecret("API_KEY", testSecret, "")
		s := e.newSession(withPrompt("Rotate " + testSecret + " in the config\nand redeploy"))
		e.llm.Push(llmtest.Text("Fixed the login bug. For reference the configured key is " + testSecret + "."))
		e.setChanges(true, 0)
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		prs := e.gh.host.PRRequests()
		if len(prs) != 1 {
			t.Fatalf("PR requests = %d", len(prs))
		}
		if strings.Contains(prs[0].Title, testSecret) || strings.Contains(prs[0].Body, testSecret) {
			t.Errorf("the worker's PR carries the secret: %+v", prs[0])
		}
		if !strings.Contains(prs[0].Body, secrets.Redacted) {
			t.Errorf("PR body = %q", prs[0].Body)
		}
		for _, c := range e.execs(e.sandbox(s.ID), " commit ") {
			if strings.Contains(c.Cmd, testSecret) {
				t.Errorf("commit message carries the secret: %s", c.Cmd)
			}
		}
		e.noSecrets(s.ID, testSecret)
	})
	t.Run("open_pr", func(t *testing.T) {
		e := newEnv(t)
		e.addSecret("API_KEY", testSecret, "")
		s := e.newSession()
		e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
			e.setChanges(true, 0)
			return toolCall("c1", "open_pr", map[string]any{
				"title": "Use key " + testSecret, "body": "The key is " + testSecret + ".",
			}), nil
		})
		e.llm.Push(llmtest.Text("Opened."))
		if err := e.run(s.ID); err != nil {
			t.Fatal(err)
		}
		prs := e.gh.host.PRRequests()
		if len(prs) != 1 || strings.Contains(prs[0].Title, testSecret) || strings.Contains(prs[0].Body, testSecret) {
			t.Errorf("open_pr's PR carries the secret: %+v", prs)
		}
		for _, c := range e.execs(e.sandbox(s.ID), " commit ") {
			if strings.Contains(c.Cmd, testSecret) {
				t.Errorf("commit message carries the secret: %s", c.Cmd)
			}
		}
	})
}

// failingStore fails one call with an error carrying a secret, as a
// driver or command error quoting output might.
type failingStore struct {
	Store
	err error
}

func (f failingStore) IsCancelRequested(context.Context, string) (bool, error) { return false, f.err }

// Task errors are logged and handed to asynq (which stores their text):
// they must be redacted, and still unwrap.
func TestTaskErrorsAreRedacted(t *testing.T) {
	e := newEnv(t)
	e.addSecret("API_KEY", testSecret, "")
	cause := errors.New("pq: value " + testSecret + " is wrong")
	e.store = failingStore{Store: e.st, err: cause}
	var logs strings.Builder
	w := e.worker()
	w.log = slog.New(slog.NewTextHandler(&lockedWriter{w: &logs}, nil))
	s := e.newSession()
	err := e.run(s.ID)
	if err == nil {
		t.Fatal("expected a task error")
	}
	if strings.Contains(err.Error(), testSecret) || !strings.Contains(err.Error(), secrets.Redacted) {
		t.Errorf("task error = %q", err)
	}
	if !errors.Is(err, cause) {
		t.Error("the redacted error does not unwrap to its cause")
	}
	if strings.Contains(logs.String(), testSecret) {
		t.Errorf("logs carry the secret:\n%s", logs.String())
	}
}

// A failing command's output is redacted before it is cut to its tail,
// so the cut cannot leave the end of a secret in clear.
func TestCommandErrorsRedactBeforeCutting(t *testing.T) {
	e := newEnv(t)
	e.addSecret("API_KEY", testSecret, "")
	s := e.newSession()
	// The 2000-byte tail starts in the middle of the secret.
	stderr := "fatal: token " + testSecret + strings.Repeat("x", 2000-len(testSecret)/2)
	e.sbRules = append(e.sbRules, memsandbox.On(memsandbox.Contains("git clone"), memsandbox.Exit(128, "", stderr)))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	half := testSecret[len(testSecret)/2:]
	for _, ev := range e.events(s.ID, events.TypeError) {
		if strings.Contains(string(ev.Payload), half) {
			t.Errorf("error event carries the end of the secret: %s", ev.Payload)
		}
	}
}

// lockedWriter serializes writes from the worker's goroutines.
type lockedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}
