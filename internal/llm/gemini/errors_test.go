package gemini

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/llm"
)

func errBody(code int, status, message string, details ...string) string {
	d := ""
	if len(details) > 0 {
		d = `,"details":[` + strings.Join(details, ",") + `]`
	}
	return fmt.Sprintf(`{"error":{"code":%d,"message":%q,"status":%q%s}}`, code, message, status, d)
}

func retryInfo(delay string) string {
	return fmt.Sprintf(`{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":%q}`, delay)
}

func TestErrorMapping(t *testing.T) {
	quota := `{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaMetric":"generativelanguage.googleapis.com/generate_content_free_tier_requests"}]}`
	keyInvalid := `{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com","metadata":{"service":"generativelanguage.googleapis.com"}}`

	tests := []struct {
		name       string
		reply      http.HandlerFunc
		kind       llm.ErrorKind
		status     int
		retryAfter time.Duration
	}{
		{
			name:       "429 with RetryInfo",
			reply:      jsonError(429, errBody(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.", quota, retryInfo("37s"))),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 37 * time.Second,
		},
		{
			name:       "429 with fractional RetryInfo",
			reply:      jsonError(429, errBody(429, "RESOURCE_EXHAUSTED", "Resource has been exhausted.", retryInfo("1.5s"))),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 1500 * time.Millisecond,
		},
		{
			name:       "429 with Retry-After header",
			reply:      jsonError(429, errBody(429, "RESOURCE_EXHAUSTED", "Resource has been exhausted."), "Retry-After", "12"),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 12 * time.Second,
		},
		{
			name:   "503 overloaded",
			reply:  jsonError(503, errBody(503, "UNAVAILABLE", "The model is overloaded. Please try again later.")),
			kind:   llm.ErrOverloaded,
			status: 503,
		},
		{
			name:   "500 internal",
			reply:  jsonError(500, errBody(500, "INTERNAL", "An internal error has occurred.")),
			kind:   llm.ErrServer,
			status: 500,
		},
		{
			name: "502 non-JSON body",
			reply: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "<html>Bad Gateway</html>", http.StatusBadGateway)
			},
			kind:   llm.ErrServer,
			status: 502,
		},
		{
			name:   "504 deadline exceeded",
			reply:  jsonError(504, errBody(504, "DEADLINE_EXCEEDED", "Deadline expired before operation could complete.")),
			kind:   llm.ErrServer,
			status: 504,
		},
		{
			name:   "400 input too long",
			reply:  jsonError(400, errBody(400, "INVALID_ARGUMENT", "The input token count (1250000) exceeds the maximum number of tokens allowed (1048576).")),
			kind:   llm.ErrContextTooLong,
			status: 400,
		},
		{
			name:   "400 API key invalid",
			reply:  jsonError(400, errBody(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", keyInvalid)),
			kind:   llm.ErrAuth,
			status: 400,
		},
		{
			name:   "400 message echoing the key",
			reply:  jsonError(400, errBody(400, "INVALID_ARGUMENT", "API key not valid: "+testKey)),
			kind:   llm.ErrAuth,
			status: 400,
		},
		{
			name:   "401 unauthenticated",
			reply:  jsonError(401, errBody(401, "UNAUTHENTICATED", "Request had invalid authentication credentials.")),
			kind:   llm.ErrAuth,
			status: 401,
		},
		{
			name:   "403 permission denied",
			reply:  jsonError(403, errBody(403, "PERMISSION_DENIED", "Method doesn't allow unregistered callers.")),
			kind:   llm.ErrAuth,
			status: 403,
		},
		{
			name:   "404 unknown model",
			reply:  jsonError(404, errBody(404, "NOT_FOUND", "models/gemini-9-ultra is not found for API version v1beta.")),
			kind:   llm.ErrInvalidRequest,
			status: 404,
		},
		{
			name:   "400 other invalid argument",
			reply:  jsonError(400, errBody(400, "INVALID_ARGUMENT", "* GenerateContentRequest.tools[0].function_declarations[0].name: Invalid function name. Function name is too long.")),
			kind:   llm.ErrInvalidRequest,
			status: 400,
		},
		{
			name:   "error event inside the stream",
			reply:  sse(chunk(txt("partial")), errBody(500, "INTERNAL", "An internal error has occurred.")),
			kind:   llm.ErrServer,
			status: 500,
		},
		{
			name:       "rate limit event inside the stream",
			reply:      sse(errBody(429, "RESOURCE_EXHAUSTED", "Quota exceeded.", retryInfo("5s"))),
			kind:       llm.ErrRateLimited,
			status:     429,
			retryAfter: 5 * time.Second,
		},
		{
			name: "connection dropped",
			reply: func(w http.ResponseWriter, r *http.Request) {
				conn, _, err := w.(http.Hijacker).Hijack()
				if err == nil {
					conn.Close()
				}
			},
			kind: llm.ErrServer,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFake(t, tc.reply)
			start := time.Now()
			_, err := f.provider().Generate(context.Background(), llm.Request{
				Model:    "gemini-3-pro-preview",
				Messages: []llm.Message{llm.UserText("hi")},
			}, nil)
			e := asLLMError(t, err)
			if e.Kind != tc.kind {
				t.Errorf("Kind = %q, want %q (%v)", e.Kind, tc.kind, e)
			}
			if e.StatusCode != tc.status {
				t.Errorf("StatusCode = %d, want %d", e.StatusCode, tc.status)
			}
			if e.RetryAfter != tc.retryAfter {
				t.Errorf("RetryAfter = %v, want %v", e.RetryAfter, tc.retryAfter)
			}
			if e.Err == nil {
				t.Error("Err is nil, want the underlying error")
			}
			if e.Message == "" {
				t.Error("Message is empty")
			}
			if strings.Contains(err.Error(), testKey) || strings.Contains(e.Message, testKey) {
				t.Errorf("error leaks the API key: %v", err)
			}
			wantRetry := tc.kind == llm.ErrRateLimited || tc.kind == llm.ErrOverloaded || tc.kind == llm.ErrServer
			if llm.IsRetryable(err) != wantRetry {
				t.Errorf("IsRetryable = %v, want %v", llm.IsRetryable(err), wantRetry)
			}
			if n := len(f.requests()); n != 1 {
				t.Errorf("server got %d requests, want exactly 1 (no SDK retries)", n)
			}
			if d := time.Since(start); d > 2*time.Second {
				t.Errorf("Generate took %v; the SDK seems to be retrying", d)
			}
		})
	}
}

func TestTransportErrorIsServer(t *testing.T) {
	p, err := New(testKey, WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Generate(context.Background(), llm.Request{
		Model:    "gemini-3-pro-preview",
		Messages: []llm.Message{llm.UserText("hi")},
	}, nil)
	e := asLLMError(t, err)
	if e.Kind != llm.ErrServer || e.StatusCode != 0 || !e.Retryable() {
		t.Errorf("error = %v, want retryable server error without status", e)
	}
}

func TestContextCancellation(t *testing.T) {
	req := llm.Request{Model: "gemini-3-pro-preview", Messages: []llm.Message{llm.UserText("hi")}}

	t.Run("before the call", func(t *testing.T) {
		f := newFake(t, okStream)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := f.provider().Generate(ctx, req, nil)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
		if n := len(f.requests()); n != 0 {
			t.Errorf("sent %d requests, want 0", n)
		}
	})

	// hang streams one chunk, then waits until the client goes away.
	hang := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", chunk(txt("first")))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	}

	t.Run("mid-stream", func(t *testing.T) {
		f := newFake(t, hang)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var got []string
		_, err := f.provider().Generate(ctx, req, func(ev llm.StreamEvent) {
			got = append(got, ev.Text)
			cancel()
		})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
		if len(got) != 1 || got[0] != "first" {
			t.Errorf("events before cancel = %q", got)
		}
	})

	t.Run("deadline", func(t *testing.T) {
		f := newFake(t, hang)
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		_, err := f.provider().Generate(ctx, req, nil)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("err = %v, want context.DeadlineExceeded", err)
		}
	})
}

func TestRetryAfterParsing(t *testing.T) {
	now := time.Date(2026, 11, 3, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		in   string
		want time.Duration
	}{
		{"", 0},
		{"30", 30 * time.Second},
		{"0", 0},
		{"-5", 0},
		{"Tue, 03 Nov 2026 12:00:45 GMT", 45 * time.Second},
		{"Tue, 03 Nov 2026 11:00:00 GMT", 0},
		{"soon", 0},
	}
	for _, tc := range tests {
		if got := parseRetryAfter(tc.in, now); got != tc.want {
			t.Errorf("parseRetryAfter(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}

	details := []map[string]any{{
		"@type":      "type.googleapis.com/google.rpc.RetryInfo",
		"retryDelay": map[string]any{"seconds": "2", "nanos": float64(500000000)},
	}}
	if got := retryInfoDelay(details); got != 2500*time.Millisecond {
		t.Errorf("retryInfoDelay(object) = %v, want 2.5s", got)
	}
}
