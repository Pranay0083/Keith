package gemini

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/Pranay0083/keith/internal/llm"
)

// testKey is a fake API key. Tests assert it never shows up in errors.
const testKey = "test-key-SECRET-0123456789"

// recorded is one request the fake Gemini server received.
type recorded struct {
	Method   string
	Path     string
	RawQuery string
	APIKey   string
	RawBody  []byte
	Body     map[string]any
}

// fakeGemini is an httptest server that answers each request with the
// next scripted handler.
type fakeGemini struct {
	t   *testing.T
	srv *httptest.Server

	mu      sync.Mutex
	reqs    []recorded
	replies []http.HandlerFunc
}

func newFake(t *testing.T, replies ...http.HandlerFunc) *fakeGemini {
	t.Helper()
	f := &fakeGemini{t: t, replies: replies}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeGemini) serve(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)

	f.mu.Lock()
	f.reqs = append(f.reqs, recorded{
		Method:   r.Method,
		Path:     r.URL.Path,
		RawQuery: r.URL.RawQuery,
		APIKey:   r.Header.Get("x-goog-api-key"),
		RawBody:  raw,
		Body:     body,
	})
	var h http.HandlerFunc
	if len(f.replies) > 0 {
		h, f.replies = f.replies[0], f.replies[1:]
	}
	f.mu.Unlock()

	if h == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"fake: no scripted reply","status":"INTERNAL"}}`)
		return
	}
	h(w, r)
}

func (f *fakeGemini) requests() []recorded {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recorded(nil), f.reqs...)
}

// lastBody returns the body of the only or last request.
func (f *fakeGemini) lastBody() map[string]any {
	f.t.Helper()
	reqs := f.requests()
	if len(reqs) == 0 {
		f.t.Fatal("fake server received no request")
	}
	return reqs[len(reqs)-1].Body
}

func (f *fakeGemini) provider(opts ...Option) *Provider {
	f.t.Helper()
	p, err := New(testKey, append([]Option{WithBaseURL(f.srv.URL)}, opts...)...)
	if err != nil {
		f.t.Fatalf("New: %v", err)
	}
	return p
}

// sse replies with a streamGenerateContent?alt=sse body made of chunks.
func sse(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if fl != nil {
				fl.Flush()
			}
		}
	}
}

// jsonError replies with an HTTP error status and a JSON body.
func jsonError(status int, body string, headers ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i+1 < len(headers); i += 2 {
			w.Header().Set(headers[i], headers[i+1])
		}
		w.Header().Set("Content-Type", "application/json; charset=UTF-8")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// Chunk builders producing Gemini's streamed GenerateContentResponse JSON.

type part = map[string]any

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func txt(s string) part     { return part{"text": s} }
func thought(s string) part { return part{"text": s, "thought": true} }

func signed(p part, sig string) part {
	p["thoughtSignature"] = b64(sig)
	return p
}

func call(id, name string, args map[string]any) part {
	fc := map[string]any{"name": name, "args": args}
	if id != "" {
		fc["id"] = id
	}
	return part{"functionCall": fc}
}

func chunk(parts ...part) string {
	return mustJSON(map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{"role": "model", "parts": parts},
			"index":   0,
		}},
		"modelVersion": "gemini-test-001",
		"responseId":   "resp-abc",
	})
}

// finish is the last chunk of a stream: an empty text part (as Gemini
// sends), the finish reason and usage.
func finish(reason string, usage map[string]any) string {
	m := map[string]any{
		"candidates": []any{map[string]any{
			"content":      map[string]any{"role": "model", "parts": []any{map[string]any{"text": ""}}},
			"finishReason": reason,
			"index":        0,
		}},
		"modelVersion": "gemini-test-001",
		"responseId":   "resp-abc",
	}
	if usage != nil {
		m["usageMetadata"] = usage
	}
	return mustJSON(m)
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// assertJSON compares got (decoded JSON or any marshalable value) with the
// JSON document want.
func assertJSON(t *testing.T, what string, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("%s: bad want JSON: %v", what, err)
	}
	gb, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("%s: marshal got: %v", what, err)
	}
	var g any
	if err := json.Unmarshal(gb, &g); err != nil {
		t.Fatalf("%s: unmarshal got: %v", what, err)
	}
	if !reflect.DeepEqual(g, w) {
		wb, _ := json.MarshalIndent(w, "", "  ")
		gi, _ := json.MarshalIndent(g, "", "  ")
		t.Errorf("%s mismatch\n got: %s\nwant: %s", what, gi, wb)
	}
}

// get walks a decoded JSON value by object keys and array indexes.
func get(t *testing.T, v any, path ...any) any {
	t.Helper()
	cur := v
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("get %v: %v is not an object", path, cur)
			}
			cur = m[k]
		case int:
			a, ok := cur.([]any)
			if !ok || k >= len(a) {
				t.Fatalf("get %v: %v is not an array with index %d", path, cur, k)
			}
			cur = a[k]
		}
	}
	return cur
}

// rawOf decodes a block's ProviderRaw.
func rawOf(t *testing.T, b llm.Block) rawData {
	t.Helper()
	var r rawData
	if len(b.ProviderRaw) == 0 {
		return r
	}
	if err := json.Unmarshal(b.ProviderRaw, &r); err != nil {
		t.Fatalf("ProviderRaw %s: %v", b.ProviderRaw, err)
	}
	return r
}

// asLLMError asserts err is an *llm.Error from this provider.
func asLLMError(t *testing.T, err error) *llm.Error {
	t.Helper()
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	var e *llm.Error
	if !errors.As(err, &e) {
		t.Fatalf("error %T (%v) is not *llm.Error", err, err)
	}
	if e.Provider != Name {
		t.Errorf("Provider = %q, want %q", e.Provider, Name)
	}
	return e
}
