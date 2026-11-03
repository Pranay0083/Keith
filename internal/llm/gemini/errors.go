package gemini

import (
	"errors"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"google.golang.org/genai"

	"github.com/Pranay0083/keith/internal/llm"
)

// maxMessageLen caps llm.Error.Message; non-JSON error bodies (proxy HTML
// pages) can be long.
const maxMessageLen = 1024

// mapError turns an error from the genai stream into an *llm.Error.
// retryAfter is the Retry-After header of the failed response, if any.
func (p *Provider) mapError(err error, retryAfter time.Duration) error {
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return p.fromAPIError(apiErr, err, retryAfter)
	}
	var apiErrPtr *genai.APIError
	if errors.As(err, &apiErrPtr) && apiErrPtr != nil {
		return p.fromAPIError(*apiErrPtr, err, retryAfter)
	}
	return &llm.Error{
		Provider: Name,
		Kind:     llm.ErrServer,
		Message:  p.message(err.Error()),
		Err:      err,
	}
}

func (p *Provider) fromAPIError(apiErr genai.APIError, err error, headerRetryAfter time.Duration) *llm.Error {
	kind := classify(apiErr.Code, apiErr.Status, apiErr.Message, apiErr.Details)
	msg := strings.TrimSpace(apiErr.Message)
	if status := strings.TrimSpace(apiErr.Status); status != "" && !strings.HasPrefix(msg, status) {
		if msg == "" {
			msg = status
		} else {
			msg = status + ": " + msg
		}
	}
	if msg == "" {
		msg = http.StatusText(apiErr.Code)
	}
	e := &llm.Error{
		Provider:   Name,
		Kind:       kind,
		StatusCode: apiErr.Code,
		Message:    p.message(msg),
		Err:        err,
	}
	if kind == llm.ErrRateLimited || kind == llm.ErrOverloaded {
		e.RetryAfter = retryInfoDelay(apiErr.Details)
		if e.RetryAfter == 0 {
			e.RetryAfter = headerRetryAfter
		}
	}
	return e
}

// classify maps an API error to an ErrorKind; see the package doc.
func classify(code int, status, message string, details []map[string]any) llm.ErrorKind {
	status = strings.ToUpper(strings.TrimSpace(status))
	msg := strings.ToLower(message)
	switch {
	case code == http.StatusTooManyRequests || status == "RESOURCE_EXHAUSTED":
		return llm.ErrRateLimited
	case code == http.StatusServiceUnavailable || status == "UNAVAILABLE":
		return llm.ErrOverloaded
	case code >= 500, code == http.StatusRequestTimeout,
		status == "INTERNAL", status == "UNKNOWN", status == "DEADLINE_EXCEEDED",
		status == "ABORTED", status == "DATA_LOSS":
		return llm.ErrServer
	case code == http.StatusUnauthorized, code == http.StatusForbidden,
		status == "UNAUTHENTICATED", status == "PERMISSION_DENIED",
		apiKeyRejected(msg, details):
		return llm.ErrAuth
	case code == http.StatusRequestEntityTooLarge || tooLong(msg):
		return llm.ErrContextTooLong
	case code >= 400 || code == 0 && status != "":
		return llm.ErrInvalidRequest
	}
	return llm.ErrServer
}

func apiKeyRejected(msg string, details []map[string]any) bool {
	for _, d := range details {
		reason, _ := d["reason"].(string)
		switch reason {
		case "API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED":
			return true
		}
	}
	return strings.Contains(msg, "api key not valid") ||
		strings.Contains(msg, "api key expired") ||
		strings.Contains(msg, "api key is invalid")
}

// tooLongPhrases identify "the prompt does not fit" in an API message
// (already lower-cased). Bare "too long" is not enough on its own: Gemini
// also says it about function names and descriptions.
var tooLongPhrases = []string{
	"exceeds the maximum number of tokens",
	"input token count",
	"token count exceeds",
	"too many tokens",
	"context window",
	"context length",
	"prompt is too long",
	"input is too long",
	"input too long",
	"request is too long",
	"request payload size exceeds",
}

func tooLong(msg string) bool {
	for _, p := range tooLongPhrases {
		if strings.Contains(msg, p) {
			return true
		}
	}
	return false
}

// retryInfoDelay returns the retryDelay of a google.rpc.RetryInfo detail,
// or 0. The delay is a protobuf Duration, normally JSON-encoded as a
// string such as "37s" or "1.5s".
func retryInfoDelay(details []map[string]any) time.Duration {
	for _, d := range details {
		typ, _ := d["@type"].(string)
		if !strings.HasSuffix(typ, "google.rpc.RetryInfo") {
			continue
		}
		switch v := d["retryDelay"].(type) {
		case string:
			if dur, err := time.ParseDuration(v); err == nil && dur > 0 {
				return dur
			}
		case map[string]any:
			secs := toFloat(v["seconds"]) + toFloat(v["nanos"])/1e9
			if secs > 0 {
				return time.Duration(secs * float64(time.Second))
			}
		}
	}
	return 0
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case string:
		f, _ := strconv.ParseFloat(n, 64)
		return f
	}
	return 0
}

// parseRetryAfter parses a Retry-After header: delay seconds or an HTTP
// date.
func parseRetryAfter(v string, now time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(v, 64); err == nil {
		if secs > 0 {
			return time.Duration(secs * float64(time.Second))
		}
		return 0
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// message makes s safe and short enough for llm.Error.Message.
func (p *Provider) message(s string) string {
	s = redact(s, p.apiKey)
	if len(s) > maxMessageLen {
		cut := maxMessageLen
		for cut > 0 && !utf8.RuneStart(s[cut]) {
			cut--
		}
		s = s[:cut] + "…"
	}
	return s
}

func redact(s, key string) string {
	if key == "" {
		return s
	}
	return strings.ReplaceAll(s, key, "[REDACTED]")
}

// genai's APIError does not carry response headers, so the transport
// records the Retry-After header of error responses into a headerCapture
// that Generate puts in the request context.

type headerCaptureKey struct{}

type headerCapture struct {
	mu    sync.Mutex
	value string
}

func (h *headerCapture) set(v string) {
	h.mu.Lock()
	h.value = v
	h.mu.Unlock()
}

func (h *headerCapture) retryAfter() time.Duration {
	h.mu.Lock()
	v := h.value
	h.mu.Unlock()
	return parseRetryAfter(v, time.Now())
}

type captureTransport struct {
	base http.RoundTripper
}

func (t captureTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if resp != nil && resp.StatusCode >= 400 {
		if h, ok := req.Context().Value(headerCaptureKey{}).(*headerCapture); ok {
			h.set(resp.Header.Get("Retry-After"))
		}
	}
	return resp, err
}

// CloseIdleConnections keeps http.Client.CloseIdleConnections working
// through the wrapper.
func (t captureTransport) CloseIdleConnections() {
	if c, ok := t.base.(interface{ CloseIdleConnections() }); ok {
		c.CloseIdleConnections()
	}
}

// withHeaderCapture returns a shallow copy of c (or a new client) whose
// transport feeds headerCapture.
func withHeaderCapture(c *http.Client) *http.Client {
	var out http.Client
	if c != nil {
		out = *c
	}
	base := out.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	out.Transport = captureTransport{base: base}
	return &out
}
