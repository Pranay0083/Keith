package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/Pranay0083/keith/internal/llm"
)

// invalidRequest reports a request this provider can't send.
func invalidRequest(msg string) *llm.Error {
	return &llm.Error{Provider: Name, Kind: llm.ErrInvalidRequest, Message: msg}
}

// mapError turns an error from the SDK into the error Generate returns:
// ctx's error when the call was cancelled, otherwise an *llm.Error.
func mapError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		// Transport failure: connection refused or reset, TLS, a stream
		// cut mid-body. Worth retrying.
		return &llm.Error{Provider: Name, Kind: llm.ErrServer, Message: err.Error(), Err: err}
	}

	errType, msg := apiErrorBody(apiErr)
	if errType == "" {
		errType = string(apiErr.Type())
	}
	status := apiErr.StatusCode
	if msg == "" {
		msg = http.StatusText(status)
		if errType != "" {
			msg = errType
		}
	}
	if apiErr.RequestID != "" {
		msg += " (request-id " + apiErr.RequestID + ")"
	}

	e := &llm.Error{
		Provider:   Name,
		Kind:       classify(status, errType, msg),
		StatusCode: status,
		Message:    msg,
		Err:        err,
	}
	if status < 300 {
		// An error event inside a stream that started with 200: there
		// is no HTTP status for the error itself.
		e.StatusCode = 0
	}
	if apiErr.Response != nil {
		e.RetryAfter = retryAfter(apiErr.Response.Header, time.Now())
	}
	return e
}

// classify picks the error kind. The error type from the body is checked
// first because an error event inside a stream arrives on a 200 response.
func classify(status int, errType, msg string) llm.ErrorKind {
	lower := strings.ToLower(msg)
	switch {
	case errType == "rate_limit_error" || status == http.StatusTooManyRequests:
		return llm.ErrRateLimited
	case errType == "overloaded_error" || status == 529:
		return llm.ErrOverloaded
	case errType == "authentication_error" || errType == "permission_error" ||
		status == http.StatusUnauthorized || status == http.StatusForbidden:
		return llm.ErrAuth
	case errType == "request_too_large" || status == http.StatusRequestEntityTooLarge ||
		strings.Contains(lower, "prompt is too long") ||
		strings.Contains(lower, "exceed context limit") ||
		strings.Contains(lower, "context window"):
		return llm.ErrContextTooLong
	case errType == "api_error" || errType == "timeout_error" || status >= 500:
		return llm.ErrServer
	case status >= 400:
		return llm.ErrInvalidRequest
	case errType == "invalid_request_error" || errType == "not_found_error" || errType == "billing_error":
		return llm.ErrInvalidRequest
	}
	// An error event with an unknown type mid-stream.
	return llm.ErrServer
}

// apiErrorBody reads the {"type":"error","error":{"type","message"}} body.
func apiErrorBody(e *sdk.Error) (errType, msg string) {
	var body struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(e.RawJSON()), &body) != nil {
		return "", ""
	}
	return body.Error.Type, body.Error.Message
}

// retryAfter reads retry-after-ms, then retry-after (seconds or an HTTP
// date). It returns 0 when neither is usable.
func retryAfter(h http.Header, now time.Time) time.Duration {
	if v := h.Get("Retry-After-Ms"); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && ms > 0 {
			return time.Duration(ms * float64(time.Millisecond))
		}
	}
	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}
	if s, err := strconv.ParseFloat(v, 64); err == nil {
		if s <= 0 {
			return 0
		}
		return time.Duration(s * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil && t.After(now) {
		return t.Sub(now)
	}
	return 0
}
