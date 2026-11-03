package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"strconv"
	"strings"
)

// Error codes (docs/api.md, Errors).
const (
	codeBadRequest          = "bad_request"
	codeUnauthenticated     = "unauthenticated"
	codeNotAMember          = "not_a_member"
	codeForbiddenRole       = "forbidden_role"
	codeBadOrigin           = "bad_origin"
	codeNotFound            = "not_found"
	codeNotWaiting          = "not_waiting"
	codeQuestionMismatch    = "question_mismatch"
	codeAnswerRequired      = "answer_required"
	codeNotRunning          = "not_running"
	codeRepoAccessLost      = "repo_access_lost"
	codeTooLarge            = "too_large"
	codeUnsupportedMedia    = "unsupported_media_type"
	codeValidationFailed    = "validation_failed"
	codeProviderUnavailable = "provider_unavailable"
	codeRateLimited         = "rate_limited"
	codeInternal            = "internal"
)

// maxBody is the largest request body the API reads (1 MiB).
const maxBody = 1 << 20

// apiError is an error with its HTTP status and docs/api.md body.
type apiError struct {
	Status  int
	Code    string
	Message string
	Fields  map[string]string
	// RetryAfter, in seconds, sets the Retry-After header when > 0.
	RetryAfter int
}

func (e *apiError) Error() string { return e.Code + ": " + e.Message }

func errorf(status int, code, format string, args ...any) *apiError {
	return &apiError{Status: status, Code: code, Message: fmt.Sprintf(format, args...)}
}

// validation returns a 422 validation_failed for one field.
func validation(field, msg string) *apiError {
	return &apiError{
		Status: http.StatusUnprocessableEntity, Code: codeValidationFailed,
		Message: field + ": " + msg, Fields: map[string]string{field: msg},
	}
}

func errBadRequest(format string, args ...any) *apiError {
	return errorf(http.StatusBadRequest, codeBadRequest, format, args...)
}

var (
	errUnauthenticated = errorf(http.StatusUnauthorized, codeUnauthenticated, "Sign in to continue.")
	errNotAMember      = errorf(http.StatusForbidden, codeNotAMember, "You are not a member of this organization.")
	errForbiddenRole   = errorf(http.StatusForbidden, codeForbiddenRole, "Your role in this organization does not allow this action.")
	errBadOrigin       = errorf(http.StatusForbidden, codeBadOrigin, "The request origin is not allowed.")
	errNotFound        = errorf(http.StatusNotFound, codeNotFound, "Not found.")
	errInternal        = errorf(http.StatusInternalServerError, codeInternal, "Something went wrong. Try again.")
)

type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Code    string            `json:"code"`
	Message string            `json:"message"`
	Fields  map[string]string `json:"fields,omitempty"`
}

// writeJSON writes v with status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		status, b = http.StatusInternalServerError,
			[]byte(`{"error":{"code":"internal","message":"Something went wrong. Try again."}}`)
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(append(b, '\n'))
}

// writeError writes err as a docs/api.md error. Anything but an *apiError
// is logged and becomes 500 internal, so internals never leak.
func (s *Server) writeError(w http.ResponseWriter, r *http.Request, err error) {
	var ae *apiError
	if !errors.As(err, &ae) {
		s.log(r).Error("request failed", "err", err)
		ae = errInternal
	}
	if ae.RetryAfter > 0 {
		w.Header().Set("Retry-After", strconv.Itoa(ae.RetryAfter))
	}
	writeJSON(w, ae.Status, errorBody{errorDetail{ae.Code, ae.Message, ae.Fields}})
}

// handlerFunc is a handler that returns its error.
type handlerFunc func(w http.ResponseWriter, r *http.Request) error

// handle adapts a handlerFunc, writing its error.
func (s *Server) handle(h handlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			s.writeError(w, r, err)
		}
	}
}

// hasBody reports whether the request carries a body.
func hasBody(r *http.Request) bool {
	return r.ContentLength > 0 || (r.ContentLength < 0 && len(r.TransferEncoding) > 0)
}

// checkBody enforces the JSON content type and the size limit on a
// request with a body, and caps the body reader at maxBody.
func checkBody(w http.ResponseWriter, r *http.Request) error {
	if !hasBody(r) {
		return nil
	}
	mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mt != "application/json" {
		return errorf(http.StatusUnsupportedMediaType, codeUnsupportedMedia, "Send the body as application/json.")
	}
	if r.ContentLength > maxBody {
		return errorf(http.StatusRequestEntityTooLarge, codeTooLarge, "The body is larger than 1 MiB.")
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	return nil
}

// decodeJSON reads the request body into v. Unknown fields are ignored.
func decodeJSON(r *http.Request, v any) error {
	if !hasBody(r) {
		return errBadRequest("A JSON body is required.")
	}
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		return jsonError(err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return jsonError(err)
		}
		return errBadRequest("The body must be a single JSON value.")
	}
	return nil
}

func jsonError(err error) error {
	var mbe *http.MaxBytesError
	var te *json.UnmarshalTypeError
	switch {
	case errors.As(err, &mbe):
		return errorf(http.StatusRequestEntityTooLarge, codeTooLarge, "The body is larger than 1 MiB.")
	case errors.As(err, &te) && te.Field != "":
		field := te.Field
		if i := strings.IndexByte(field, '.'); i >= 0 {
			field = field[:i]
		}
		return validation(field, "must be "+jsonKind(te.Type.Kind().String()))
	case errors.As(err, &te):
		return errBadRequest("The body must be a JSON object.")
	default:
		return errBadRequest("The body is not valid JSON.")
	}
}

func jsonKind(k string) string {
	switch k {
	case "string":
		return "a string"
	case "slice", "array":
		return "an array"
	case "bool":
		return "a boolean"
	case "struct", "map":
		return "an object"
	}
	if strings.HasPrefix(k, "int") || strings.HasPrefix(k, "uint") || strings.HasPrefix(k, "float") {
		return "a number"
	}
	return "of the right type"
}

// log returns the request's logger.
func (s *Server) log(r *http.Request) *slog.Logger {
	if id := requestID(r.Context()); id != "" {
		return s.logger.With("request_id", id)
	}
	return s.logger
}

// validationFields returns a 422 validation_failed for every failed
// field, or nil if there are none. The message names the first field in
// alphabetical order, so it is stable.
func validationFields(fields map[string]string) error {
	if len(fields) == 0 {
		return nil
	}
	first := ""
	for k := range fields {
		if first == "" || k < first {
			first = k
		}
	}
	return &apiError{
		Status: http.StatusUnprocessableEntity, Code: codeValidationFailed,
		Message: first + ": " + fields[first], Fields: fields,
	}
}

func itoa(n int) string { return strconv.Itoa(n) }
