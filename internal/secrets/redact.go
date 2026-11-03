package secrets

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/url"
	"slices"
	"strings"

	"github.com/Pranay0083/keith/internal/events"
)

// Redacted is the text that replaces each occurrence of a secret.
const Redacted = "[redacted]"

// MinRedactLen is the shortest secret value, in bytes, that a Redactor
// looks for. Shorter values ("1", "on", "dev") occur all over ordinary
// output, so redacting them would mangle logs, diffs and JSON while
// protecting almost nothing: a value that short can be guessed anyway.
const MinRedactLen = 4

// minCoreLen is the shortest offset base64 core (see base64Cores) a
// Redactor uses. Shorter cores are common substrings of unrelated base64
// data, so they would cause frequent false positives.
const minCoreLen = 8

// ErrInvalidJSON is returned by Redactor.JSON and Redactor.Payload for
// input that is not a single valid JSON value. It never quotes the input.
var ErrInvalidJSON = errors.New("secrets: redact: invalid JSON")

// Redactor replaces secret values, and common encodings of them, with
// Redacted.
//
// Matching is leftmost, then longest: the text is scanned left to right,
// and at each position the longest pattern that matches there wins, so a
// secret that contains another is redacted whole. A match is not undone by
// a longer one starting later, so if two different secrets overlap in the
// text (the end of one is the start of the other), the tail of the second
// stays visible.
//
// A Redactor is immutable and safe for concurrent use. A nil *Redactor, or
// one built from no usable values, redacts nothing.
type Redactor struct {
	// byFirst[c] lists the patterns that start with byte c, longest
	// first (ties in lexical order, for determinism).
	byFirst [256][]string
	// exact holds every pattern, to match whole JSON number literals.
	exact map[string]struct{}
	// maxLen is the length of the longest pattern; a Stream holds back
	// fewer bytes than this.
	maxLen int
}

// NewRedactor returns a Redactor for the given secret values. Values
// shorter than MinRedactLen bytes (including empty ones) are ignored. For
// a value with leading or trailing whitespace, the trimmed value is also
// redacted, since that is how a pasted value usually shows up in output.
func NewRedactor(values []string) *Redactor {
	set := make(map[string]struct{})
	for _, v := range values {
		addPatterns(set, v)
		if t := strings.TrimSpace(v); t != v {
			addPatterns(set, t)
		}
	}
	r := &Redactor{exact: set}
	for p := range set {
		r.byFirst[p[0]] = append(r.byFirst[p[0]], p)
		r.maxLen = max(r.maxLen, len(p))
	}
	for i := range r.byFirst {
		slices.SortFunc(r.byFirst[i], func(a, b string) int {
			if c := cmp.Compare(len(b), len(a)); c != 0 {
				return c
			}
			return strings.Compare(a, b)
		})
	}
	return r
}

// addPatterns adds the forms of v that a Redactor looks for.
func addPatterns(set map[string]struct{}, v string) {
	if len(v) < MinRedactLen {
		return
	}
	b := []byte(v)
	forms := []string{
		v,
		base64.StdEncoding.EncodeToString(b),
		base64.URLEncoding.EncodeToString(b),
		base64.RawStdEncoding.EncodeToString(b),
		base64.RawURLEncoding.EncodeToString(b),
		url.QueryEscape(v),
		url.PathEscape(v),
		jsonEscape(v, false),
		jsonEscape(v, true),
	}
	forms = append(forms, base64Cores(b)...)
	for _, f := range forms {
		if f != "" {
			set[f] = struct{}{}
		}
	}
}

// jsonEscape returns v as the inside of a JSON string literal, which
// differs from v when it contains quotes, backslashes, control characters
// or (with html) <, > and &. It catches a value inside JSON that is itself
// inside a string (a tool input holding a config file, say).
func jsonEscape(v string, html bool) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(html)
	_ = enc.Encode(v) // encoding a string cannot fail
	s := buf.String() // `"...."` + "\n"
	return s[1 : len(s)-2]
}

// base64Cores returns, for each way b can be aligned in a larger base64
// stream (0, 1 or 2 bytes after a 3-byte group boundary), the run of
// base64 characters that depends only on b's bytes, in the standard and
// URL alphabets. Such a run appears in the encoding of any data that
// contains b at that alignment, such as "Basic " + base64(user + ":" +
// secret). Characters at the edges also depend on neighboring bytes, so
// they are left out. Cores shorter than minCoreLen are dropped, so a
// value shorter than 7 bytes is caught inside a larger blob only when it
// is group-aligned.
func base64Cores(b []byte) []string {
	var cores []string
	for off := 0; off < 3; off++ {
		// Character i covers bits [6i, 6i+6) of the stream; b occupies
		// bits [8*off, 8*(off+len(b))).
		start := (8*off + 5) / 6
		end := 8 * (off + len(b)) / 6
		if end-start < minCoreLen {
			continue
		}
		buf := make([]byte, off+len(b))
		copy(buf[off:], b)
		for _, enc := range []*base64.Encoding{base64.RawStdEncoding, base64.RawURLEncoding} {
			cores = append(cores, enc.EncodeToString(buf)[start:end])
		}
	}
	return cores
}

// active reports whether r has anything to redact.
func (r *Redactor) active() bool {
	return r != nil && r.maxLen > 0
}

// String returns s with every match replaced by Redacted.
func (r *Redactor) String(s string) string {
	if !r.active() {
		return s
	}
	out, _ := r.redact(s, true)
	return out
}

// redact scans s and replaces matches with Redacted. With final false, s
// may be followed by more text, so it stops at the first position whose
// outcome depends on bytes not yet seen (s[i:] is a proper prefix of some
// pattern) and returns s[i:] as tail, unscanned. Every decision it makes
// is the one a final scan of the whole text would make, which is what
// lets a Stream's output equal String's.
func (r *Redactor) redact(s string, final bool) (out, tail string) {
	var b strings.Builder
	done := 0 // s[:done] has been written to b; 0 means nothing redacted yet
	for i := 0; i < len(s); {
		n, more := r.matchAt(s[i:], final)
		switch {
		case more:
			if done == 0 {
				return s[:i], s[i:]
			}
			b.WriteString(s[done:i])
			return b.String(), s[i:]
		case n > 0:
			b.WriteString(s[done:i])
			b.WriteString(Redacted)
			i += n
			done = i
		default:
			i++
		}
	}
	if done == 0 {
		return s, ""
	}
	b.WriteString(s[done:])
	return b.String(), ""
}

// matchAt returns the length of the longest pattern that rest starts with,
// or 0. With final false it instead reports more if rest is a proper
// prefix of a longer pattern, since that pattern may yet match.
func (r *Redactor) matchAt(rest string, final bool) (n int, more bool) {
	// Patterns are sorted longest first, so every pattern longer than rest
	// is seen before any that could match it in full.
	for _, p := range r.byFirst[rest[0]] {
		if len(p) <= len(rest) {
			if rest[:len(p)] == p {
				return len(p), false
			}
		} else if !final && strings.HasPrefix(p, rest) {
			return 0, true
		}
	}
	return 0, false
}

// containsAny reports whether any pattern occurs in b.
func (r *Redactor) containsAny(b []byte) bool {
	for i, c := range b {
		for _, p := range r.byFirst[c] {
			if len(p) <= len(b)-i && string(b[i:i+len(p)]) == p {
				return true
			}
		}
	}
	return false
}

// JSON returns raw with every string value and object key redacted, and
// every number literal that equals a pattern replaced by the string
// Redacted. Strings are matched in decoded form, so a secret containing a
// quote, a backslash or a \u escape is caught. Object key order is kept.
//
// If something was redacted, the result is compact JSON with HTML
// escaping off; otherwise raw itself is returned. Redacting keys can
// produce duplicate keys. A number literal that merely contains a secret
// is left alone: it cannot be partly redacted, and turning it into a
// string could break decoding of a typed payload.
//
// For input that is not a single valid JSON value it returns nil and
// ErrInvalidJSON, even if r is nil.
func (r *Redactor) JSON(raw json.RawMessage) (json.RawMessage, error) {
	if !json.Valid(raw) {
		return nil, ErrInvalidJSON
	}
	if !r.active() {
		return raw, nil
	}
	// Without escapes, every decoded string and number is a substring of
	// raw, so if no pattern occurs in raw there is nothing to redact.
	if bytes.IndexByte(raw, '\\') < 0 && !r.containsAny(raw) {
		return raw, nil
	}
	out, changed, err := r.redactJSON(raw)
	if err != nil {
		return nil, err
	}
	if !changed {
		return raw, nil
	}
	return out, nil
}

// redactJSON re-encodes raw token by token, redacting strings and
// matching numbers. raw must be valid JSON.
func (r *Redactor) redactJSON(raw []byte) (json.RawMessage, bool, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()

	var buf bytes.Buffer
	buf.Grow(len(raw))
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	writeString := func(s string) {
		_ = enc.Encode(s)           // cannot fail for a string and a bytes.Buffer
		buf.Truncate(buf.Len() - 1) // drop Encode's newline
	}

	// One frame per open object or array; n counts the tokens written in
	// it (in an object, keys and values alternate, so even n is a key).
	type frame struct {
		object bool
		n      int
	}
	var stack []frame
	changed := false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			// The decoder's message can quote input bytes; don't pass it on.
			return nil, false, ErrInvalidJSON
		}
		if d, ok := tok.(json.Delim); ok && (d == '}' || d == ']') {
			stack = stack[:len(stack)-1]
			buf.WriteByte(byte(d))
			continue
		}
		if len(stack) > 0 {
			top := &stack[len(stack)-1]
			switch {
			case top.object && top.n%2 == 1:
				buf.WriteByte(':')
			case top.n > 0:
				buf.WriteByte(',')
			}
			top.n++
		}
		switch t := tok.(type) {
		case json.Delim: // '{' or '['
			buf.WriteByte(byte(t))
			stack = append(stack, frame{object: t == '{'})
		case string:
			red := r.String(t)
			if red != t {
				changed = true
			}
			writeString(red)
		case json.Number:
			if _, ok := r.exact[string(t)]; ok {
				changed = true
				writeString(Redacted)
			} else {
				buf.WriteString(string(t))
			}
		case bool:
			if t {
				buf.WriteString("true")
			} else {
				buf.WriteString("false")
			}
		case nil:
			buf.WriteString("null")
		default:
			return nil, false, ErrInvalidJSON
		}
	}
	return buf.Bytes(), changed, nil
}

// Payload encodes an event payload with events.MarshalPayload and redacts
// it with JSON. The worker calls it on every payload before
// events.Emitter.Emit.
func (r *Redactor) Payload(payload any) (json.RawMessage, error) {
	raw, err := events.MarshalPayload(payload)
	if err != nil {
		return nil, err
	}
	return r.JSON(raw)
}

// Stream returns a new Stream that redacts text arriving in chunks, such
// as one shell's terminal output.
func (r *Redactor) Stream() *Stream {
	return &Stream{r: r}
}

// Stream redacts a sequence of chunks so that a secret split across chunk
// boundaries is still caught. The concatenation of everything Write and
// Flush return equals Redactor.String of the concatenated input.
//
// Write holds back at most the longest pattern's length minus one byte:
// the tail that might still turn out to be a secret. Call Flush when the
// stream ends (or when the output must be complete, e.g. when the tool
// call finishes); the Stream can then be reused.
//
// A Stream is not safe for concurrent use; use one per output stream.
type Stream struct {
	r       *Redactor
	pending string
}

// Write adds chunk to the stream and returns the text that is now safe to
// emit, which may be empty.
func (s *Stream) Write(chunk string) string {
	if !s.r.active() {
		return chunk
	}
	out, tail := s.r.redact(s.pending+chunk, false)
	s.pending = tail
	return out
}

// Flush returns the held-back text, redacted, and resets the stream.
func (s *Stream) Flush() string {
	if s.pending == "" {
		return ""
	}
	out, _ := s.r.redact(s.pending, true)
	s.pending = ""
	return out
}
