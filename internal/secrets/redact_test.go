package secrets

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/Pranay0083/keith/internal/events"
)

func TestRedactRaw(t *testing.T) {
	r := NewRedactor([]string{"hunter2hunter2"})
	got := r.String("password=hunter2hunter2; again hunter2hunter2.")
	want := "password=[redacted]; again [redacted]."
	if got != want {
		t.Errorf("String = %q, want %q", got, want)
	}
	if s := "nothing to see"; r.String(s) != s {
		t.Errorf("String changed text without secrets")
	}
}

func TestRedactEncodings(t *testing.T) {
	// "~~~" encodes to "fn5+" and "???" to "Pz8/", so the standard and
	// URL alphabets differ; the space, '&' and '/' make the query and path
	// escapes differ from the raw value and from each other.
	secret := "a~~~b???c d&e/f"
	r := NewRedactor([]string{secret})
	forms := map[string]string{
		"std":        base64.StdEncoding.EncodeToString([]byte(secret)),
		"url":        base64.URLEncoding.EncodeToString([]byte(secret)),
		"raw std":    base64.RawStdEncoding.EncodeToString([]byte(secret)),
		"raw url":    base64.RawURLEncoding.EncodeToString([]byte(secret)),
		"query":      url.QueryEscape(secret),
		"path":       url.PathEscape(secret),
		"raw":        secret,
		"json value": strings.Trim(mustJSON(t, secret), `"`),
	}
	if forms["std"] == forms["url"] || forms["query"] == forms["path"] || forms["query"] == secret {
		t.Fatalf("test secret does not exercise distinct encodings: %v", forms)
	}
	for name, enc := range forms {
		in := "before " + enc + " after"
		if got, want := r.String(in), "before [redacted] after"; got != want {
			t.Errorf("%s (%q): String = %q, want %q", name, enc, got, want)
		}
	}
}

func TestRedactPaddedBase64Whole(t *testing.T) {
	// The padded form is longer than the unpadded one, so it wins and no
	// "=" is left behind.
	secret := "abcdefg" // 7 bytes: base64 has one "=" of padding
	r := NewRedactor([]string{secret})
	enc := base64.StdEncoding.EncodeToString([]byte(secret))
	if !strings.HasSuffix(enc, "=") {
		t.Fatalf("expected padding in %q", enc)
	}
	if got := r.String("x " + enc + " y"); got != "x [redacted] y" {
		t.Errorf("String = %q", got)
	}
}

func TestRedactEmbeddedBase64(t *testing.T) {
	// A secret inside a larger base64 blob, at every alignment: Basic auth
	// with usernames of different lengths, in both alphabets.
	for _, secret := range []string{"s3cr3tP", "s3cr3tPa", "s3cr3tPas", "correct-horse-battery-staple"} {
		r := NewRedactor([]string{secret})
		for _, user := range []string{"", "a", "ab", "abc", "abcd", "deploy-bot"} {
			for _, suffix := range []string{"", "x", "\n", "!!"} {
				blob := []byte(user + ":" + secret + suffix)
				for _, enc := range []*base64.Encoding{base64.StdEncoding, base64.URLEncoding, base64.RawStdEncoding} {
					in := "Authorization: Basic " + enc.EncodeToString(blob)
					got := r.String(in)
					if !strings.Contains(got, Redacted) {
						t.Errorf("secret %q, user %q, suffix %q: %q not redacted", secret, user, suffix, in)
					}
					if !strings.HasPrefix(got, "Authorization: Basic ") {
						t.Errorf("header mangled: %q", got)
					}
				}
			}
		}
	}
}

func TestRedactJSON(t *testing.T) {
	r := NewRedactor([]string{`pa"ss\word`, "tok-123456", "12345678"})
	for _, tc := range []struct {
		name, in, want string
	}{
		{
			name: "escaped quote and backslash in value",
			in:   `{"cmd": "echo pa\"ss\\word"}`,
			want: `{"cmd":"echo [redacted]"}`,
		},
		{
			name: "secret as key",
			in:   `{"pa\"ss\\word": 1, "tok-123456": {"x": true}}`,
			want: `{"[redacted]":1,"[redacted]":{"x":true}}`,
		},
		{
			name: "unicode escapes",
			in:   `["tok-123456"]`,
			want: `["[redacted]"]`,
		},
		{
			name: "key order preserved, nested",
			in:   `{"z": 1, "a": "tok-123456", "m": [{"y": "x", "b": "Bearer tok-123456"}], "c": null}`,
			want: `{"z":1,"a":"[redacted]","m":[{"y":"x","b":"Bearer [redacted]"}],"c":null}`,
		},
		{
			name: "number equal to secret",
			in:   `{"n": 12345678, "m": 123456789, "f": 1.5e3, "neg": -2}`,
			want: `{"n":"[redacted]","m":123456789,"f":1.5e3,"neg":-2}`,
		},
		{
			name: "html not escaped",
			in:   `{"html": "<b>&amp;</b>", "t": "tok-123456"}`,
			want: `{"html":"<b>&amp;</b>","t":"[redacted]"}`,
		},
		{
			name: "top-level string",
			in:   `"tok-123456"`,
			want: `"[redacted]"`,
		},
		{
			name: "empty containers",
			in:   `{"a": [], "b": {}, "c": [[], {}], "d": "tok-123456"}`,
			want: `{"a":[],"b":{},"c":[[],{}],"d":"[redacted]"}`,
		},
		{
			name: "double-encoded JSON",
			in:   mustJSON(t, map[string]string{"file": `{"password":"pa\"ss\\word"}`}),
			want: `{"file":"{\"password\":\"[redacted]\"}"}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := r.JSON(json.RawMessage(tc.in))
			if err != nil {
				t.Fatalf("JSON: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("JSON =\n %s\nwant\n %s", got, tc.want)
			}
			if !json.Valid(got) {
				t.Errorf("result is not valid JSON")
			}
		})
	}
}

func TestRedactJSONUnchanged(t *testing.T) {
	r := NewRedactor([]string{"tok-123456"})
	for _, in := range []string{
		`{ "a" : [1, 2, "x"] }`,
		`{"escaped": "line\nbreak \"quoted\""}`,
		`"tok-12345"`, // one byte short
	} {
		got, err := r.JSON(json.RawMessage(in))
		if err != nil {
			t.Fatalf("JSON(%s): %v", in, err)
		}
		if string(got) != in {
			t.Errorf("JSON(%s) = %s, want the input unchanged", in, got)
		}
	}
}

func TestRedactJSONInvalid(t *testing.T) {
	r := NewRedactor([]string{"tok-123456"})
	for _, in := range []string{``, `{`, `{"a": "tok-123456"`, `{"a":1} {"b":2}`, `tok-123456`, `{"a": tok-123456}`} {
		got, err := r.JSON(json.RawMessage(in))
		if !errors.Is(err, ErrInvalidJSON) {
			t.Errorf("JSON(%q): err = %v, want ErrInvalidJSON", in, err)
		}
		if got != nil {
			t.Errorf("JSON(%q) returned %q alongside an error", in, got)
		}
		if err != nil && strings.Contains(err.Error(), "tok-") {
			t.Errorf("error %q quotes the input", err)
		}
	}
	var nilR *Redactor
	if _, err := nilR.JSON(json.RawMessage(`{`)); !errors.Is(err, ErrInvalidJSON) {
		t.Errorf("nil Redactor, invalid JSON: err = %v", err)
	}
}

func TestRedactPayload(t *testing.T) {
	secret := "ghp_abcdefghijklmnop"
	r := NewRedactor([]string{secret})

	raw, err := r.Payload(events.ToolCallStartedPayload{
		CallID: "c1",
		Tool:   "bash",
		Input:  json.RawMessage(`{"command":"curl -H 'Authorization: token ` + secret + `' https://api.github.com"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	var started events.ToolCallStartedPayload
	if err := json.Unmarshal(raw, &started); err != nil {
		t.Fatalf("redacted payload does not decode: %v", err)
	}
	if want := `{"command":"curl -H 'Authorization: token [redacted]' https://api.github.com"}`; string(started.Input) != want {
		t.Errorf("Input = %s, want %s", started.Input, want)
	}

	raw, err = r.Payload(&events.ToolCallOutputPayload{CallID: "c1", Chunk: "token: " + secret + "\r\n"})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(secret)) || !bytes.Contains(raw, []byte(Redacted)) {
		t.Errorf("output payload not redacted: %s", raw)
	}

	raw, err = r.Payload(nil)
	if err != nil || string(raw) != "{}" {
		t.Errorf("Payload(nil) = %s, %v; want {}", raw, err)
	}
	if _, err := r.Payload(json.RawMessage(`{"a":`)); !errors.Is(err, ErrInvalidJSON) {
		t.Errorf("Payload(invalid raw) err = %v, want ErrInvalidJSON", err)
	}
}

func TestRedactShortAndEmptyIgnored(t *testing.T) {
	r := NewRedactor([]string{"", "a", "abc"})
	if r.active() {
		t.Error("redactor built from short values has patterns")
	}
	for _, s := range []string{"abc abc", "a", ""} {
		if got := r.String(s); got != s {
			t.Errorf("String(%q) = %q", s, got)
		}
	}
	// MinRedactLen bytes is enough.
	r = NewRedactor([]string{"", "abcd"})
	if got := r.String("x abcd y"); got != "x [redacted] y" {
		t.Errorf("String = %q", got)
	}
}

func TestRedactTrimmedValue(t *testing.T) {
	r := NewRedactor([]string{"pasted-token\n"})
	if got := r.String("value=pasted-token;"); got != "value=[redacted];" {
		t.Errorf("String = %q", got)
	}
}

func TestNilRedactor(t *testing.T) {
	for _, r := range []*Redactor{nil, NewRedactor(nil)} {
		s := "any text, tok-123456"
		if got := r.String(s); got != s {
			t.Errorf("String = %q", got)
		}
		in := json.RawMessage(`{ "a": "b" }`)
		got, err := r.JSON(in)
		if err != nil || string(got) != string(in) {
			t.Errorf("JSON = %s, %v", got, err)
		}
		raw, err := r.Payload(events.ErrorPayload{Message: "m"})
		if err != nil || string(raw) != `{"message":"m"}` {
			t.Errorf("Payload = %s, %v", raw, err)
		}
		st := r.Stream()
		if got := st.Write("abc") + st.Write("def") + st.Flush(); got != "abcdef" {
			t.Errorf("Stream = %q", got)
		}
	}
}

func TestRedactOverlapLongestFirst(t *testing.T) {
	// The order of values must not matter.
	for _, values := range [][]string{
		{"password", "password123"},
		{"password123", "password"},
	} {
		r := NewRedactor(values)
		for in, want := range map[string]string{
			"x password123 y":  "x [redacted] y",
			"x password12 y":   "x [redacted]12 y",
			"passwordpassword": "[redacted][redacted]",
		} {
			if got := r.String(in); got != want {
				t.Errorf("values %q: String(%q) = %q, want %q", values, in, got, want)
			}
		}
	}

	// A secret that is a substring of another, starting later: the
	// leftmost match covers it.
	r := NewRedactor([]string{"xyzTOKEN", "TOKEN"})
	if got := r.String("axyzTOKENb TOKEN"); got != "a[redacted]b [redacted]" {
		t.Errorf("String = %q", got)
	}

	// Documented limitation: when two different secrets overlap
	// end-to-start, the leftmost wins and the second's tail stays.
	r = NewRedactor([]string{"abcdefgh", "efghijkl"})
	if got := r.String("abcdefghijkl"); got != "[redacted]ijkl" {
		t.Errorf("String = %q", got)
	}
}

// streamAll feeds chunks through a fresh Stream and returns everything it
// emitted, checking that the held-back tail stays bounded.
func streamAll(t *testing.T, r *Redactor, chunks []string) string {
	t.Helper()
	st := r.Stream()
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString(st.Write(c))
		if r.active() && len(st.pending) >= r.maxLen {
			t.Fatalf("Stream holds back %d bytes, longest pattern is %d", len(st.pending), r.maxLen)
		}
	}
	b.WriteString(st.Flush())
	if st.pending != "" {
		t.Fatal("Flush left pending text")
	}
	return b.String()
}

func TestStreamEverySplit(t *testing.T) {
	secret := "sk-live-0123456789"
	r := NewRedactor([]string{secret, "abcdefgh", "efghijkl"})
	texts := []string{
		"export KEY=" + secret + "\n",
		secret,
		"sk-live-012 partial then " + secret + secret,
		"Basic " + base64.StdEncoding.EncodeToString([]byte("me:"+secret)) + "\r\n",
		"abcdefghijkl",
	}
	for _, text := range texts {
		want := r.String(text)
		for i := 0; i <= len(text); i++ {
			for j := i; j <= len(text); j++ {
				got := streamAll(t, r, []string{text[:i], text[i:j], text[j:]})
				if got != want {
					t.Fatalf("split %q at %d,%d: got %q, want %q", text, i, j, got, want)
				}
			}
		}
	}
}

func TestStreamRandomChunks(t *testing.T) {
	values := []string{"password", "password123", "word1234", "s3cr3t-T0ken", "abcdefgh", "efghijkl", "x+y/z=="}
	r := NewRedactor(values)

	// Build text from secrets, their encodings, prefixes of them and
	// filler drawn from overlapping alphabets, so partial matches abound.
	var pieces []string
	for _, v := range values {
		pieces = append(pieces, v, base64.StdEncoding.EncodeToString([]byte(v)), url.QueryEscape(v))
		for n := 1; n < len(v); n++ {
			pieces = append(pieces, v[:n])
		}
	}
	pieces = append(pieces, "a", "e", "p", "1", " ", "\n", "=", "/", "\x1b[0m", "é")

	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 2000 {
		var tb strings.Builder
		for range rng.IntN(40) {
			tb.WriteString(pieces[rng.IntN(len(pieces))])
		}
		text := tb.String()
		want := r.String(text)

		var chunks []string
		for rest := text; ; {
			n := rng.IntN(min(len(rest), 24) + 1) // includes empty chunks
			chunks = append(chunks, rest[:n])
			rest = rest[n:]
			if rest == "" {
				break
			}
		}
		if got := streamAll(t, r, chunks); got != want {
			t.Fatalf("trial %d: chunks %q\n got %q\nwant %q", trial, chunks, got, want)
		}
	}
}

func TestStreamReuseAfterFlush(t *testing.T) {
	r := NewRedactor([]string{"secretvalue"})
	st := r.Stream()
	out := st.Write("a secret")
	out += st.Flush()
	if out != "a secret" {
		t.Errorf("first stream = %q", out)
	}
	out = st.Write("value secretval") + st.Write("ue") + st.Flush()
	if out != "value [redacted]" {
		t.Errorf("second stream = %q", out)
	}
}

func TestRedactorConcurrent(t *testing.T) {
	r := NewRedactor([]string{"concurrent-secret"})
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 100 {
				s := fmt.Sprintf("%d %d concurrent-secret", g, i)
				if got := r.String(s); strings.Contains(got, "concurrent-secret") {
					t.Errorf("not redacted: %q", got)
					return
				}
				if _, err := r.JSON(json.RawMessage(mustJSONNoT(s))); err != nil {
					t.Error(err)
					return
				}
				st := r.Stream()
				_ = st.Write(s[:5]) + st.Write(s[5:]) + st.Flush()
			}
		}()
	}
	wg.Wait()
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func mustJSONNoT(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
