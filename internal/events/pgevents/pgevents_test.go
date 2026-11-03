package pgevents

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Pranay0083/keith/internal/events"
)

func TestSanitizeNUL(t *testing.T) {
	cases := map[string]string{
		`{"a":"x"}`:                `{"a":"x"}`,
		`{"a":"x\u0000y"}`:         `{"a":"x�y"}`,
		`{"a":"\u0000\u0000"}`:     `{"a":"��"}`,
		`{"a":"\\u0000"}`:          `{"a":"\\u0000"}`, // an escaped backslash, then text
		`{"a":"\\\u0000"}`:         `{"a":"\\�"}`,
		`{"\u0000":1}`:             `{"�":1}`,
		`{"a":"\u00001"}`:          `{"a":"�1"}`,
		`{"a":"\u0001","b":"\\"}`:  `{"a":"\u0001","b":"\\"}`,
		`["\"\u0000\"","\u00000"]`: `["\"�\"","�0"]`,
	}
	for in, want := range cases {
		got := string(sanitizeNUL(json.RawMessage(in)))
		if got != want {
			t.Errorf("sanitizeNUL(%s) = %s, want %s", in, got, want)
		}
		if !json.Valid([]byte(got)) {
			t.Errorf("sanitizeNUL(%s) is invalid JSON", in)
		}
	}

	// What encoding/json produces for terminal output with NULs.
	raw, err := json.Marshal(events.ToolCallOutputPayload{CallID: "c", Chunk: "a\x00b"})
	if err != nil {
		t.Fatal(err)
	}
	var p events.ToolCallOutputPayload
	if err := json.Unmarshal(sanitizeNUL(raw), &p); err != nil {
		t.Fatal(err)
	}
	if p.Chunk != "a�b" {
		t.Errorf("chunk = %q", p.Chunk)
	}
}

func TestCompact(t *testing.T) {
	// jsonb's text form puts spaces after ':' and ','.
	got := string(compact(`{"a": 1, "b": [true, null, "x, y: z"]}`))
	if want := `{"a":1,"b":[true,null,"x, y: z"]}`; got != want {
		t.Errorf("compact = %s, want %s", got, want)
	}
}

func TestAppendRejectsEphemeralWithoutDB(t *testing.T) {
	s := New(nil) // never touched: the type is checked first
	_, err := s.Append(context.Background(), "s", events.TypeMessageDelta, nil)
	if !errors.Is(err, events.ErrEphemeral) {
		t.Fatalf("err = %v, want ErrEphemeral", err)
	}
}
