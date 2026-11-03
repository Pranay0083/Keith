package store

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestNewInputID(t *testing.T) {
	seen := map[string]bool{}
	prev := ""
	for range 1000 {
		id := NewInputID()
		if !ValidInputID(id) || len(id) != 28 || !strings.HasPrefix(id, "m_0") {
			t.Fatalf("NewInputID() = %q", id)
		}
		if seen[id] {
			t.Fatalf("duplicate ID %q", id)
		}
		seen[id] = true
		// UUIDv7 IDs share the millisecond prefix; later IDs never sort
		// before an earlier millisecond.
		if prev != "" && id[:12] < prev[:12] {
			t.Fatalf("ID %q sorts before %q", id, prev)
		}
		prev = id
	}
	for _, bad := range []string{"", "m_", "x_01J", "m_01J-x", "m_" + strings.Repeat("A", 70)} {
		if ValidInputID(bad) {
			t.Errorf("ValidInputID(%q) = true", bad)
		}
	}
}

func TestCleanDiff(t *testing.T) {
	if got, cut := cleanDiff("a\x00b"); got != "a�b" || cut {
		t.Errorf("NUL: %q %v", got, cut)
	}
	if got, cut := cleanDiff("a\xffb"); got != "a�b" || cut {
		t.Errorf("invalid UTF-8: %q %v", got, cut)
	}
	// A multi-byte rune straddling the limit is dropped whole.
	big := strings.Repeat("a", MaxDiffBytes-1) + "é" + "tail"
	got, cut := cleanDiff(big)
	if !cut || len(got) != MaxDiffBytes-1 || !utf8.ValidString(got) {
		t.Errorf("cut: len %d cut %v valid %v", len(got), cut, utf8.ValidString(got))
	}
	exact := strings.Repeat("b", MaxDiffBytes)
	if got, cut := cleanDiff(exact); cut || len(got) != MaxDiffBytes {
		t.Errorf("exact: len %d cut %v", len(got), cut)
	}
}
