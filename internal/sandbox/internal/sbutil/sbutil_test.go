package sbutil

import (
	"errors"
	"io/fs"
	"os/exec"
	"strings"
	"testing"
)

func TestResolvePath(t *testing.T) {
	for in, want := range map[string]string{
		"repo/a.go":       "/workspace/repo/a.go",
		"/etc/hosts":      "/etc/hosts",
		"repo/../x":       "/workspace/x",
		"/workspace/./y/": "/workspace/y",
	} {
		if got := ResolvePath(in); got != want {
			t.Errorf("ResolvePath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestQuote(t *testing.T) {
	for _, s := range []string{"", "plain", "it's", `a "b" $c \d`, "new\nline"} {
		out, err := exec.Command("sh", "-c", "printf %s "+Quote(s)).Output()
		if err != nil || string(out) != s {
			t.Errorf("Quote(%q) round trip = %q, %v", s, out, err)
		}
	}
}

func TestTailBuffer(t *testing.T) {
	b := NewTailBuffer(5)
	b.Write([]byte("abc"))
	b.Write([]byte("defg"))
	if b.String() != "cdefg" || b.Dropped() != 2 {
		t.Fatalf("got %q dropped %d", b.String(), b.Dropped())
	}
	b.Write([]byte(strings.Repeat("x", 10)))
	if b.String() != "xxxxx" || b.Dropped() != 12 {
		t.Fatalf("got %q dropped %d", b.String(), b.Dropped())
	}
	b.Reset()
	if b.String() != "" || b.Dropped() != 0 {
		t.Fatal("Reset")
	}
}

func TestNotExist(t *testing.T) {
	if err := NotExist("read", "/x"); !errors.Is(err, fs.ErrNotExist) {
		t.Fatal(err)
	}
}
