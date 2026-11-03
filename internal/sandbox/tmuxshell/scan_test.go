package tmuxshell

import (
	"fmt"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/sandbox"
)

// render feeds chunks and renders the segments as text with markers
// shown as <S> and <D:code>.
func render(sc *scanner, chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		for _, s := range sc.feed([]byte(c)) {
			switch s.kind {
			case evText:
				b.Write(s.raw)
			case evStart:
				b.WriteString("<S>")
			case evDone:
				fmt.Fprintf(&b, "<D:%d>", s.code)
			}
		}
	}
	b.WriteString("|pend=" + string(sc.pend))
	return b.String()
}

func TestScannerWhole(t *testing.T) {
	sc := newScanner("abc123")
	in := " __KEITH_ID=abc123; printf '__KEITH_S_%s__\\n' \"$__KEITH_ID\"; . /x\r\n" +
		"__KEITH_S_abc123__\r\nhello\r\nworld\r\n__KEITH_D_abc123_2__\r\n"
	got := render(sc, in)
	want := " __KEITH_ID=abc123; printf '__KEITH_S_%s__\\n' \"$__KEITH_ID\"; . /x\r\n<S>hello\r\nworld\r\n<D:2>|pend="
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
}

// Every split point of the stream must give the same result.
func TestScannerSplitEverywhere(t *testing.T) {
	in := "x\r\n__KEITH_S_id9__\r\nout\x1b[31mred\x1b[0m\r\n__KEITH_D_id9_130__\r\ntail"
	want := render(newScanner("id9"), in)
	if !strings.Contains(want, "<S>out\x1b[31mred\x1b[0m\r\n<D:130>tail") {
		t.Fatalf("baseline = %q", want)
	}
	for i := 0; i <= len(in); i++ {
		for j := i; j <= len(in); j++ {
			got := render(newScanner("id9"), in[:i], in[i:j], in[j:])
			if got != want {
				t.Fatalf("split %d,%d:\ngot  %q\nwant %q", i, j, got, want)
			}
		}
	}
}

func TestScannerANSIInsideMarker(t *testing.T) {
	sc := newScanner("ff")
	got := render(sc, "a__KEITH_\x1b[0mD_ff_0\x1b]0;title\x07__\r\nb")
	if want := "a<D:0>b|pend="; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestScannerLookalikes(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"other id", "__KEITH_D_other_0__\n", "__KEITH_D_other_0__\n|pend="},
		{"no digits", "__KEITH_D_id_x__\n", "__KEITH_D_id_x__\n|pend="},
		{"too many digits", "__KEITH_D_id_1234__\n", "__KEITH_D_id_1234__\n|pend="},
		{"single underscore", "__KEITH_D_id_1_\n", "__KEITH_D_id_1_\n|pend="},
		{"typed line", "__KEITH_ID=id; printf '__KEITH_S_%s__'\n", "__KEITH_ID=id; printf '__KEITH_S_%s__'\n|pend="},
		{"start of other", "__KEITH_S_idx__ __KEITH_S_i__\n", "__KEITH_S_idx__ __KEITH_S_i__\n|pend="},
		{"partial at end held", "ok __KEITH_D_id_1", "ok |pend=__KEITH_D_id_1"},
		{"partial lead held", "ok __KEI", "ok |pend=__KEI"},
		{"underscores pass", "a_b__c\n", "a_b__c\n|pend="},
		{"real after fake", "__KEITH_D_id_9x __KEITH_D_id_7__", "__KEITH_D_id_9x <D:7>|pend="},
		{"incomplete escape held", "out\x1b[3", "out|pend=\x1b[3"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := render(newScanner("id"), c.in); got != c.want {
				t.Fatalf("got %q want %q", got, c.want)
			}
		})
	}
}

func TestScannerPassThroughWithoutID(t *testing.T) {
	sc := newScanner("")
	if got := render(sc, "__KEITH_D_x_0__ \x1b[3"); got != "__KEITH_D_x_0__ \x1b[3|pend=" {
		t.Fatalf("got %q", got)
	}
}

func TestScannerStrayESCDoesNotHoldForever(t *testing.T) {
	sc := newScanner("id")
	long := "\x1b[" + strings.Repeat("1;", 200) // no final byte
	got := render(sc, long+"x")
	if strings.Contains(got, "|pend=\x1b") {
		t.Fatalf("held back a stray escape: %q", got)
	}
}

func TestStripperAndNormalize(t *testing.T) {
	var st stripper
	vis := string(st.write([]byte("\x1b[32mok\x1b[0m line\r\nprog 10%\rprog 100%\r\nab\bc\x1b[")))
	vis += string(st.write([]byte("1mz\x07\r\n")))
	if got, want := normalize(vis), "ok line\nprog 100%\nacz\n"; got != want {
		t.Fatalf("normalize = %q, want %q", got, want)
	}
}

func TestTail(t *testing.T) {
	if got := tail("short", 10, 0); got != "short" {
		t.Fatalf("tail = %q", got)
	}
	s := strings.Repeat("line\n", 100) // 500 bytes
	got := tail(s, 100, 0)
	if !strings.HasPrefix(got, "[... 400 earlier bytes of output omitted ...]\nline\n") || len(got) > 100+60 {
		t.Fatalf("tail = %q", got)
	}
	// Never split a UTF-8 sequence.
	got = tail(strings.Repeat("é", 100), 51, 0)
	if !strings.HasSuffix(got, strings.Repeat("é", 25)) || strings.ContainsRune(got, '\uFFFD') {
		t.Fatalf("utf-8 tail = %q", got)
	}
}

func TestParsePoll(t *testing.T) {
	ok := func(stdout string) pollResult {
		t.Helper()
		pr, err := parsePoll(&sandbox.ExecResult{Stdout: stdout}, 7)
		if err != nil {
			t.Fatal(err)
		}
		return pr
	}
	if pr := ok("0 |7|-\naGVs\nbG8=\n"); pr.state != paneAlive || pr.from != 7 || string(pr.data) != "hello" || pr.skippedDone != "" {
		t.Fatalf("alive = %+v", pr)
	}
	if pr := ok("1 3|9|__KEITH_D_ab_0__\n"); pr.state != paneDead || pr.status != 3 || pr.from != 9 || pr.skippedDone != "__KEITH_D_ab_0__" {
		t.Fatalf("dead = %+v", pr)
	}
	if pr := ok("gone|0|-\n"); pr.state != paneGone {
		t.Fatalf("gone = %+v", pr)
	}
	for _, bad := range []string{"", "junk", "0 |x|-\n", "0 |7|-\n!!!"} {
		if _, err := parsePoll(&sandbox.ExecResult{Stdout: bad}, 7); err == nil {
			t.Errorf("parsePoll(%q) succeeded", bad)
		}
	}
	if _, err := parsePoll(&sandbox.ExecResult{ExitCode: 1, Stderr: "bash: not found"}, 0); err == nil {
		t.Error("non-zero exit accepted")
	}
}
