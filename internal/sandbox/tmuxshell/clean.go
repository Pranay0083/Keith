package tmuxshell

import (
	"fmt"
	"strings"
)

// stripper removes terminal escape sequences from a byte stream, keeping
// an incomplete sequence at the end of a chunk for the next one.
type stripper struct {
	pend []byte
}

// write returns the visible bytes of p.
func (s *stripper) write(p []byte) []byte {
	buf := p
	if len(s.pend) > 0 {
		buf = append(s.pend, p...)
		s.pend = nil
	}
	vis, _, cut := project(buf, 0)
	if cut < len(buf) {
		s.pend = append([]byte(nil), buf[cut:]...)
	}
	return vis
}

func (s *stripper) reset() { s.pend = nil }

// normalize turns terminal text (escape sequences already stripped) into
// what a reader would see: CRLF becomes LF, a bare CR rewinds the line (so
// a progress bar keeps only its last state), backspace erases the
// previous character, and other control characters except tab are
// dropped.
func normalize(s string) string {
	if s == "" {
		return s
	}
	var out strings.Builder
	out.Grow(len(s))
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		line = strings.TrimRight(line, "\r")
		if j := strings.LastIndexByte(line, '\r'); j >= 0 {
			line = line[j+1:]
		}
		var b []byte
		for k := 0; k < len(line); k++ {
			c := line[k]
			switch {
			case c == '\b':
				if len(b) > 0 {
					b = b[:len(b)-1]
				}
			case c == '\t' || c >= 0x20 && c != 0x7f:
				b = append(b, c)
			}
		}
		out.Write(b)
		if i < len(lines)-1 {
			out.WriteByte('\n')
		}
	}
	return out.String()
}

// tail keeps at most max bytes of the end of s, cutting at a line start
// when one is near, and says how much was dropped.
func tail(s string, max int, dropped int64) string {
	if max <= 0 {
		return ""
	}
	if len(s) > max {
		cut := len(s) - max
		if nl := strings.IndexByte(s[cut:], '\n'); s[cut-1] != '\n' && nl >= 0 && nl < max/4 {
			cut += nl + 1
		}
		for cut < len(s) && s[cut]&0xc0 == 0x80 { // don't split a UTF-8 sequence
			cut++
		}
		dropped += int64(cut)
		s = s[cut:]
	}
	if dropped > 0 {
		return fmt.Sprintf("[... %d earlier bytes of output omitted ...]\n", dropped) + s
	}
	return s
}
