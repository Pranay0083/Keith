package tmuxshell

import (
	"bytes"
	"strconv"
)

// Markers. Every Exec types a line that sets __KEITH_ID to a fresh random
// ID and prints the start marker; the shell's PROMPT_COMMAND prints the
// done marker with $? when the command (or its Ctrl-C) returns:
//
//	__KEITH_S_<id>__            command output starts after this
//	__KEITH_D_<id>_<status>__   command finished with exit status
//
// The typed line itself contains the marker text only in pieces (the ID
// is passed to printf as an argument), so its echo never matches.
const (
	markerLead  = "__KEITH_"
	startPrefix = "__KEITH_S_"
	donePrefix  = "__KEITH_D_"
	markerTail  = "__"
	maxCodeLen  = 3 // $? is 0..255
)

type eventKind int

const (
	evText eventKind = iota
	evStart
	evDone
)

// segment is a piece of the terminal stream: raw output bytes (evText) or
// a marker (evStart, evDone with code).
type segment struct {
	kind eventKind
	raw  []byte
	code int
}

// scanner splits a raw terminal stream into text and the markers of one
// command ID. It is robust to:
//   - markers split across reads (a possible marker prefix at the end of a
//     chunk is held back until the next one),
//   - ANSI escape sequences, which are skipped when matching (so a marker
//     with an escape in the middle still matches and is removed whole),
//     and incomplete escape sequences at the end of a chunk (held back),
//   - marker-like text that is not ours (another ID, a malformed status),
//     which passes through as text.
//
// With an empty ID it passes text through untouched.
type scanner struct {
	id    string
	start []byte // __KEITH_S_<id>__
	done  []byte // __KEITH_D_<id>_
	pend  []byte
	// skipNL drops the line ending that follows a marker, which may arrive
	// in the next chunk: 1 = expect "\r" or "\n", 2 = expect "\n".
	skipNL int
}

func newScanner(id string) *scanner {
	s := &scanner{}
	s.reset(id)
	return s
}

// reset switches to a new ID and drops held-back bytes.
func (s *scanner) reset(id string) {
	s.id = id
	s.pend = nil
	s.skipNL = 0
	if id == "" {
		s.start, s.done = nil, nil
		return
	}
	s.start = []byte(startPrefix + id + markerTail)
	s.done = []byte(donePrefix + id + "_")
}

// pending returns the held-back bytes and forgets them.
func (s *scanner) flush() []byte {
	p := s.pend
	s.pend = nil
	return p
}

// feed scans the next chunk of the stream.
func (s *scanner) feed(chunk []byte) []segment {
	buf := chunk
	if len(s.pend) > 0 {
		buf = append(s.pend, chunk...)
		s.pend = nil
	}
	var segs []segment
	text := func(b []byte) {
		if len(b) == 0 {
			return
		}
		if n := len(segs); n > 0 && segs[n-1].kind == evText {
			segs[n-1].raw = append(segs[n-1].raw, b...)
			return
		}
		segs = append(segs, segment{kind: evText, raw: append([]byte(nil), b...)})
	}

	pos := s.dropNL(buf, 0)
	if s.id == "" {
		text(buf[pos:])
		return segs
	}

	vis, idx, cut := project(buf, pos)
	hold := cut // raw offset from which bytes are held back
	for j := 0; j < len(vis); {
		k := bytes.IndexByte(vis[j:], '_')
		if k < 0 {
			break
		}
		j += k
		kind, code, n := s.match(vis[j:])
		switch kind {
		case matchNone:
			j++
			continue
		case matchPartial:
			if idx[j] < hold {
				hold = idx[j]
			}
			j = len(vis) // stop: everything from here is held back
			continue
		}
		// Full marker: vis[j:j+n] → raw [idx[j], idx[j+n-1]+1).
		rawStart, rawEnd := idx[j], idx[j+n-1]+1
		text(buf[pos:rawStart])
		if kind == matchStart {
			segs = append(segs, segment{kind: evStart})
		} else {
			segs = append(segs, segment{kind: evDone, code: code})
		}
		s.skipNL = 1
		pos = s.dropNL(buf, rawEnd)
		j += n
		// Re-align j to the first visible byte at or after pos.
		for j < len(vis) && idx[j] < pos {
			j++
		}
	}
	if hold < pos {
		hold = pos
	}
	text(buf[pos:hold])
	if hold < len(buf) {
		s.pend = append([]byte(nil), buf[hold:]...)
	}
	return segs
}

// dropNL skips the "\r\n" (or "\n") after a marker when skipNL is set.
func (s *scanner) dropNL(buf []byte, pos int) int {
	for s.skipNL > 0 && pos < len(buf) {
		switch {
		case s.skipNL == 1 && buf[pos] == '\r':
			s.skipNL = 2
			pos++
		case buf[pos] == '\n':
			s.skipNL = 0
			pos++
		default:
			s.skipNL = 0
		}
	}
	return pos
}

type matchKind int

const (
	matchNone matchKind = iota
	matchPartial
	matchStart
	matchDone
)

// match checks whether v starts with one of our markers. It returns the
// marker's visible length for a full match, and matchPartial when v is a
// proper prefix of a possible marker (more input is needed to decide).
func (s *scanner) match(v []byte) (kind matchKind, code, n int) {
	// Start marker.
	if pk := prefixMatch(v, s.start); pk == matchPartial {
		kind = matchPartial
	} else if pk == matchStart {
		return matchStart, 0, len(s.start)
	}
	// Done marker: prefix, 1-3 digits, "__".
	switch prefixMatch(v, s.done) {
	case matchPartial:
		return matchPartial, 0, 0
	case matchNone:
		return kind, 0, 0
	}
	i := len(s.done)
	d := 0
	for i+d < len(v) && d < maxCodeLen+1 && v[i+d] >= '0' && v[i+d] <= '9' {
		d++
	}
	if d > maxCodeLen {
		return kind, 0, 0
	}
	if i+d == len(v) {
		return matchPartial, 0, 0
	}
	if d == 0 {
		return kind, 0, 0
	}
	rest := v[i+d:]
	switch {
	case len(rest) >= 2 && rest[0] == '_' && rest[1] == '_':
		c, _ := strconv.Atoi(string(v[i : i+d]))
		return matchDone, c, i + d + 2
	case len(rest) == 1 && rest[0] == '_':
		return matchPartial, 0, 0
	}
	return kind, 0, 0
}

// prefixMatch reports whether v starts with tok (matchStart is used as
// "full"), is a proper prefix of tok (matchPartial), or neither.
func prefixMatch(v, tok []byte) matchKind {
	if len(tok) == 0 {
		return matchNone
	}
	if len(v) >= len(tok) {
		if bytes.Equal(v[:len(tok)], tok) {
			return matchStart
		}
		return matchNone
	}
	if bytes.Equal(v, tok[:len(v)]) {
		return matchPartial
	}
	return matchNone
}

// project returns the visible bytes of buf[from:] (escape sequences
// removed), the raw index of each visible byte, and the raw offset of an
// incomplete escape sequence at the end (len(buf) if there is none).
func project(buf []byte, from int) (vis []byte, idx []int, cut int) {
	vis = make([]byte, 0, len(buf)-from)
	idx = make([]int, 0, len(buf)-from)
	cut = len(buf)
	for i := from; i < len(buf); {
		if buf[i] == 0x1b {
			n, ok := escLen(buf[i:])
			if !ok {
				cut = i
				break
			}
			i += n
			continue
		}
		vis = append(vis, buf[i])
		idx = append(idx, i)
		i++
	}
	return vis, idx, cut
}

// maxEscLen bounds how long an unterminated escape sequence may be before
// it is treated as text, so a stray ESC can't hold back output forever.
const maxEscLen = 256

// escLen returns the length of the escape sequence at the start of b
// (b[0] == ESC). ok is false when the sequence is incomplete.
func escLen(b []byte) (n int, ok bool) {
	if len(b) < 2 {
		return 0, false
	}
	switch b[1] {
	case '[': // CSI: parameters and intermediates, then a final byte 0x40-0x7e.
		for i := 2; i < len(b) && i < maxEscLen; i++ {
			if b[i] >= 0x40 && b[i] <= 0x7e {
				return i + 1, true
			}
		}
	case ']', 'P', 'X', '^', '_': // OSC, DCS, SOS, PM, APC: until BEL or ST.
		for i := 2; i < len(b) && i < maxEscLen; i++ {
			if b[i] == 0x07 {
				return i + 1, true
			}
			if b[i] == 0x1b && i+1 < len(b) && b[i+1] == '\\' {
				return i + 2, true
			}
		}
	case '(', ')', '*', '+', '#', '%': // charset selection etc.: one more byte.
		if len(b) >= 3 {
			return 3, true
		}
		return 0, false
	default: // two-byte sequence (ESC =, ESC >, ESC 7, ...).
		return 2, true
	}
	if len(b) >= maxEscLen {
		return 1, true // give up: drop the lone ESC
	}
	return 0, false
}
