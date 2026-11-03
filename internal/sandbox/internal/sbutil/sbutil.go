// Package sbutil holds small helpers shared by the sandbox providers and
// the shell manager. It is internal to internal/sandbox.
package sbutil

import (
	"io/fs"
	"path"
	"strings"
	"sync"

	"github.com/Pranay0083/keith/internal/sandbox"
)

// ResolvePath makes p absolute inside the sandbox: a relative path is
// joined to sandbox.WorkspaceDir. The result is cleaned.
func ResolvePath(p string) string {
	if !path.IsAbs(p) {
		p = path.Join(sandbox.WorkspaceDir, p)
	}
	return path.Clean(p)
}

// Quote returns s as a single POSIX shell word.
func Quote(s string) string {
	if s == "" {
		return "''"
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// NotExist returns an *fs.PathError wrapping fs.ErrNotExist.
func NotExist(op, p string) error {
	return &fs.PathError{Op: op, Path: p, Err: fs.ErrNotExist}
}

// TailBuffer keeps the last Max bytes written to it. It is safe for
// concurrent use.
type TailBuffer struct {
	Max int

	mu      sync.Mutex
	buf     []byte
	dropped int64
}

// NewTailBuffer returns a TailBuffer that keeps max bytes.
func NewTailBuffer(max int) *TailBuffer { return &TailBuffer{Max: max} }

// Write appends p, dropping the oldest bytes beyond Max. It never fails.
func (t *TailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := len(p)
	if t.Max <= 0 {
		t.dropped += int64(n)
		return n, nil
	}
	if len(p) >= t.Max {
		t.dropped += int64(len(t.buf) + len(p) - t.Max)
		t.buf = append(t.buf[:0], p[len(p)-t.Max:]...)
		return n, nil
	}
	if over := len(t.buf) + len(p) - t.Max; over > 0 {
		t.dropped += int64(over)
		t.buf = append(t.buf[:0], t.buf[over:]...)
	}
	t.buf = append(t.buf, p...)
	return n, nil
}

// String returns the kept bytes.
func (t *TailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}

// Dropped reports how many bytes were discarded from the front.
func (t *TailBuffer) Dropped() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.dropped
}

// Reset empties the buffer.
func (t *TailBuffer) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = t.buf[:0]
	t.dropped = 0
}
