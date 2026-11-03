package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/secrets"
)

// errLeaseLost is returned by the emitter once the session lease is no
// longer valid: another worker may own the session's log by now.
var errLeaseLost = errors.New("worker: session lease lost")

// emitter is the session's single path into the event log. It
//   - refuses to append once the lease is gone (the fencing gap:
//     pgevents does not check the fencing token),
//   - redacts every payload with the session's secrets (org and repo
//     secrets, plus the sandbox git token as it rotates),
//   - turns streamed shell output into tool_call.output events through
//     one secrets.Stream per shell, so a secret split across chunks is
//     still caught, and flushes those streams before each
//     tool_call.finished so a call's output precedes its finish,
//   - records the open question (store.SetOpenQuestion) before
//     question.asked is stored, so /answers accepts it as soon as the UI
//     shows it.
//
// It is safe for concurrent use; appends are serialized.
type emitter struct {
	sessionID string
	em        events.Emitter
	lease     Lease
	log       *slog.Logger
	// onQuestion runs before question.asked is appended.
	onQuestion func(ctx context.Context, questionID string) error

	mu      sync.Mutex
	values  []string
	red     *secrets.Redactor
	gen     int
	streams map[string]*shellStream
	order   []string
}

type shellStream struct {
	s    *secrets.Stream
	gen  int
	call string
	// partial holds the bytes of a UTF-8 sequence cut by a chunk
	// boundary.
	partial []byte
}

func newEmitter(sessionID string, store events.Store, bus events.Bus, l Lease, log *slog.Logger) *emitter {
	return &emitter{
		sessionID: sessionID,
		em:        events.Emitter{Store: store, Bus: bus},
		lease:     l,
		log:       log,
		streams:   map[string]*shellStream{},
	}
}

// addSecrets adds values to redact from now on.
func (e *emitter) addSecrets(values ...string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	changed := false
	for _, v := range values {
		if len(v) >= secrets.MinRedactLen && !slices.Contains(e.values, v) {
			e.values = append(e.values, v)
			changed = true
		}
	}
	if changed {
		e.red = secrets.NewRedactor(e.values)
		e.gen++
	}
}

// redactor returns the current Redactor (nil when there is nothing to
// redact, which redacts nothing).
func (e *emitter) redactor() *secrets.Redactor {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.red
}

// redact returns s with every secret value replaced.
func (e *emitter) redact(s string) string { return e.redactor().String(s) }

// Emit has agent.Run.Emit's signature.
func (e *emitter) Emit(ctx context.Context, typ string, payload any) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if typ == events.TypeToolCallFinished {
		e.flushLocked(ctx)
	}
	if typ == events.TypeQuestionAsked && e.onQuestion != nil {
		if p, ok := payload.(events.QuestionAskedPayload); ok {
			if !e.lease.Valid() {
				return errLeaseLost
			}
			if err := e.onQuestion(ctx, p.QuestionID); err != nil {
				return fmt.Errorf("worker: record open question: %w", err)
			}
		}
	}
	return e.appendLocked(ctx, typ, payload)
}

func (e *emitter) appendLocked(ctx context.Context, typ string, payload any) error {
	if !e.lease.Valid() {
		return errLeaseLost
	}
	raw, err := e.red.Payload(payload)
	if err != nil {
		return fmt.Errorf("worker: redact %s payload: %w", typ, err)
	}
	_, err = e.em.Emit(ctx, e.sessionID, typ, raw)
	return err
}

// shellOutput emits a chunk of a shell's terminal output as
// tool_call.output for callID (or, when "", the shell's last call).
func (e *emitter) shellOutput(ctx context.Context, shell string, chunk []byte, callID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := e.streams[shell]
	if st == nil {
		st = &shellStream{}
		e.streams[shell] = st
		e.order = append(e.order, shell)
	}
	if callID != "" && callID != st.call {
		// A new call: what is held back belongs to the previous one.
		e.flushStreamLocked(ctx, shell, st)
		st.call = callID
	}
	if st.s == nil || st.gen != e.gen {
		if st.s != nil {
			e.flushStreamLocked(ctx, shell, st)
		}
		st.s, st.gen = e.red.Stream(), e.gen
	}
	data := append(st.partial, chunk...)
	data, st.partial = splitUTF8(data)
	text := st.s.Write(strings.ToValidUTF8(string(data), "�"))
	e.outputLocked(ctx, shell, st.call, text)
}

// flushLocked emits what every shell stream holds back.
func (e *emitter) flushLocked(ctx context.Context) {
	for _, shell := range e.order {
		e.flushStreamLocked(ctx, shell, e.streams[shell])
	}
}

func (e *emitter) flushStreamLocked(ctx context.Context, shell string, st *shellStream) {
	if st == nil || st.s == nil {
		return
	}
	text := ""
	if len(st.partial) > 0 {
		text = st.s.Write(strings.ToValidUTF8(string(st.partial), "�"))
		st.partial = nil
	}
	text += st.s.Flush()
	e.outputLocked(ctx, shell, st.call, text)
}

func (e *emitter) outputLocked(ctx context.Context, shell, callID, text string) {
	if text == "" {
		return
	}
	err := e.appendLocked(ctx, events.TypeToolCallOutput, events.ToolCallOutputPayload{
		CallID: callID, ShellID: shell, Chunk: text,
	})
	if err != nil && !errors.Is(err, events.ErrPublish) && ctx.Err() == nil && !errors.Is(err, errLeaseLost) {
		e.log.Warn("emit shell output", "shell", shell, "err", err)
	}
}

// flush emits held-back shell output now.
func (e *emitter) flush(ctx context.Context) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.flushLocked(ctx)
}

// splitUTF8 splits b before a trailing incomplete UTF-8 sequence.
func splitUTF8(b []byte) (complete, rest []byte) {
	for i := len(b) - 1; i >= 0 && i >= len(b)-utf8.UTFMax; i-- {
		if utf8.RuneStart(b[i]) {
			if !utf8.FullRune(b[i:]) {
				return b[:i], append([]byte(nil), b[i:]...)
			}
			break
		}
	}
	return b, nil
}
