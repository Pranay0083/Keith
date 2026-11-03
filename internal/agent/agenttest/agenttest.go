// Package agenttest has fakes and checks for testing code that runs the
// agent loop: an event recorder, a scripted sandbox.Shells, a fake
// RepoHost, an input queue, a provider wrapper that records responses and
// a checker for the loop's append-only invariant.
package agenttest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// ---- events ----

// Event is one recorded Emit call.
type Event struct {
	Type    string
	Payload any
}

// Recorder records emitted events. Its Emit has Run.Emit's signature.
// It checks that each payload is the events.*Payload struct for its type
// and round-trips through JSON, and records a problem otherwise. It is
// safe for concurrent use.
type Recorder struct {
	mu       sync.Mutex
	events   []Event
	problems []string
	// Fail, if set, makes Emit return its error for matching types.
	Fail func(typ string) error
}

// Emit implements Run.Emit.
func (r *Recorder) Emit(ctx context.Context, typ string, payload any) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Fail != nil {
		if err := r.Fail(typ); err != nil {
			return err
		}
	}
	if want, ok := events.NewPayload(typ); !ok {
		r.problems = append(r.problems, fmt.Sprintf("unknown event type %q", typ))
	} else if got, wantT := reflect.TypeOf(payload), reflect.TypeOf(want).Elem(); got != wantT {
		r.problems = append(r.problems, fmt.Sprintf("%s: payload is %v, want %v", typ, got, wantT))
	} else if raw, err := events.MarshalPayload(payload); err != nil {
		r.problems = append(r.problems, fmt.Sprintf("%s: marshal: %v", typ, err))
	} else if _, err := events.Decode(events.Event{Type: typ, Payload: raw}); err != nil {
		r.problems = append(r.problems, fmt.Sprintf("%s: decode: %v", typ, err))
	}
	r.events = append(r.events, Event{Type: typ, Payload: payload})
	return nil
}

// Events returns the recorded events in order.
func (r *Recorder) Events() []Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.events)
}

// Types returns the recorded event types in order.
func (r *Recorder) Types() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.events))
	for i, e := range r.events {
		out[i] = e.Type
	}
	return out
}

// Stored returns the recorded types without ephemeral ones
// (message.delta).
func (r *Recorder) Stored() []string {
	var out []string
	for _, t := range r.Types() {
		if !events.IsEphemeral(t) {
			out = append(out, t)
		}
	}
	return out
}

// Payloads returns the payloads of every event of type typ, in order.
func Payloads[T any](r *Recorder, typ string) []T {
	var out []T
	for _, e := range r.Events() {
		if e.Type == typ {
			if p, ok := e.Payload.(T); ok {
				out = append(out, p)
			}
		}
	}
	return out
}

// Check fails t if any emitted payload had the wrong type or did not
// round-trip.
func (r *Recorder) Check(t testing.TB) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, p := range r.problems {
		t.Errorf("event: %s", p)
	}
}

// Reset forgets the recorded events.
func (r *Recorder) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = nil
}

// ---- inputs ----

// Queue is a user-input queue whose Poll has Config.Inputs' signature.
// It is safe for concurrent use.
type Queue struct {
	mu  sync.Mutex
	in  []agent.Input
	err error
}

// Push queues inputs for the next Poll.
func (q *Queue) Push(in ...agent.Input) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.in = append(q.in, in...)
}

// SetErr makes Poll fail with err (nil clears it).
func (q *Queue) SetErr(err error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.err = err
}

// Poll returns and clears the queued inputs.
func (q *Queue) Poll(ctx context.Context) ([]agent.Input, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.err != nil {
		return nil, q.err
	}
	in := q.in
	q.in = nil
	return in, nil
}

// ---- provider ----

// Call is one recorded model call.
type Call struct {
	Req  llm.Request
	Resp *llm.Response
	Err  error
}

// Recording wraps a Provider and records each request with its response
// or error. It is safe for concurrent use.
type Recording struct {
	llm.Provider

	mu    sync.Mutex
	calls []Call
	// Before, if set, runs before each call with its 0-based index.
	Before func(i int)
}

// Record wraps p.
func Record(p llm.Provider) *Recording { return &Recording{Provider: p} }

// Generate implements llm.Provider.
func (r *Recording) Generate(ctx context.Context, req llm.Request, onEvent func(llm.StreamEvent)) (*llm.Response, error) {
	r.mu.Lock()
	i := len(r.calls)
	r.calls = append(r.calls, Call{})
	before := r.Before
	r.mu.Unlock()
	if before != nil {
		before(i)
	}
	resp, err := r.Provider.Generate(ctx, req, onEvent)
	r.mu.Lock()
	r.calls[i] = Call{Req: cloneRequest(req), Resp: resp, Err: err}
	r.mu.Unlock()
	return resp, err
}

// Calls returns the recorded calls in order.
func (r *Recording) Calls() []Call {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.calls)
}

func cloneRequest(req llm.Request) llm.Request {
	b, err := json.Marshal(req)
	if err != nil {
		panic(err)
	}
	var out llm.Request
	if err := json.Unmarshal(b, &out); err != nil {
		panic(err)
	}
	return out
}

// CheckAppendOnly fails t unless every request's conversation starts with
// exactly the previous successful request's messages plus that request's
// response, with the same system prompt and tools as the first request.
// A failed request's own new user turn may grow before the retry
// succeeds, so it is never used as the baseline. It also checks that user
// messages never follow each other and that every tool_use is answered,
// in order, by a tool_result in the next message, with tool_result blocks
// first.
func CheckAppendOnly(t testing.TB, calls []Call) {
	t.Helper()
	var prev *Call
	for i := range calls {
		c := &calls[i]
		if i > 0 {
			if c.Req.System != calls[0].Req.System {
				t.Errorf("request %d: system prompt changed", i)
			}
			if !jsonEqual(c.Req.Tools, calls[0].Req.Tools) {
				t.Errorf("request %d: tools changed", i)
			}
		}
		if !c.Req.CacheSystemAndTools {
			t.Errorf("request %d: CacheSystemAndTools not set", i)
		}
		checkShape(t, i, c.Req.Messages)
		if prev != nil {
			want := append(slices.Clone(prev.Req.Messages), prev.Resp.Message)
			if len(c.Req.Messages) < len(want) {
				t.Errorf("request %d: %d messages, want at least %d (previous + response)", i, len(c.Req.Messages), len(want))
			} else if !jsonEqual(c.Req.Messages[:len(want)], want) {
				t.Errorf("request %d: history is not the previous request plus its response:\n got %s\nwant %s",
					i, mustJSON(c.Req.Messages[:len(want)]), mustJSON(want))
			}
		}
		// An empty response is retried and never appended.
		if c.Err == nil && c.Resp != nil && len(c.Resp.Message.Content) > 0 {
			prev = c
		}
	}
}

func checkShape(t testing.TB, i int, msgs []llm.Message) {
	t.Helper()
	for j, m := range msgs {
		// Roles alternate, except that a pause_turn continuation adds
		// an assistant message after an assistant message.
		switch {
		case j == 0 && m.Role != llm.RoleUser:
			t.Errorf("request %d: first message has role %s", i, m.Role)
		case m.Role != llm.RoleUser && m.Role != llm.RoleAssistant:
			t.Errorf("request %d: message %d has role %q", i, j, m.Role)
		case j > 0 && m.Role == llm.RoleUser && msgs[j-1].Role == llm.RoleUser:
			t.Errorf("request %d: messages %d and %d are both from the user", i, j-1, j)
		}
		if len(m.Content) == 0 {
			t.Errorf("request %d: message %d is empty", i, j)
		}
		if m.Role != llm.RoleAssistant {
			continue
		}
		uses := m.ToolUses()
		if len(uses) == 0 {
			continue
		}
		if j+1 >= len(msgs) {
			t.Errorf("request %d: message %d's tool calls are unanswered", i, j)
			continue
		}
		next := msgs[j+1].Content
		seenText := false
		var ids []string
		for _, b := range next {
			switch b.Type {
			case llm.BlockToolResult:
				if seenText {
					t.Errorf("request %d: message %d has a tool_result after a text block", i, j+1)
				}
				ids = append(ids, b.ToolUseID)
			default:
				seenText = true
			}
		}
		var want []string
		for _, u := range uses {
			want = append(want, u.ID)
		}
		if !slices.Equal(ids, want) {
			t.Errorf("request %d: message %d answers %v, want %v in order", i, j+1, ids, want)
		}
	}
}

func jsonEqual(a, b any) bool {
	return mustJSON(a) == mustJSON(b)
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// ToolResults returns the tool_result blocks of the last user message of
// req.
func ToolResults(req llm.Request) []llm.Block {
	if len(req.Messages) == 0 {
		return nil
	}
	var out []llm.Block
	for _, b := range req.Messages[len(req.Messages)-1].Content {
		if b.Type == llm.BlockToolResult {
			out = append(out, b)
		}
	}
	return out
}

// BlockText joins the text of blocks (and of tool_result contents).
func BlockText(blocks ...llm.Block) string {
	var parts []string
	for _, b := range blocks {
		switch b.Type {
		case llm.BlockText:
			parts = append(parts, b.Text)
		case llm.BlockToolResult:
			parts = append(parts, BlockText(b.Content...))
		}
	}
	return strings.Join(parts, "\n")
}

// ---- shells ----

// ShellCall is one recorded Shells method call.
type ShellCall struct {
	Op      string // "create", "exec", "read", "interrupt"
	Name    string
	Cmd     string
	Timeout time.Duration
}

// ExecFunc scripts Shells.Exec.
type ExecFunc func(name, cmd string, timeout time.Duration) (sandbox.ShellResult, error)

// Shells is a scripted sandbox.Shells. Exec answers with OnExec (default:
// exit 0, no output); output it returns is also passed to OnOutput, the
// way tmuxshell's WithOutput callback is called while Exec runs. Read
// returns and clears text queued with QueueRead. It is safe for
// concurrent use.
type Shells struct {
	OnExec   ExecFunc
	OnOutput func(shell string, chunk []byte)

	mu     sync.Mutex
	shells []string
	reads  map[string][]string
	calls  []ShellCall
}

var _ sandbox.Shells = (*Shells)(nil)

func (s *Shells) record(c ShellCall) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, c)
}

func (s *Shells) has(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Contains(s.shells, name)
}

// Create implements sandbox.Shells.
func (s *Shells) Create(ctx context.Context, name string) error {
	s.record(ShellCall{Op: "create", Name: name})
	s.mu.Lock()
	defer s.mu.Unlock()
	if slices.Contains(s.shells, name) {
		return fmt.Errorf("agenttest: shell %s: %w", name, sandbox.ErrShellExists)
	}
	s.shells = append(s.shells, name)
	return nil
}

// Exec implements sandbox.Shells.
func (s *Shells) Exec(ctx context.Context, name, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
	s.record(ShellCall{Op: "exec", Name: name, Cmd: cmd, Timeout: timeout})
	if !s.has(name) {
		return sandbox.ShellResult{}, fmt.Errorf("agenttest: shell %s: %w", name, sandbox.ErrNotFound)
	}
	if s.OnExec == nil {
		return sandbox.ShellResult{}, nil
	}
	res, err := s.OnExec(name, cmd, timeout)
	if err == nil && s.OnOutput != nil && res.OutputTail != "" {
		s.OnOutput(name, []byte(res.OutputTail))
	}
	return res, err
}

// QueueRead queues output for the next Read of the shell.
func (s *Shells) QueueRead(name, out string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.reads == nil {
		s.reads = map[string][]string{}
	}
	s.reads[name] = append(s.reads[name], out)
}

// Read implements sandbox.Shells.
func (s *Shells) Read(ctx context.Context, name string) (string, error) {
	s.record(ShellCall{Op: "read", Name: name})
	if !s.has(name) {
		return "", fmt.Errorf("agenttest: shell %s: %w", name, sandbox.ErrNotFound)
	}
	s.mu.Lock()
	out := strings.Join(s.reads[name], "")
	delete(s.reads, name)
	s.mu.Unlock()
	if out != "" && s.OnOutput != nil {
		s.OnOutput(name, []byte(out))
	}
	return out, nil
}

// Interrupt implements sandbox.Shells.
func (s *Shells) Interrupt(ctx context.Context, name string) error {
	s.record(ShellCall{Op: "interrupt", Name: name})
	if !s.has(name) {
		return fmt.Errorf("agenttest: shell %s: %w", name, sandbox.ErrNotFound)
	}
	return nil
}

// List implements sandbox.Shells.
func (s *Shells) List() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.shells)
}

// Calls returns the recorded calls in order.
func (s *Shells) Calls() []ShellCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.calls)
}

// ---- host ----

// Host is a fake agent.RepoHost. It is safe for concurrent use.
type Host struct {
	// Issues maps "owner/name#number" to an issue.
	Issues map[string]agent.Issue
	// PRErr, if set, is returned by OpenPR.
	PRErr error

	mu      sync.Mutex
	prs     []agent.PRRequest
	nextNum int
}

var _ agent.RepoHost = (*Host)(nil)

// ErrNoIssue is ReadIssue's error for an unknown issue.
var ErrNoIssue = errors.New("agenttest: issue not found")

// ReadIssue implements agent.RepoHost.
func (h *Host) ReadIssue(ctx context.Context, repo string, number int) (agent.Issue, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	is, ok := h.Issues[fmt.Sprintf("%s#%d", repo, number)]
	if !ok {
		return agent.Issue{}, ErrNoIssue
	}
	return is, nil
}

// OpenPR implements agent.RepoHost: PRs are numbered from 100.
func (h *Host) OpenPR(ctx context.Context, req agent.PRRequest) (agent.PR, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.prs = append(h.prs, req)
	if h.PRErr != nil {
		return agent.PR{}, h.PRErr
	}
	if h.nextNum == 0 {
		h.nextNum = 100
	}
	n := h.nextNum
	h.nextNum++
	return agent.PR{Repo: req.Repo, Number: n, URL: fmt.Sprintf("https://github.com/%s/pull/%d", req.Repo, n), Head: req.Head, Base: req.Base}, nil
}

// PRRequests returns OpenPR's requests in order.
func (h *Host) PRRequests() []agent.PRRequest {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.prs)
}

// ---- misc ----

// IDs returns a Config.NewID that numbers IDs per prefix: msg_1, msg_2.
func IDs() func(prefix string) string {
	var mu sync.Mutex
	n := map[string]int{}
	return func(prefix string) string {
		mu.Lock()
		defer mu.Unlock()
		n[prefix]++
		return fmt.Sprintf("%s_%d", prefix, n[prefix])
	}
}

// Sleeps returns a Config.Sleep that records the requested durations
// without waiting, and a func returning them.
func Sleeps() (func(context.Context, time.Duration) error, func() []time.Duration) {
	var mu sync.Mutex
	var ds []time.Duration
	return func(ctx context.Context, d time.Duration) error {
			mu.Lock()
			defer mu.Unlock()
			ds = append(ds, d)
			return ctx.Err()
		}, func() []time.Duration {
			mu.Lock()
			defer mu.Unlock()
			return slices.Clone(ds)
		}
}
