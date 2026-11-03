// Package memsandbox is an in-memory sandbox.Sandbox and sandbox.Provider
// for tests of code that drives a sandbox (the agent's tools, the worker).
//
// Files live in a map. Exec runs nothing: it calls a pluggable Handler that
// returns a scripted result. Every call is recorded for assertions:
//
//	sb := memsandbox.New(memsandbox.WithHandler(memsandbox.Route(
//		memsandbox.On(memsandbox.Prefix("go test"), memsandbox.Exit(1, "FAIL ./...", "")),
//		memsandbox.On(memsandbox.Prefix("git status"), memsandbox.OK("clean\n")),
//	)))
//	// ... run the code under test with sb ...
//	for _, req := range sb.ExecRequests() { ... }
package memsandbox

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// Name is the Provider's name.
const Name = "mem"

// Handler computes the result of one Exec. Returning a nil result and a
// nil error is treated as exit 0 with no output.
type Handler func(req sandbox.ExecRequest) (*sandbox.ExecResult, error)

// Op names a recorded call.
type Op string

const (
	OpExec     Op = "exec"
	OpRead     Op = "read"
	OpWrite    Op = "write"
	OpRemove   Op = "remove"
	OpPortURL  Op = "port_url"
	OpPause    Op = "pause"
	OpResume   Op = "resume"
	OpSnapshot Op = "snapshot"
	OpDelete   Op = "delete"
)

// Call is one recorded method call. Only the fields relevant to Op are
// set.
type Call struct {
	Op   Op
	Exec sandbox.ExecRequest // OpExec
	Path string              // OpRead, OpWrite, OpRemove (resolved, absolute)
	Data []byte              // OpWrite
	Mode fs.FileMode         // OpWrite
	Port int                 // OpPortURL
	Name string              // OpSnapshot
}

type file struct {
	data []byte
	mode fs.FileMode
}

// Sandbox is an in-memory sandbox.Sandbox. It is safe for concurrent use.
type Sandbox struct {
	id string

	mu      sync.Mutex
	files   map[string]file
	handler Handler
	calls   []Call
	paused  bool
	deleted bool
	// snapshot is called by Snapshot; set by Provider.
	snapshot func(files map[string]file, name string) string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// Option configures a Sandbox.
type Option func(*Sandbox)

// WithID sets the sandbox ID (default "mem-<n>").
func WithID(id string) Option { return func(s *Sandbox) { s.id = id } }

// WithHandler sets the Exec handler (default: every command exits 127).
func WithHandler(h Handler) Option { return func(s *Sandbox) { s.handler = h } }

// WithFiles seeds the filesystem. Relative paths are resolved against
// sandbox.WorkspaceDir. Files get mode 0644.
func WithFiles(files map[string]string) Option {
	return func(s *Sandbox) {
		for p, data := range files {
			s.files[sbutil.ResolvePath(p)] = file{data: []byte(data), mode: 0o644}
		}
	}
}

var idSeq atomic.Int64

// New returns an empty, running Sandbox.
func New(opts ...Option) *Sandbox {
	s := &Sandbox{files: map[string]file{}}
	for _, o := range opts {
		o(s)
	}
	if s.id == "" {
		s.id = fmt.Sprintf("mem-%d", idSeq.Add(1))
	}
	return s
}

// ID implements sandbox.Sandbox.
func (s *Sandbox) ID() string { return s.id }

// SetHandler replaces the Exec handler.
func (s *Sandbox) SetHandler(h Handler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handler = h
}

func (s *Sandbox) record(c Call) {
	s.calls = append(s.calls, c)
}

// usable reports why the sandbox can't run commands or touch files.
func (s *Sandbox) usable() error {
	switch {
	case s.deleted:
		return fmt.Errorf("memsandbox: sandbox %s: %w", s.id, sandbox.ErrNotFound)
	case s.paused:
		return fmt.Errorf("memsandbox: sandbox %s is paused", s.id)
	}
	return nil
}

// Exec implements sandbox.Sandbox by calling the handler.
func (s *Sandbox) Exec(ctx context.Context, req sandbox.ExecRequest, onOutput func(chunk []byte)) (*sandbox.ExecResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	req.Env = maps.Clone(req.Env)
	s.record(Call{Op: OpExec, Exec: req})
	if err := s.usable(); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	h := s.handler
	s.mu.Unlock()

	if h == nil {
		h = Exit(127, "", "memsandbox: no handler for command\n")
	}
	res, err := h(req)
	if err != nil {
		return nil, err
	}
	if res == nil {
		res = &sandbox.ExecResult{}
	}
	out := *res
	if onOutput != nil {
		if out.Stdout != "" {
			onOutput([]byte(out.Stdout))
		}
		if out.Stderr != "" {
			onOutput([]byte(out.Stderr))
		}
	}
	return &out, nil
}

// ReadFile implements sandbox.Sandbox.
func (s *Sandbox) ReadFile(ctx context.Context, path string) ([]byte, error) {
	p := sbutil.ResolvePath(path)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpRead, Path: p})
	if err := s.usable(); err != nil {
		return nil, err
	}
	f, ok := s.files[p]
	if !ok {
		return nil, sbutil.NotExist("read", p)
	}
	return slices.Clone(f.data), nil
}

// WriteFile implements sandbox.Sandbox. Mode 0 means 0644.
func (s *Sandbox) WriteFile(ctx context.Context, path string, data []byte, mode fs.FileMode) error {
	p := sbutil.ResolvePath(path)
	if mode == 0 {
		mode = 0o644
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpWrite, Path: p, Data: slices.Clone(data), Mode: mode})
	if err := s.usable(); err != nil {
		return err
	}
	s.files[p] = file{data: slices.Clone(data), mode: mode.Perm()}
	return nil
}

// RemoveFile implements sandbox.Sandbox.
func (s *Sandbox) RemoveFile(ctx context.Context, path string) error {
	p := sbutil.ResolvePath(path)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpRemove, Path: p})
	if err := s.usable(); err != nil {
		return err
	}
	if _, ok := s.files[p]; !ok {
		return sbutil.NotExist("remove", p)
	}
	delete(s.files, p)
	return nil
}

// PortURL implements sandbox.Sandbox with a fake URL.
func (s *Sandbox) PortURL(ctx context.Context, port int) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpPortURL, Port: port})
	if err := s.usable(); err != nil {
		return "", err
	}
	return fmt.Sprintf("http://%s.mem.invalid:%d", s.id, port), nil
}

// Pause implements sandbox.Sandbox.
func (s *Sandbox) Pause(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpPause})
	if s.deleted {
		return s.usable()
	}
	s.paused = true
	return nil
}

// Resume implements sandbox.Sandbox.
func (s *Sandbox) Resume(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpResume})
	if s.deleted {
		return s.usable()
	}
	s.paused = false
	return nil
}

// Snapshot implements sandbox.Sandbox. A Sandbox made by a Provider
// registers the snapshot there so Provider.Create can start from it;
// a standalone Sandbox returns an ID that nothing can use.
func (s *Sandbox) Snapshot(ctx context.Context, name string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpSnapshot, Name: name})
	if s.deleted {
		return "", s.usable()
	}
	if s.snapshot == nil {
		return "snap-" + s.id + "-" + name, nil
	}
	return s.snapshot(maps.Clone(s.files), name), nil
}

// Delete implements sandbox.Sandbox. Deleting twice returns nil.
func (s *Sandbox) Delete(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.record(Call{Op: OpDelete})
	s.deleted = true
	return nil
}

// Calls returns every recorded call in order.
func (s *Sandbox) Calls() []Call {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.calls)
}

// ExecRequests returns the requests of every Exec call in order.
func (s *Sandbox) ExecRequests() []sandbox.ExecRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []sandbox.ExecRequest
	for _, c := range s.calls {
		if c.Op == OpExec {
			out = append(out, c.Exec)
		}
	}
	return out
}

// ResetCalls forgets the recorded calls.
func (s *Sandbox) ResetCalls() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = nil
}

// File returns a file's contents and whether it exists. Relative paths are
// resolved against sandbox.WorkspaceDir.
func (s *Sandbox) File(path string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, ok := s.files[sbutil.ResolvePath(path)]
	return slices.Clone(f.data), ok
}

// FileMode returns a file's permission bits and whether it exists.
func (s *Sandbox) FileMode(path string) (fs.FileMode, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, ok := s.files[sbutil.ResolvePath(path)]
	return f.mode, ok
}

// Files returns a copy of the filesystem as path → contents.
func (s *Sandbox) Files() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]string, len(s.files))
	for p, f := range s.files {
		out[p] = string(f.data)
	}
	return out
}

// Paused reports whether the sandbox is paused.
func (s *Sandbox) Paused() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.paused
}

// Deleted reports whether the sandbox was deleted.
func (s *Sandbox) Deleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deleted
}

// ---- Handlers ----

// OK returns a Handler that exits 0 printing stdout.
func OK(stdout string) Handler { return Exit(0, stdout, "") }

// Exit returns a Handler that exits with code, printing stdout and stderr.
func Exit(code int, stdout, stderr string) Handler {
	return func(sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		return &sandbox.ExecResult{ExitCode: code, Stdout: stdout, Stderr: stderr}, nil
	}
}

// TimedOut returns a Handler that reports a timeout with partial output.
func TimedOut(stdout string) Handler {
	return func(sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		return &sandbox.ExecResult{ExitCode: -1, Stdout: stdout, TimedOut: true}, nil
	}
}

// Fail returns a Handler whose Exec fails with err (the command could not
// be run at all).
func Fail(err error) Handler {
	return func(sandbox.ExecRequest) (*sandbox.ExecResult, error) { return nil, err }
}

// Sequence returns a Handler that answers successive calls with hs in
// order and repeats the last one when they run out.
func Sequence(hs ...Handler) Handler {
	if len(hs) == 0 {
		panic("memsandbox: Sequence needs at least one handler")
	}
	var mu sync.Mutex
	i := 0
	return func(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		mu.Lock()
		h := hs[min(i, len(hs)-1)]
		i++
		mu.Unlock()
		return h(req)
	}
}

// Matcher selects commands for Route.
type Matcher func(req sandbox.ExecRequest) bool

// Prefix matches commands starting with p (after leading spaces).
func Prefix(p string) Matcher {
	return func(req sandbox.ExecRequest) bool { return strings.HasPrefix(strings.TrimSpace(req.Cmd), p) }
}

// Contains matches commands containing sub.
func Contains(sub string) Matcher {
	return func(req sandbox.ExecRequest) bool { return strings.Contains(req.Cmd, sub) }
}

// Regexp matches commands matching the regular expression expr. It panics
// if expr does not compile.
func Regexp(expr string) Matcher {
	re := regexp.MustCompile(expr)
	return func(req sandbox.ExecRequest) bool { return re.MatchString(req.Cmd) }
}

// Any matches every command.
func Any() Matcher { return func(sandbox.ExecRequest) bool { return true } }

// RouteRule pairs a Matcher with the Handler for matching commands.
type RouteRule struct {
	Match   Matcher
	Handler Handler
}

// On builds a RouteRule.
func On(m Matcher, h Handler) RouteRule { return RouteRule{Match: m, Handler: h} }

// ErrNoRoute is the error text of Route's fallback.
var ErrNoRoute = errors.New("memsandbox: no route for command")

// Route returns a Handler that uses the first rule whose Matcher matches.
// Commands no rule matches exit 127 with ErrNoRoute's text on stderr.
func Route(rules ...RouteRule) Handler {
	return func(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		for _, r := range rules {
			if r.Match(req) {
				return r.Handler(req)
			}
		}
		return &sandbox.ExecResult{ExitCode: 127, Stderr: ErrNoRoute.Error() + ": " + req.Cmd + "\n"}, nil
	}
}

// ---- Provider ----

// Provider is an in-memory sandbox.Provider. It is safe for concurrent
// use.
type Provider struct {
	// NewHandler, if set, gives each created sandbox its Exec handler.
	NewHandler func(spec sandbox.Spec) Handler

	mu        sync.Mutex
	seq       int
	sandboxes map[string]*Sandbox
	snapshots map[string]map[string]file
	specs     []sandbox.Spec
}

var _ sandbox.Provider = (*Provider)(nil)

// NewProvider returns an empty Provider.
func NewProvider() *Provider {
	return &Provider{sandboxes: map[string]*Sandbox{}, snapshots: map[string]map[string]file{}}
}

// Name implements sandbox.Provider.
func (p *Provider) Name() string { return Name }

// Create implements sandbox.Provider. A sandbox created from a
// Spec.SnapshotID starts with that snapshot's files; an unknown snapshot
// ID is an error wrapping sandbox.ErrNotFound.
func (p *Provider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Sandbox, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	spec.Env = maps.Clone(spec.Env)
	spec.Labels = maps.Clone(spec.Labels)
	spec.EgressAllowlist = slices.Clone(spec.EgressAllowlist)
	p.specs = append(p.specs, spec)

	p.seq++
	s := New(WithID(fmt.Sprintf("mem-p%d", p.seq)))
	if spec.SnapshotID != "" {
		files, ok := p.snapshots[spec.SnapshotID]
		if !ok {
			return nil, fmt.Errorf("memsandbox: snapshot %s: %w", spec.SnapshotID, sandbox.ErrNotFound)
		}
		s.files = maps.Clone(files)
	}
	if p.NewHandler != nil {
		s.handler = p.NewHandler(spec)
	}
	s.snapshot = func(files map[string]file, name string) string {
		p.mu.Lock()
		defer p.mu.Unlock()
		id := fmt.Sprintf("memsnap-%d-%s", len(p.snapshots)+1, name)
		p.snapshots[id] = files
		return id
	}
	p.sandboxes[s.id] = s
	return s, nil
}

// Get implements sandbox.Provider. Deleted sandboxes are not found.
func (p *Provider) Get(ctx context.Context, id string) (sandbox.Sandbox, error) {
	p.mu.Lock()
	s, ok := p.sandboxes[id]
	p.mu.Unlock()
	if !ok || s.Deleted() {
		return nil, fmt.Errorf("memsandbox: sandbox %s: %w", id, sandbox.ErrNotFound)
	}
	return s, nil
}

// Sandbox returns the created sandbox with that ID, for assertions.
func (p *Provider) Sandbox(id string) (*Sandbox, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	s, ok := p.sandboxes[id]
	return s, ok
}

// Specs returns the Spec of every Create call in order.
func (p *Provider) Specs() []sandbox.Spec {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.specs)
}
