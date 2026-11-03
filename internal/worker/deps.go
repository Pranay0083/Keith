package worker

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
	"github.com/Pranay0083/keith/internal/store"
)

// Store is the part of *store.Store the worker uses. memstore implements
// it in memory for tests and `keith run --local`.
type Store interface {
	GetSession(ctx context.Context, id string) (store.Session, error)
	SetSessionStatus(ctx context.Context, id string, status events.SessionStatus, reason string) error
	// SetSessionStatusCapped is SetSessionStatus, except that making a
	// parked session active fails (false) when its org already has max
	// active sessions.
	SetSessionStatusCapped(ctx context.Context, orgID, id string, status events.SessionStatus, reason string, max int) (bool, error)
	SetSessionSandbox(ctx context.Context, id, sandboxID, provider string) error
	IsCancelRequested(ctx context.Context, id string) (bool, error)
	ClearCancelRequest(ctx context.Context, id string) error

	SetSessionRepoBranch(ctx context.Context, sessionID, repoID, branch string) error
	SetSessionRepoPR(ctx context.Context, sessionID, repoID string, number int, url string) error
	SetSessionRepoDiff(ctx context.Context, sessionID, repoID, diff string, truncated bool) error

	ListPendingInputs(ctx context.Context, sessionID string) ([]store.Input, error)
	MarkInputsConsumed(ctx context.Context, sessionID string, ids []string) (int64, error)
	SetOpenQuestion(ctx context.Context, sessionID, questionID string) error
	ClearOpenQuestion(ctx context.Context, sessionID string) error

	SaveAgentState(ctx context.Context, st store.AgentState) error
	LoadAgentState(ctx context.Context, sessionID string) (store.AgentState, error)

	GetRepoProfile(ctx context.Context, repoID string) (store.RepoProfile, error)
	SetRepoSnapshot(ctx context.Context, repoID string, snap store.Snapshot) error
	LoadSecretCiphertexts(ctx context.Context, orgID string, repoIDs ...string) ([]store.SecretCiphertext, error)

	ListIdleSandboxSessions(ctx context.Context, idleSince time.Time, limit int) ([]store.Session, error)
	SandboxIDsInUse(ctx context.Context, ids []string) (map[string]bool, error)
	// ClearSessionSandbox clears the session's sandbox only if it is
	// still sandboxID (the reaper's compare-and-set).
	ClearSessionSandbox(ctx context.Context, id, sandboxID string) (bool, error)
	// ListStaleLiveSessions lists active sessions not updated since
	// before (the sweeper).
	ListStaleLiveSessions(ctx context.Context, before time.Time, limit int) ([]store.Session, error)
}

var _ Store = (*store.Store)(nil)

// TokenSource is a refreshing GitHub installation token
// (*github.TokenSource), or a static one for a personal access token.
type TokenSource interface {
	Token(ctx context.Context) (github.Token, error)
	// Run keeps the token fresh until ctx ends or Revoke is called.
	Run(ctx context.Context) error
	Revoke(ctx context.Context) error
	// OnRefresh registers fn for every newly minted token, the first one
	// included. fn must be quick.
	OnRefresh(fn func(github.Token))
}

var _ TokenSource = (*github.TokenSource)(nil)

// GitHub mints a session's two credentials. They never mix: the sandbox
// token reaches the sandbox (through the git credential helper) and
// nothing else; the worker host's token stays in the worker.
type GitHub interface {
	// SandboxTokens returns a source of sandbox tokens: contents: write on
	// exactly repos (owner/name).
	SandboxTokens(installationID int64, repos []string) (TokenSource, error)
	// WorkerHost returns the RepoHost behind read_issue and open_pr, with
	// WorkerPermissions on repos, and its token source to revoke when the
	// session parks.
	WorkerHost(installationID int64, repos []string) (agent.RepoHost, TokenSource, error)
}

// NewAppGitHub returns the production GitHub: installation tokens of the
// GitHub App.
func NewAppGitHub(app *github.App) GitHub { return appGitHub{app} }

type appGitHub struct{ app *github.App }

func (a appGitHub) SandboxTokens(installationID int64, repos []string) (TokenSource, error) {
	return github.NewTokenSource(a.app, installationID, repos, github.SandboxPermissions)
}

func (a appGitHub) WorkerHost(installationID int64, repos []string) (agent.RepoHost, TokenSource, error) {
	ts, err := github.NewTokenSource(a.app, installationID, repos, github.WorkerPermissions)
	if err != nil {
		return nil, nil, err
	}
	return github.NewHost(ts), ts, nil
}

// NewPATGitHub returns a GitHub that uses one personal access token for
// both the sandbox (clone and push, through the same credential helper
// file) and the worker host, for `keith run --local`. installationID is
// ignored.
func NewPATGitHub(token string, opts github.PATHostOptions) GitHub {
	return patGitHub{token: token, opts: opts}
}

type patGitHub struct {
	token string
	opts  github.PATHostOptions
}

func (p patGitHub) SandboxTokens(_ int64, repos []string) (TokenSource, error) {
	if p.token == "" {
		return nil, errors.New("worker: empty GitHub token")
	}
	return NewStaticTokenSource(p.token), nil
}

func (p patGitHub) WorkerHost(_ int64, repos []string) (agent.RepoHost, TokenSource, error) {
	h, err := github.NewPATHost(p.token, repos, p.opts)
	if err != nil {
		return nil, nil, err
	}
	return h, NewStaticTokenSource(p.token), nil
}

// NewStaticTokenSource returns a TokenSource that always hands out value
// (a personal access token), never refreshes and revokes nothing.
func NewStaticTokenSource(value string) TokenSource {
	return &staticTokens{tok: github.Token{Value: value, ExpiresAt: time.Now().Add(100 * 365 * 24 * time.Hour)}}
}

type staticTokens struct {
	tok github.Token

	mu      sync.Mutex
	hooks   []func(github.Token)
	handed  bool
	revoked bool
	done    chan struct{}
}

func (s *staticTokens) Token(context.Context) (github.Token, error) {
	s.mu.Lock()
	if s.revoked {
		s.mu.Unlock()
		return github.Token{}, github.ErrTokenSourceClosed
	}
	first := !s.handed
	s.handed = true
	hooks := slices.Clone(s.hooks)
	s.mu.Unlock()
	if first {
		for _, h := range hooks {
			h(s.tok)
		}
	}
	return s.tok, nil
}

func (s *staticTokens) Run(ctx context.Context) error {
	if _, err := s.Token(ctx); err != nil {
		return nil
	}
	s.mu.Lock()
	if s.done == nil {
		s.done = make(chan struct{})
	}
	done := s.done
	s.mu.Unlock()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		return nil
	}
}

func (s *staticTokens) Revoke(context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.revoked {
		s.revoked = true
		if s.done == nil {
			s.done = make(chan struct{})
		}
		close(s.done)
	}
	return nil
}

func (s *staticTokens) OnRefresh(fn func(github.Token)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hooks = append(s.hooks, fn)
}

// Shells is the shell manager the worker needs: sandbox.Shells plus
// adopting shells a previous worker left (Reattach) and closing one.
// *tmuxshell.Manager implements it.
type Shells interface {
	sandbox.Shells
	Reattach(ctx context.Context) ([]string, error)
	Close(ctx context.Context, name string) error
}

var _ Shells = (*tmuxshell.Manager)(nil)

// ShellsFactory builds the Shells for a sandbox. user is the sandbox
// user shells run as ("" for the sandbox's default). onOutput receives
// live terminal output.
type ShellsFactory func(sb sandbox.Sandbox, user string, onOutput func(shell string, chunk []byte)) Shells

// TmuxShells is the default ShellsFactory.
func TmuxShells(sb sandbox.Sandbox, user string, onOutput func(shell string, chunk []byte)) Shells {
	opts := []tmuxshell.Option{tmuxshell.WithOutput(onOutput)}
	if user != "" {
		opts = append(opts, tmuxshell.WithUser(user))
	}
	return tmuxshell.New(sb, opts...)
}

// Leaser takes session leases. Acquire returns an error wrapping
// lease.ErrHeld when another worker holds the key.
type Leaser interface {
	Acquire(ctx context.Context, key string, ttl time.Duration) (Lease, error)
}

// Lease is a held lease (*lease.Lease).
type Lease interface {
	// Valid reports, without a round trip, whether the lease is neither
	// lost nor past its deadline.
	Valid() bool
	// KeepAlive renews the lease until ctx ends; it sends one error if the
	// lease is lost.
	KeepAlive(ctx context.Context) <-chan error
	Release(ctx context.Context) error
}

var _ Lease = (*lease.Lease)(nil)

// NewRedisLeaser returns the production Leaser.
func NewRedisLeaser(m *lease.Manager) Leaser { return redisLeaser{m} }

type redisLeaser struct{ m *lease.Manager }

func (r redisLeaser) Acquire(ctx context.Context, key string, ttl time.Duration) (Lease, error) {
	l, err := r.m.Acquire(ctx, key, ttl)
	if err != nil {
		return nil, err
	}
	return l, nil
}

// NewMemLeaser returns an in-process Leaser, for tests and
// `keith run --local`. Its leases never expire on their own.
func NewMemLeaser() *MemLeaser { return &MemLeaser{held: map[string]*memLease{}} }

// MemLeaser is an in-process Leaser.
type MemLeaser struct {
	mu   sync.Mutex
	held map[string]*memLease
}

// Acquire implements Leaser.
func (m *MemLeaser) Acquire(_ context.Context, key string, _ time.Duration) (Lease, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.held[key]; ok {
		return nil, fmt.Errorf("worker: lease %s: %w", key, lease.ErrHeld)
	}
	l := &memLease{m: m, key: key, lost: make(chan struct{})}
	m.held[key] = l
	return l, nil
}

// Lose marks the holder of key as having lost its lease (as if it
// expired and another worker took it), for tests.
func (m *MemLeaser) Lose(key string) {
	m.mu.Lock()
	l := m.held[key]
	delete(m.held, key)
	m.mu.Unlock()
	if l != nil {
		l.markLost()
	}
}

// Held reports whether key is held.
func (m *MemLeaser) Held(key string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.held[key]
	return ok
}

type memLease struct {
	m    *MemLeaser
	key  string
	once sync.Once
	lost chan struct{}

	mu       sync.Mutex
	isLost   bool
	released bool
}

func (l *memLease) markLost() {
	l.once.Do(func() {
		l.mu.Lock()
		l.isLost = true
		l.mu.Unlock()
		close(l.lost)
	})
}

func (l *memLease) Valid() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return !l.isLost && !l.released
}

func (l *memLease) KeepAlive(ctx context.Context) <-chan error {
	out := make(chan error, 1)
	go func() {
		defer close(out)
		select {
		case <-ctx.Done():
		case <-l.lost:
			l.mu.Lock()
			released := l.released
			l.mu.Unlock()
			if !released {
				out <- fmt.Errorf("worker: lease %s: %w", l.key, lease.ErrLost)
			}
		}
	}()
	return out
}

func (l *memLease) Release(context.Context) error {
	l.mu.Lock()
	if l.released {
		l.mu.Unlock()
		return nil
	}
	l.released = true
	lost := l.isLost
	l.mu.Unlock()
	if lost {
		return fmt.Errorf("worker: lease %s: %w", l.key, lease.ErrLost)
	}
	l.m.mu.Lock()
	if l.m.held[l.key] == l {
		delete(l.m.held, l.key)
	}
	l.m.mu.Unlock()
	l.once.Do(func() { close(l.lost) })
	return nil
}
