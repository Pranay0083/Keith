package github

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"
)

const (
	// refreshBefore is how long before expiry a token is replaced.
	refreshBefore = 10 * time.Minute
	// refreshRetry is how long Run waits after a failed refresh.
	refreshRetry = 30 * time.Second
	// minUsable is the least remaining life for which Token still hands
	// out the current token when a refresh fails.
	minUsable = time.Minute
	// revokeTimeout bounds the best-effort revoke of a token minted
	// after the source was closed.
	revokeTimeout = 10 * time.Second
)

// ErrTokenSourceClosed is returned by Token after Revoke.
var ErrTokenSourceClosed = errors.New("github: token source revoked")

// TokenSource holds an installation token for one (installation, repos,
// permissions) triple and replaces it when 10 minutes or less remain.
// It is safe for concurrent use; concurrent callers share one refresh.
//
// The worker keeps two per session: one with SandboxPermissions, whose
// OnRefresh hook rewrites the sandbox's git token file, and one with
// WorkerPermissions behind Host. Run keeps a source fresh in the
// background so the sandbox file rotates even when nothing calls Token.
// Revoke both when the session ends.
//
// A replaced token is not revoked: a git operation in the sandbox may
// still be using it, and it expires on its own within 10 minutes.
type TokenSource struct {
	app            *App
	installationID int64
	repos          []string
	perms          Permissions

	refreshBefore time.Duration
	refreshRetry  time.Duration

	// sem serializes minting and OnRefresh delivery.
	sem  chan struct{}
	done chan struct{}

	mu        sync.Mutex
	tok       Token
	closed    bool
	onRefresh []func(Token)
	onError   []func(error)
}

// NewTokenSource checks the scope and returns a source that mints its
// first token on the first Token or Run call. repos and perms follow
// App.InstallationToken: repos must be non-empty and perms non-zero.
func NewTokenSource(app *App, installationID int64, repos []string, perms Permissions) (*TokenSource, error) {
	if app == nil {
		return nil, errors.New("github: nil App")
	}
	if installationID <= 0 {
		return nil, errors.New("github: installation ID must be positive")
	}
	if _, err := repoNames(repos); err != nil {
		return nil, err
	}
	if err := perms.validate(); err != nil {
		return nil, err
	}
	return &TokenSource{
		app:            app,
		installationID: installationID,
		repos:          slices.Clone(repos),
		perms:          perms,
		refreshBefore:  refreshBefore,
		refreshRetry:   refreshRetry,
		sem:            make(chan struct{}, 1),
		done:           make(chan struct{}),
	}, nil
}

// InstallationID returns the installation the tokens belong to.
func (s *TokenSource) InstallationID() int64 { return s.installationID }

// Repos returns the repos the tokens are scoped to, as given.
func (s *TokenSource) Repos() []string { return slices.Clone(s.repos) }

// Permissions returns the permissions the tokens carry.
func (s *TokenSource) Permissions() Permissions { return s.perms }

// OnRefresh registers fn to receive every newly minted token, the first
// one included, in mint order. Hooks run synchronously inside the mint,
// so they must be quick and must not call Token. The worker uses this to
// rewrite the sandbox's git token file.
func (s *TokenSource) OnRefresh(fn func(Token)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onRefresh = append(s.onRefresh, fn)
}

// OnError registers fn to receive every failed mint, including the
// background ones Run retries.
func (s *TokenSource) OnError(fn func(error)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onError = append(s.onError, fn)
}

// Token returns the current token, minting a new one when there is none
// or 10 minutes or less remain. If that mint fails while the current
// token still has at least a minute left, Token returns the current
// token and reports the error only to OnError hooks.
func (s *TokenSource) Token(ctx context.Context) (Token, error) {
	if t, ok, err := s.cached(); ok || err != nil {
		return t, err
	}
	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		return Token{}, ctx.Err()
	}
	defer func() { <-s.sem }()
	// Another caller may have refreshed while this one waited.
	if t, ok, err := s.cached(); ok || err != nil {
		return t, err
	}

	t, err := s.app.InstallationToken(ctx, s.installationID, s.repos, s.perms)
	if err != nil {
		s.mu.Lock()
		old, hooks := s.tok, slices.Clone(s.onError)
		s.mu.Unlock()
		for _, h := range hooks {
			h(err)
		}
		if old.Value != "" && old.ExpiresAt.Sub(s.app.clock.Now()) >= minUsable {
			return old, nil
		}
		return Token{}, err
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), revokeTimeout)
		defer cancel()
		_ = s.app.RevokeToken(rctx, t.Value)
		return Token{}, ErrTokenSourceClosed
	}
	s.tok = t
	hooks := slices.Clone(s.onRefresh)
	s.mu.Unlock()
	for _, h := range hooks {
		h(t)
	}
	return t, nil
}

// cached returns the current token if it has more than refreshBefore
// left. ok is false when a mint is needed.
func (s *TokenSource) cached() (t Token, ok bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Token{}, false, ErrTokenSourceClosed
	}
	if s.tok.Value != "" && s.tok.ExpiresAt.Sub(s.app.clock.Now()) > s.refreshBefore {
		return s.tok, true, nil
	}
	return Token{}, false, nil
}

// Run keeps the token fresh until ctx is done or Revoke is called: it
// mints when 10 minutes or less remain and retries failures every 30
// seconds. It returns ctx.Err() on cancellation and nil after Revoke.
func (s *TokenSource) Run(ctx context.Context) error {
	for {
		t, err := s.Token(ctx)
		if errors.Is(err, ErrTokenSourceClosed) {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		wait := t.ExpiresAt.Sub(s.app.clock.Now()) - s.refreshBefore
		if err != nil || wait <= 0 {
			wait = s.refreshRetry
		}
		select {
		case <-s.app.clock.After(wait):
		case <-s.done:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Revoke closes the source and revokes its current token. Token fails
// with ErrTokenSourceClosed afterwards, and Run returns. Calling Revoke
// again does nothing.
func (s *TokenSource) Revoke(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.done)
	t := s.tok
	s.tok = Token{}
	s.mu.Unlock()

	if t.Value == "" || !t.ExpiresAt.After(s.app.clock.Now()) {
		return nil
	}
	return s.app.RevokeToken(ctx, t.Value)
}
