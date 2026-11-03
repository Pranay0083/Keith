// Package api is keith api: the control plane's HTTP and WebSocket
// server, implementing docs/api.md.
//
// The API writes rows, enqueues jobs and streams events. It never runs
// the agent loop and never writes events: the worker holding a session's
// lease is the only event writer. User input goes into session_inputs and
// a resume_session job; Stop sets cancel_requested_at and publishes a
// control signal. This package never imports internal/agent.
//
// Its GitHub dependency is the GitHub interface (satisfied by
// *github.App), so tests run against a fake and never call GitHub.
package api

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
)

// GitHub is the part of *github.App the API uses.
type GitHub interface {
	AuthCodeURL(state string) (string, error)
	Exchange(ctx context.Context, code string) (github.UserToken, error)
	GetUser(ctx context.Context, userToken string) (github.User, error)
	UserCanAccessInstallation(ctx context.Context, userToken string, installationID int64) (bool, error)
	GetInstallation(ctx context.Context, installationID int64) (github.Installation, error)
	ListInstallationRepos(ctx context.Context, installationID int64) ([]github.Repo, error)
	InstallURL(appSlug, state string) (string, error)
	Slug(ctx context.Context) (string, error)
}

var _ GitHub = (*github.App)(nil)

// Deduper records webhook delivery IDs. Its Redis implementation is
// RedisDeduper.
type Deduper interface {
	// FirstSeen marks key as seen for ttl and reports whether it was new.
	FirstSeen(ctx context.Context, key string, ttl time.Duration) (bool, error)
	// Forget removes key, so a delivery whose enqueue failed can be
	// redelivered.
	Forget(ctx context.Context, key string) error
}

// Deps are the Server's collaborators.
type Deps struct {
	Config *config.Config
	Store  *store.Store
	// Events reads stored events for the WebSocket; nil means
	// Store.Events().
	Events events.Store
	Bus    events.Bus
	Queue  queue.Enqueuer
	// Control publishes Stop's cancel signal.
	Control control.Publisher
	// GitHub is the App configured with the sign-in redirect URL
	// (PUBLIC_URL/auth/github/callback).
	GitHub GitHub
	// SetupGitHub exchanges the OAuth code GitHub sends after an install.
	// It should be an App without a RedirectURL, since that
	// authorization went to the App's own callback, not the sign-in one.
	// nil means GitHub.
	SetupGitHub GitHub
	Dedupe      Deduper
	// Keyring seals secret values (SECRETS_KEY).
	Keyring *secrets.Keyring
	// Ready reports whether Postgres and Redis are reachable, for
	// /healthz. nil means always ready.
	Ready  func(ctx context.Context) error
	Logger *slog.Logger
	// Now is the clock; nil means time.Now.
	Now func() time.Time
}

// Timing and limits.
const (
	sessionTTL      = 30 * 24 * time.Hour
	oauthStateTTL   = 10 * time.Minute
	setupStateTTL   = 15 * time.Minute
	webhookDedupTTL = 7 * 24 * time.Hour
	// rateLimitRetryAfter is the Retry-After for the per-org cap.
	rateLimitRetryAfter = 30
)

// Cookie names.
const (
	sessionCookie    = "keith_session"
	oauthStateCookie = "keith_oauth_state"
)

// Server serves docs/api.md.
type Server struct {
	cfg     *config.Config
	store   *store.Store
	events  events.Store
	bus     events.Bus
	queue   queue.Enqueuer
	control control.Publisher
	gh      GitHub
	setupGH GitHub
	dedupe  Deduper
	keyring *secrets.Keyring
	ready   func(ctx context.Context) error
	logger  *slog.Logger
	now     func() time.Time
	signer  *signer
	secure  bool // Secure cookie attribute
	webURL  *url.URL

	slugMu sync.Mutex
	slug   string

	// shutdown is closed by Shutdown, to close WebSockets with 1001.
	shutdown     chan struct{}
	shutdownOnce sync.Once
	wsWG         sync.WaitGroup
}

// New checks deps and returns a Server.
func New(d Deps) (*Server, error) {
	switch {
	case d.Config == nil:
		return nil, errors.New("api: nil Config")
	case d.Store == nil:
		return nil, errors.New("api: nil Store")
	case d.Bus == nil || d.Queue == nil || d.Control == nil || d.GitHub == nil || d.Dedupe == nil || d.Keyring == nil:
		return nil, errors.New("api: Bus, Queue, Control, GitHub, Dedupe and Keyring are required")
	case len(d.Config.SessionSecret) < config.MinSessionSecretLen:
		return nil, errors.New("api: SESSION_SECRET is too short")
	case d.Config.GitHubWebhookSecret == "":
		return nil, errors.New("api: GITHUB_WEBHOOK_SECRET is empty")
	}
	web, err := url.Parse(d.Config.WebOrigin)
	if err != nil || d.Config.WebOrigin == "" || d.Config.PublicURL == "" {
		return nil, errors.New("api: PUBLIC_URL and WEB_ORIGIN are required")
	}
	s := &Server{
		cfg:      d.Config,
		store:    d.Store,
		events:   d.Events,
		bus:      d.Bus,
		queue:    d.Queue,
		control:  d.Control,
		gh:       d.GitHub,
		setupGH:  d.SetupGitHub,
		dedupe:   d.Dedupe,
		keyring:  d.Keyring,
		ready:    d.Ready,
		logger:   d.Logger,
		now:      d.Now,
		signer:   newSigner([]byte(d.Config.SessionSecret)),
		webURL:   web,
		slug:     d.Config.GitHubAppSlug,
		shutdown: make(chan struct{}),
	}
	if s.events == nil {
		s.events = d.Store.Events()
	}
	if s.setupGH == nil {
		s.setupGH = s.gh
	}
	if s.logger == nil {
		s.logger = slog.Default()
	}
	if s.now == nil {
		s.now = time.Now
	}
	// Secure cookies, except for plain-http local development.
	s.secure = !strings.HasPrefix(d.Config.PublicURL, "http://")
	return s, nil
}

// Handler returns the API's routes with request IDs, logging and panic
// recovery.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Browser navigations and GitHub callbacks.
	mux.HandleFunc("GET /auth/github/login", s.handleLogin)
	mux.HandleFunc("GET /auth/github/callback", s.handleCallback)
	mux.Handle("POST /auth/logout", s.browserAPI(s.handle(s.handleLogout)))
	mux.Handle("OPTIONS /auth/logout", s.cors(http.HandlerFunc(preflight)))
	mux.HandleFunc("GET /github/setup", s.handleSetup)
	mux.HandleFunc("GET /healthz", s.handleHealth)

	// JSON API: CORS, CSRF, body checks, then the session cookie.
	api := func(pattern string, h handlerFunc) {
		mux.Handle(pattern, s.browserAPI(s.authed(s.handle(h))))
	}
	api("GET /api/me", s.handleMe)
	api("GET /api/providers", s.handleProviders)
	api("GET /api/orgs/{org_id}/github/installations", s.handleListInstallations)
	api("GET /api/orgs/{org_id}/github/install-url", s.handleInstallURL)
	api("GET /api/orgs/{org_id}/repos", s.handleListRepos)
	api("GET /api/repos/{repo_id}/profile", s.handleGetProfile)
	api("PUT /api/repos/{repo_id}/profile", s.handlePutProfile)
	api("GET /api/orgs/{org_id}/secrets", s.handleListSecrets)
	api("PUT /api/orgs/{org_id}/secrets/{name}", s.handlePutSecret)
	api("DELETE /api/orgs/{org_id}/secrets/{name}", s.handleDeleteSecret)
	api("POST /api/orgs/{org_id}/sessions", s.handleCreateSession)
	api("GET /api/orgs/{org_id}/sessions", s.handleListSessions)
	api("GET /api/sessions/{id}", s.handleGetSession)
	api("POST /api/sessions/{id}/messages", s.handlePostMessage)
	api("POST /api/sessions/{id}/answers", s.handlePostAnswer)
	api("POST /api/sessions/{id}/stop", s.handleStop)
	api("GET /api/sessions/{id}/diff", s.handleDiff)
	mux.Handle("OPTIONS /api/", s.cors(http.HandlerFunc(preflight)))
	api("/api/", func(http.ResponseWriter, *http.Request) error { return errNotFound })

	return s.withRequestID(s.withLogging(s.withRecover(mux)))
}

// Shutdown closes open WebSockets with 1001 (going away) and waits for
// them, up to ctx. Call it after http.Server.Shutdown has stopped new
// requests: hijacked connections are not tracked by http.Server.
func (s *Server) Shutdown(ctx context.Context) error {
	s.shutdownOnce.Do(func() { close(s.shutdown) })
	done := make(chan struct{})
	go func() {
		s.wsWG.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if s.ready != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()
		if err := s.ready(ctx); err != nil {
			s.log(r).Warn("health check failed", "err", err)
			writeJSON(w, http.StatusServiceUnavailable, map[string]bool{"ok": false})
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// appSlug returns the App's slug, from config or looked up once.
func (s *Server) appSlug(ctx context.Context) (string, error) {
	s.slugMu.Lock()
	defer s.slugMu.Unlock()
	if s.slug != "" {
		return s.slug, nil
	}
	slug, err := s.gh.Slug(ctx)
	if err != nil {
		return "", err
	}
	s.slug = slug
	return slug, nil
}

// webURLFor returns WEB_ORIGIN + path (and query).
func (s *Server) webURLFor(pathAndQuery string) string {
	return strings.TrimRight(s.cfg.WebOrigin, "/") + pathAndQuery
}
