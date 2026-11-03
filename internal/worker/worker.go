// Package worker runs sessions: it is the asynq handler behind
// run_session and resume_session, and the only package that imports
// everything (agent, llm, sandbox, github, store, events, secrets).
//
// One task, start to finish (see the session lifecycle in
// docs/architecture.md):
//
//  1. Lease. Take the session's Redis lease or return an error wrapping
//     queue.ErrBusy, so the task is retried in 5 s without using up a
//     retry. KeepAlive renews it; if it is lost the run's context is
//     cancelled, and the event emitter checks Lease.Valid before every
//     append, since pgevents does not enforce the fencing token.
//  2. Work out what to do from the stored state, not from the task: a
//     resume_session is a wakeup, not a command. Honour the cancel flag,
//     drain pending inputs, and resume a session a crashed worker left
//     running.
//  3. Provision (new session, or its sandbox was reaped): mint the two
//     GitHub token sources, create the sandbox from the repo's snapshot
//     or the base image, install the git credential helper as root,
//     write the sandbox token and rotate it on refresh, clone each repo
//     to /workspace/<name> and check out agent/<session-id>.
//     Resume (sandbox parked): resume it, write a fresh token, reattach
//     the tmux shells.
//  4. Set up: run each profile's install command in a shell, streamed as
//     a synthetic tool call; snapshot the first successful setup of a
//     single-repo session.
//  5. Run the agent loop, then act on its outcome: verify (tests) and
//     ship (commit, push, PR) at the end of a turn, wait for an answer,
//     record a cancellation or a failure.
//  6. Re-check pending inputs and the cancel flag before letting go
//     (the queue package's contract), then park: delete the token file,
//     revoke both tokens, pause the sandbox, release the lease.
//
// The reaper (RunReaper) deletes sandboxes parked longer than the idle
// timeout, and orphaned ones.
package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/secrets"
)

// Deps are the worker's collaborators. Every one is an interface or a
// plain value so tests can swap in fakes (memstore, memevents,
// memsandbox, llmtest).
type Deps struct {
	Store Store
	// Events and Bus are where events go: stored, then published.
	Events events.Store
	Bus    events.Bus
	Leases Leaser
	// Control delivers Stop signals; nil means the cancel flag is only
	// polled.
	Control control.Subscriber
	// Sandboxes creates and finds sandboxes.
	Sandboxes sandbox.Provider
	GitHub    GitHub
	// LLM returns the provider a session is pinned to (sessions.provider).
	LLM func(provider string) (llm.Provider, error)
	// Keyring opens org and repo secrets. nil fails sessions that have
	// secrets.
	Keyring *secrets.Keyring
	// NewShells builds a sandbox's shell manager; nil means TmuxShells.
	NewShells ShellsFactory
	// Now is the clock; nil means time.Now.
	Now    func() time.Time
	Logger *slog.Logger
}

// Options tune the worker. Zero values mean the defaults.
type Options struct {
	// GitBaseURL is where repos are cloned from (default
	// https://github.com/).
	GitBaseURL string
	// SandboxTemplate is the image or template new sandboxes start from
	// when there is no snapshot; "" means the provider's default.
	SandboxTemplate string
	// SandboxUser is the sandbox user that clones and runs shells; ""
	// means the sandbox's default non-root user (agent).
	SandboxUser string
	// SandboxLabels are added to every sandbox's labels.
	SandboxLabels map[string]string
	// EgressAllowlist is passed to the provider (sandbox.Spec).
	EgressAllowlist []string

	// MaxSessionDuration caps one task's run time (default 2h).
	MaxSessionDuration time.Duration
	// MaxSessionTokens is the session's token budget (0: none).
	MaxSessionTokens int
	// MaxSteps caps model calls per agent run (default 100).
	MaxSteps int
	// MaxVerifyCycles is how often a failed test run is fed back to the
	// model at the end of a turn (default 2; negative turns verify off).
	MaxVerifyCycles int
	// SetupTimeout bounds each install command, TestTimeout each test
	// command (defaults 30m and 20m).
	SetupTimeout time.Duration
	TestTimeout  time.Duration

	// LeaseTTL is the session lease's TTL (default 30s).
	LeaseTTL time.Duration
	// CancelPoll is how often the cancel flag and time limit are checked
	// while a command runs (default 3s).
	CancelPoll time.Duration

	// IdleTimeout is how long a parked sandbox is kept (default 30m).
	IdleTimeout time.Duration
	// ReapInterval is the reaper's period (default 1m).
	ReapInterval time.Duration
	// OrphanLabels, when set, makes the reaper delete sandboxes carrying
	// these labels that no session references (providers that can list
	// sandboxes only: docker). Leave nil when other workers or tests
	// share the provider with different databases.
	OrphanLabels map[string]string
	// OrphanGrace is how long an unreferenced sandbox is left alone, to
	// cover the moment between Create and SetSessionSandbox (default
	// 10m).
	OrphanGrace time.Duration

	// NoPR is for local dry runs: the agent gets no open_pr tool and the
	// worker opens no pull request.
	NoPR bool
	// NoSnapshot turns repo snapshots off (one-off local runs, where
	// nothing would reuse them).
	NoSnapshot bool
}

// OptionsFromConfig returns the Options the configuration sets.
func OptionsFromConfig(cfg *config.Config) Options {
	o := Options{
		GitBaseURL:         cfg.GitBaseURL,
		SandboxTemplate:    cfg.SandboxTemplate,
		MaxSessionDuration: cfg.MaxSessionDuration,
		MaxSessionTokens:   cfg.MaxSessionTokens,
		IdleTimeout:        cfg.SandboxIdleTimeout,
	}
	if cfg.SandboxProvider == config.SandboxProviderDaytona {
		o.EgressAllowlist = DefaultEgressAllowlist
	}
	return o
}

// DefaultEgressAllowlist is what sandboxes may reach on providers that
// enforce an allowlist: GitHub and the common package registries.
var DefaultEgressAllowlist = []string{
	"github.com", "api.github.com", "codeload.github.com", "objects.githubusercontent.com",
	"raw.githubusercontent.com", "registry.npmjs.org", "registry.yarnpkg.com", "pypi.org",
	"files.pythonhosted.org", "proxy.golang.org", "sum.golang.org", "crates.io", "static.crates.io",
	"index.crates.io", "rubygems.org", "repo.maven.apache.org",
}

// Defaults.
const (
	DefaultMaxSessionDuration = 2 * time.Hour
	DefaultMaxVerifyCycles    = 2
	DefaultSetupTimeout       = 30 * time.Minute
	DefaultTestTimeout        = 20 * time.Minute
	DefaultLeaseTTL           = 30 * time.Second
	DefaultCancelPoll         = 3 * time.Second
	DefaultIdleTimeout        = 30 * time.Minute
	DefaultReapInterval       = time.Minute
	DefaultOrphanGrace        = 10 * time.Minute
)

func (o Options) withDefaults() Options {
	if o.GitBaseURL == "" {
		o.GitBaseURL = config.DefaultGitBaseURL
	}
	if o.MaxSessionDuration <= 0 {
		o.MaxSessionDuration = DefaultMaxSessionDuration
	}
	if o.MaxSteps <= 0 {
		o.MaxSteps = agent.DefaultMaxSteps
	}
	if o.MaxVerifyCycles == 0 {
		o.MaxVerifyCycles = DefaultMaxVerifyCycles
	}
	if o.SetupTimeout <= 0 {
		o.SetupTimeout = DefaultSetupTimeout
	}
	if o.TestTimeout <= 0 {
		o.TestTimeout = DefaultTestTimeout
	}
	if o.LeaseTTL <= 0 {
		o.LeaseTTL = DefaultLeaseTTL
	}
	if o.CancelPoll <= 0 {
		o.CancelPoll = DefaultCancelPoll
	}
	if o.IdleTimeout <= 0 {
		o.IdleTimeout = DefaultIdleTimeout
	}
	if o.ReapInterval <= 0 {
		o.ReapInterval = DefaultReapInterval
	}
	if o.OrphanGrace <= 0 {
		o.OrphanGrace = DefaultOrphanGrace
	}
	o.SandboxLabels = maps.Clone(o.SandboxLabels)
	return o
}

// Worker runs sessions. It is safe for concurrent use: each task gets
// its own state, and a session's lease keeps two tasks off one session.
type Worker struct {
	d      Deps
	o      Options
	remote github.GitRemote
	log    *slog.Logger

	reaper reaperState
}

// New checks d and returns a Worker.
func New(d Deps, o Options) (*Worker, error) {
	switch {
	case d.Store == nil:
		return nil, errors.New("worker: Deps.Store is required")
	case d.Events == nil || d.Bus == nil:
		return nil, errors.New("worker: Deps.Events and Deps.Bus are required")
	case d.Leases == nil:
		return nil, errors.New("worker: Deps.Leases is required")
	case d.Sandboxes == nil:
		return nil, errors.New("worker: Deps.Sandboxes is required")
	case d.GitHub == nil:
		return nil, errors.New("worker: Deps.GitHub is required")
	case d.LLM == nil:
		return nil, errors.New("worker: Deps.LLM is required")
	}
	o = o.withDefaults()
	remote, err := github.ParseGitRemote(o.GitBaseURL)
	if err != nil {
		return nil, fmt.Errorf("worker: %w", err)
	}
	if d.NewShells == nil {
		d.NewShells = TmuxShells
	}
	if d.Now == nil {
		d.Now = time.Now
	}
	if d.Logger == nil {
		d.Logger = slog.Default()
	}
	return &Worker{d: d, o: o, remote: remote, log: d.Logger}, nil
}

// Handlers returns the queue handlers for the session queue.
func (w *Worker) Handlers() queue.Handlers {
	return queue.Handlers{RunSession: w.HandleRunSession, ResumeSession: w.HandleResumeSession}
}

// HandleRunSession is the run_session handler.
func (w *Worker) HandleRunSession(ctx context.Context, p queue.RunSessionPayload) error {
	return w.handle(ctx, p.SessionID)
}

// HandleResumeSession is the resume_session handler. The payload is only
// a wakeup: the session's stored inputs and flags say what to do.
func (w *Worker) HandleResumeSession(ctx context.Context, p queue.ResumeSessionPayload) error {
	return w.handle(ctx, p.SessionID)
}

// errInterrupted is returned when a run stops without a final status:
// the worker is shutting down or lost the lease. The task is retried
// (as ErrBusy, without using up a retry) and resumes from agent_state.
var errInterrupted = errors.New("worker: run interrupted")

func (w *Worker) handle(ctx context.Context, sessionID string) error {
	l, err := w.d.Leases.Acquire(ctx, lease.SessionKey(sessionID), w.o.LeaseTTL)
	if errors.Is(err, lease.ErrHeld) {
		return fmt.Errorf("worker: session %s: %w", sessionID, queue.ErrBusy)
	}
	if err != nil {
		return fmt.Errorf("worker: lease session %s: %w", sessionID, err)
	}
	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	go func() {
		for err := range l.KeepAlive(runCtx) {
			cancel(err)
		}
	}()

	s := newSession(w, sessionID, l)
	err = s.run(runCtx)
	s.park(context.WithoutCancel(ctx))

	rctx, rcancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer rcancel()
	if rerr := l.Release(rctx); rerr != nil && !errors.Is(rerr, lease.ErrLost) {
		s.log.Warn("release lease", "err", rerr)
	}

	if cause := context.Cause(runCtx); errors.Is(cause, lease.ErrLost) {
		s.log.Warn("lost the session lease; another worker will continue", "err", cause)
		return fmt.Errorf("worker: session %s: lease lost: %w", sessionID, queue.ErrBusy)
	}
	if errors.Is(err, errInterrupted) {
		return fmt.Errorf("worker: session %s: %w: %w", sessionID, err, queue.ErrBusy)
	}
	if err != nil {
		s.log.Error("session task failed", "err", err)
		return fmt.Errorf("worker: session %s: %w", sessionID, err)
	}
	return nil
}
