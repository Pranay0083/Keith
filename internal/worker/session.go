package worker

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/store"
)

// Status reasons the worker writes (sessions.status_reason, and the code
// of the error event for failures). Agent failures use the agent's
// Reason* codes.
const (
	ReasonCancelled          = "cancelled_by_user"
	ReasonNoChanges          = "no_changes"
	ReasonTimeLimit          = "time_limit"
	ReasonStepCap            = "step_cap"
	ReasonBudgetExceeded     = "budget_exceeded"
	ReasonProvisionFailed    = "provision_failed"
	ReasonCloneFailed        = "clone_failed"
	ReasonMultiInstallation  = "repos_span_installations"
	ReasonRepoAccessLost     = "repo_access_lost"
	ReasonSessionCap         = "session_cap"
	ReasonSecretsUnreadable  = "secrets_unreadable"
	ReasonLLMUnavailable     = "llm_unavailable"
	ReasonShipFailed         = "ship_failed"
	ReasonStateUnreadable    = "agent_state_unreadable"
	ReasonInternal           = "internal_error"
	syntheticInputPrefix     = "m_keith"
	setupShell               = "setup"
	verifyShell              = "verify"
	setupToolName            = "setup"
	verifyToolName           = "run_tests"
	maxFeedbackOutput        = 6000
	stateSaveTimeout         = 30 * time.Second
	finishEmitTimeout        = 5 * time.Second
	parkTimeout              = 2 * time.Minute
	defaultCommandStepTimout = 10 * time.Minute
)

// session is one task's work on one session, under its lease.
type session struct {
	w     *Worker
	id    string
	lease Lease
	log   *slog.Logger
	start time.Time

	sess  store.Session
	em    *emitter
	env   map[string]string
	repos []*repoState

	// The environment, set up once per task by ensureEnv.
	envReady bool
	sb       sandbox.Sandbox
	shells   Shells
	// sharedShells publishes shells to the stop watcher goroutine.
	sharedShells atomic.Pointer[Shells]
	sbTokens     TokenSource
	// installationID is the session's GitHub App installation (push
	// tokens are minted on it).
	installationID int64
	host           agent.RepoHost
	hostTokens     TokenSource
	tokenStop      context.CancelFunc
	tokenDone      chan struct{}
	tokenFile      *tokenFile
	issue          *issueRef

	prov     llm.Provider
	state    *agent.State
	runner   *agent.Runner
	agentRun atomic.Pointer[agent.Run]

	// answering is set while a waiting session resumes with an answer:
	// the Runner's Delivered hook, not agentLoop, then moves it to
	// running, right after the answer's message.user.
	answering bool

	// halted is set once the session failed in this task.
	halted bool

	// stop is set once a cancel or the time limit is seen.
	stop      atomic.Bool
	timeLimit atomic.Bool

	// extra are worker-made inputs (verify feedback, setup notes) for
	// the next agent turn of this run. They never start a run on their
	// own, and a cancellation or failure drops them. Only the loop
	// goroutine touches it.
	extra []agent.Input

	synthMu sync.Mutex
	synth   map[string]string // shell -> synthetic call ID

	// pushed is the tip this task last pushed, per repo: a remote branch
	// still there may be updated even if the sandbox's lease is stale.
	pushMu sync.Mutex
	pushed map[string]string
}

// repoState is one session repo as the worker sees it.
type repoState struct {
	store.SessionRepo
	Name    string // directory name under /workspace
	Path    string
	Profile store.RepoProfile
	// fromDefault is set when this task's clone checked the session
	// branch out from origin/<default> (the agent had pushed nothing).
	fromDefault bool
}

func (r *repoState) context() agent.RepoContext {
	p := r.Profile
	return agent.RepoContext{
		FullName: r.FullName, Path: r.Path, Branch: r.Branch, DefaultBranch: r.DefaultBranch,
		Profile: agent.RepoProfile{
			InstallCmd: p.InstallCmd, BuildCmd: p.BuildCmd, TestCmd: p.TestCmd, RunCmd: p.RunCmd,
			Knowledge: p.Knowledge,
		},
	}
}

func newSession(w *Worker, id string, l Lease) *session {
	log := w.log.With("session_id", id)
	s := &session{w: w, id: id, lease: l, log: log, start: w.d.Now(), synth: map[string]string{}}
	s.em = newEmitter(id, w.d.Events, w.d.Bus, l, log)
	s.em.onQuestion = func(ctx context.Context, qid string) error {
		return w.d.Store.SetOpenQuestion(ctx, id, qid)
	}
	return s
}

// run does everything the session needs until it is terminal or
// waiting, with nothing left pending.
func (s *session) run(ctx context.Context) error {
	st := s.w.d.Store
	sess, err := st.GetSession(ctx, s.id)
	if errors.Is(err, store.ErrNotFound) {
		s.log.Warn("session not found; dropping the task")
		return nil
	}
	if err != nil {
		return err
	}
	s.sess = sess
	if err := s.loadSecrets(ctx); err != nil {
		if ctx.Err() != nil {
			return errInterrupted
		}
		return s.fail(ctx, ReasonSecretsUnreadable, "The session's secrets could not be decrypted: "+err.Error())
	}
	stopWatch := s.watchStop(ctx)
	defer stopWatch()

	// seen holds the input IDs that were pending when a pass started.
	seen := map[string]bool{}
	for range 1000 { // a bound against a bug spinning forever
		if err := ctx.Err(); err != nil {
			return errInterrupted
		}
		sess, err := st.GetSession(ctx, s.id)
		if err != nil {
			return err
		}
		s.sess.Status, s.sess.SandboxID, s.sess.SandboxProvider = sess.Status, sess.SandboxID, sess.SandboxProvider
		s.sess.OpenQuestionID = sess.OpenQuestionID

		cancelled, err := st.IsCancelRequested(ctx, s.id)
		if err != nil {
			return err
		}
		if cancelled {
			if err := s.finishCancelled(ctx); err != nil {
				return err
			}
			continue // re-check: input sent after Stop carries on
		}
		pending, err := st.ListPendingInputs(ctx, s.id)
		if err != nil {
			return err
		}
		if s.halted {
			// The session failed in this task. Input that was pending
			// when the failed pass started waits for the next wakeup
			// rather than retrying the failure in a loop. Input that
			// arrived during it has no other wakeup (the API enqueues
			// none while the session runs), so it gets one more pass,
			// as a follow-up to the failed session. After the time limit
			// it waits too: this task has no time left.
			if s.timeLimit.Load() || !slices.ContainsFunc(pending, func(in store.Input) bool { return !seen[in.ID] }) {
				return nil
			}
			s.halted = false
		}
		for _, in := range pending {
			seen[in.ID] = true
		}
		status := s.sess.Status
		live := status == events.StatusQueued || status == events.StatusProvisioning ||
			status == events.StatusSettingUp || status == events.StatusRunning
		if !live && len(pending) == 0 {
			return nil // terminal or waiting with nothing to do: a spurious wakeup
		}
		if status == events.StatusWaitingForUser && !hasAnswer(pending) {
			// Only an answer moves a waiting session (the API refuses
			// messages then); leave other input for later.
			return nil
		}
		s.stop.Store(false)
		s.answering = status == events.StatusWaitingForUser
		if err := s.work(ctx); err != nil {
			return err
		}
	}
	return errors.New("worker: session loop did not settle")
}

func hasAnswer(in []store.Input) bool {
	for _, x := range in {
		if x.Kind == store.InputAnswer {
			return true
		}
	}
	return false
}

// work provisions or resumes the environment, then runs the agent.
func (s *session) work(ctx context.Context) error {
	if err := s.emitPrompt(ctx); err != nil {
		return s.interruptedOr(ctx, err)
	}
	if !s.envReady {
		if err := s.ensureEnv(ctx); err != nil {
			var fe *failure
			if errors.As(err, &fe) {
				if ctx.Err() != nil {
					return errInterrupted
				}
				return s.fail(ctx, fe.reason, fe.msg)
			}
			return s.interruptedOr(ctx, err)
		}
	}
	if s.stop.Load() {
		// A Stop (or the time limit) during provisioning or setup: the
		// agent does not start.
		return s.stopped(ctx)
	}
	return s.agentLoop(ctx)
}

// interruptedOr returns errInterrupted when ctx is done or the lease is
// gone, else err.
func (s *session) interruptedOr(ctx context.Context, err error) error {
	if ctx.Err() != nil || !s.lease.Valid() || errors.Is(err, errLeaseLost) {
		return errInterrupted
	}
	return err
}

// failure is a reason to fail the session (not retry the task).
type failure struct {
	reason, msg string
}

func (f *failure) Error() string { return f.reason + ": " + f.msg }

func failf(reason, format string, args ...any) error {
	return &failure{reason: reason, msg: fmt.Sprintf(format, args...)}
}

// emitPrompt emits the session's first message.user, the task as the
// user gave it, if the log is still empty (docs/api.md: it is the first
// stored event).
func (s *session) emitPrompt(ctx context.Context) error {
	evs, err := s.w.d.Events.After(ctx, s.id, 0, 1)
	if err != nil || len(evs) > 0 {
		return err
	}
	text := s.sess.Prompt
	if strings.TrimSpace(text) == "" {
		if url := issueURL(s.sess.Trigger); url != "" {
			text = "Resolve " + url
		}
	}
	return s.em.Emit(ctx, events.TypeMessageUser, events.MessageUserPayload{
		MessageID: syntheticInputPrefix + "prompt", UserID: s.sess.CreatedBy, Text: text,
	})
}

// setStatus records a status and emits session.status.
func (s *session) setStatus(ctx context.Context, status events.SessionStatus, reason string) error {
	if !s.lease.Valid() {
		return errLeaseLost
	}
	if err := s.writeStatus(ctx, status, reason); err != nil {
		return err
	}
	s.sess.Status = status
	err := s.em.Emit(ctx, events.TypeSessionStatus, events.SessionStatusPayload{Status: status, Reason: reason})
	if errors.Is(err, events.ErrPublish) {
		return nil
	}
	return err
}

// writeStatus stores status. Making a parked session (finished or
// waiting_for_user) active again counts against its org's cap
// (Options.MaxActiveSessionsPerOrg), checked and written in one
// transaction under the org lock: over the cap nothing changes and a
// session_cap failure is returned. The API checks the cap before it
// queues a resume, but two resumes queued together can both pass there.
func (s *session) writeStatus(ctx context.Context, status events.SessionStatus, reason string) error {
	max := s.w.o.MaxActiveSessionsPerOrg
	if max <= 0 || !slices.Contains(store.ActiveStatuses, status) || slices.Contains(store.ActiveStatuses, s.sess.Status) {
		return s.w.d.Store.SetSessionStatus(ctx, s.id, status, reason)
	}
	ok, err := s.w.d.Store.SetSessionStatusCapped(ctx, s.sess.OrgID, s.id, status, reason, max)
	if err != nil {
		return err
	}
	if !ok {
		return failf(ReasonSessionCap,
			"This organization already has %d active sessions, its limit, so this session was not resumed. Send a message again once one of them finishes.", max)
	}
	return nil
}

// fail moves the session to failed with reason and emits an error event.
// It returns nil once recorded, so the task is not retried.
func (s *session) fail(ctx context.Context, reason, msg string) error {
	s.halted = true
	s.extra = nil
	s.log.Warn("session failed", "reason", reason, "detail", s.em.redact(msg))
	_ = s.w.d.Store.ClearOpenQuestion(ctx, s.id)
	if err := s.setStatus(ctx, events.StatusFailed, reason); err != nil {
		return s.interruptedOr(ctx, err)
	}
	err := s.em.Emit(ctx, events.TypeError, events.ErrorPayload{Code: reason, Message: msg, Retryable: true})
	if err != nil && !errors.Is(err, events.ErrPublish) {
		return s.interruptedOr(ctx, err)
	}
	return nil
}

// finishCancelled records a Stop: interrupt what runs, clear the open
// question, write cancelled (unless the session already finished) and
// clear the flag so a later follow-up message is not cancelled at once.
func (s *session) finishCancelled(ctx context.Context) error {
	s.stop.Store(true)
	s.extra = nil
	s.interruptShells(ctx)
	if !s.sess.Status.Terminal() {
		if err := s.w.d.Store.ClearOpenQuestion(ctx, s.id); err != nil {
			return s.interruptedOr(ctx, err)
		}
		reason := ReasonCancelled
		if err := s.setStatus(ctx, events.StatusCancelled, reason); err != nil {
			return s.interruptedOr(ctx, err)
		}
	}
	if err := s.w.d.Store.ClearCancelRequest(ctx, s.id); err != nil {
		return s.interruptedOr(ctx, err)
	}
	return nil
}

// stopRequested is the runner's ShouldStop: a Stop (flag or signal) or
// the time limit.
func (s *session) stopRequested(ctx context.Context) bool {
	if s.stop.Load() {
		return true
	}
	if s.overTime() {
		return true
	}
	c, err := s.w.d.Store.IsCancelRequested(ctx, s.id)
	if err == nil && c {
		s.stop.Store(true)
		return true
	}
	return false
}

func (s *session) overTime() bool {
	if s.w.d.Now().Sub(s.start) < s.w.o.MaxSessionDuration {
		return false
	}
	if !s.timeLimit.Swap(true) {
		s.log.Warn("session hit its time limit", "limit", s.w.o.MaxSessionDuration)
	}
	s.stop.Store(true)
	return true
}

// watchStop interrupts running commands as soon as a Stop arrives (by
// control signal or flag poll) or the time limit passes.
func (s *session) watchStop(ctx context.Context) func() {
	ctx, cancel := context.WithCancel(ctx)
	var sigs <-chan control.Signal
	unsub := func() {}
	if s.w.d.Control != nil {
		ch, stop, err := s.w.d.Control.Subscribe(ctx, s.id)
		if err != nil {
			s.log.Warn("subscribe to control signals; polling the cancel flag only", "err", err)
		} else {
			sigs, unsub = ch, stop
		}
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		t := time.NewTicker(s.w.o.CancelPoll)
		defer t.Stop()
		for {
			check := false
			select {
			case <-ctx.Done():
				return
			case sig, ok := <-sigs:
				if !ok {
					sigs = nil
					continue
				}
				check = sig == control.SignalCancel
			case <-t.C:
				check = true
			}
			if !check || s.stop.Load() {
				continue
			}
			if s.overTime() {
				s.interruptShells(ctx)
				continue
			}
			if c, err := s.w.d.Store.IsCancelRequested(ctx, s.id); err == nil && c {
				s.stop.Store(true)
				s.interruptShells(ctx)
			}
		}
	}()
	return func() {
		cancel()
		unsub()
		<-done
	}
}

// interruptShells sends Ctrl-C to every shell.
func (s *session) interruptShells(ctx context.Context) {
	p := s.sharedShells.Load() // the watcher goroutine calls this too
	if p == nil {
		return
	}
	sh := *p
	for _, name := range sh.List() {
		if err := sh.Interrupt(ctx, name); err != nil && ctx.Err() == nil {
			s.log.Debug("interrupt shell", "shell", name, "err", err)
		}
	}
}

// park lets go of the environment: the token file is deleted, both
// token sources are revoked and the sandbox is paused. It never touches
// the sandbox once the lease is lost (another worker may be using it).
func (s *session) park(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, parkTimeout)
	defer cancel()
	s.em.flush(ctx)
	if s.tokenStop != nil {
		s.tokenStop()
		<-s.tokenDone
	}
	if s.sb != nil && s.lease.Valid() {
		if _, err := s.sb.Exec(ctx, sandbox.ExecRequest{
			Cmd: "rm -f " + gitTokenPath + " " + gitTokenPath + ".tmp", Dir: "/", Timeout: time.Minute,
		}, nil); err != nil {
			s.log.Warn("delete the sandbox token file", "err", err)
		}
		if err := s.sb.Pause(ctx); err != nil && !errors.Is(err, sandbox.ErrNotSupported) {
			s.log.Warn("pause sandbox", "sandbox_id", s.sb.ID(), "err", err)
		}
	}
	for _, ts := range []TokenSource{s.sbTokens, s.hostTokens} {
		if ts != nil {
			if err := ts.Revoke(ctx); err != nil {
				s.log.Warn("revoke token", "err", err)
			}
		}
	}
}
