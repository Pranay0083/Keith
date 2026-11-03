package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/spf13/cobra"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/memevents"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/llm"
	llmproviders "github.com/Pranay0083/keith/internal/llm/providers"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/docker"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/worker"
	"github.com/Pranay0083/keith/internal/worker/memstore"
)

// runOptions are the flags and arguments of `keith run`.
type runOptions struct {
	Local    bool
	Repo     string
	Issue    string
	NoPR     bool
	Keep     bool
	Provider string
	Model    string
	Install  string
	Test     string
	JSON     bool
	Verbose  bool
	// Task is the positional arguments joined by spaces.
	Task string
}

// runPlan is a validated local run. It never holds the GitHub token.
type runPlan struct {
	Repo     string
	IssueURL string
	Task     string
	Provider string
	Model    string
	Install  string
	Test     string
	Keep     bool
	JSON     bool
	Options  worker.Options
}

// localLabel marks every container `keith run --local` creates.
const localLabel = "keith.local"

func newRunCmd() *cobra.Command {
	var o runOptions
	cmd := &cobra.Command{
		Use:   "run --local --repo owner/name [flags] [task...]",
		Short: "Run a coding task on a GitHub repo in a local Docker sandbox",
		Long: `run executes one task the way the cloud worker does, but in this process: it
creates a Docker sandbox (keith-sandbox:latest, or SANDBOX_TEMPLATE), clones the
repo onto the branch agent/<session-id>, runs the agent loop with the model
provider from the environment, and at the end commits, pushes and opens a pull
request (unless --no-pr). Nothing is stored: the session lives in memory and
its events stream to stdout.

Only --local is supported for now; use the web app or the API for cloud runs.

Environment:
  GITHUB_TOKEN        required: a personal access token with Contents (read and
                      write) and Pull requests (read and write) on the repo.
                      keith pushes the session branch itself, from this
                      process. The sandbox gets the token only through the
                      git credential helper's token file, to clone and fetch;
                      it is never printed. A token cannot be made read-only
                      for the sandbox, so prefer a fine-grained token limited
                      to this repo.
  ANTHROPIC_API_KEY   or GEMINI_API_KEY, for the chosen --provider.
  DEFAULT_LLM_PROVIDER, DEFAULT_MODEL, DEFAULT_GEMINI_MODEL, SANDBOX_TEMPLATE,
  MAX_SESSION_DURATION, MAX_SESSION_TOKENS, GIT_BASE_URL: as for keith worker.

When the agent asks a question, answer on stdin (a choice's number or any text;
Ctrl-D stops the session). Ctrl-C stops the session gracefully; a second Ctrl-C
aborts. The sandbox is deleted at the end unless --keep.

Build the sandbox image once with:
  docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox`,
		Example: `  keith run --local --repo acme/web "Fix the flaky date test in utils"
  keith run --local --repo acme/web --issue https://github.com/acme/web/issues/42
  keith run --local --repo acme/web --no-pr --keep --install "npm ci" --test "npm test" "Add a /health route"`,
		RunE: func(cmd *cobra.Command, args []string) error {
			o.Task = strings.TrimSpace(strings.Join(args, " "))
			plan, cfg, token, err := resolveRun(o, os.Getenv)
			if err != nil {
				return err
			}
			if _, err := exec.LookPath("docker"); err != nil {
				return errors.New("run: docker is not on PATH; --local runs the sandbox in a Docker container (https://docs.docker.com/get-docker/)")
			}
			logger := newCLILogger(cmd.ErrOrStderr(), o.Verbose)
			deps := localDeps{
				Sandboxes: docker.New(docker.Options{Logger: logger}),
				LLM:       func(name string) (llm.Provider, error) { return llmproviders.New(name, cfg) },
				GitHub:    worker.NewPATGitHub(token, github.PATHostOptions{}),
				Logger:    logger,
			}
			r, err := newLocalRun(plan, deps)
			if err != nil {
				return err
			}
			sigs := make(chan os.Signal, 4)
			signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
			defer signal.Stop(sigs)
			var info io.Writer // nil: messages share stdout with the events
			if o.JSON {
				info = cmd.ErrOrStderr()
			}
			return r.Run(cmd.Context(), cmd.InOrStdin(), cmd.OutOrStdout(), info, sigs)
		},
	}
	f := cmd.Flags()
	f.BoolVar(&o.Local, "local", false, "run in-process with a local Docker sandbox (required for now)")
	f.StringVar(&o.Repo, "repo", "", "the GitHub repo to work on, as owner/name (required)")
	f.StringVar(&o.Issue, "issue", "", "a GitHub issue URL in --repo for the agent to resolve")
	f.BoolVar(&o.NoPR, "no-pr", false, "dry run: no open_pr tool, nothing is pushed and no pull request is opened")
	f.BoolVar(&o.Keep, "keep", false, "keep the (paused) sandbox at the end instead of deleting it")
	f.StringVar(&o.Provider, "provider", "", "model provider, anthropic or gemini (default: DEFAULT_LLM_PROVIDER, else anthropic)")
	f.StringVar(&o.Model, "model", "", "model name (default: the provider's default model)")
	f.StringVar(&o.Install, "install", "", "setup command run in the repo before the agent starts (e.g. \"npm ci\")")
	f.StringVar(&o.Test, "test", "", "test command run after the agent's changes (e.g. \"npm test\")")
	f.BoolVar(&o.JSON, "json", false, "print raw events as JSON lines on stdout (messages and the summary go to stderr)")
	f.BoolVarP(&o.Verbose, "verbose", "v", false, "log the worker at INFO instead of WARN (to stderr)")
	return cmd
}

var (
	repoOwnerRE = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$`)
	repoNameRE  = regexp.MustCompile(`^[A-Za-z0-9._-]{1,100}$`)
)

// validRepo reports whether s is owner/name, as GitHub allows them.
func validRepo(s string) bool {
	owner, name, ok := strings.Cut(s, "/")
	return ok && repoOwnerRE.MatchString(owner) && repoNameRE.MatchString(name) && name != "." && name != ".."
}

// resolveRun validates the options against the environment (getenv) and
// returns the plan, the configuration and the GitHub token.
func resolveRun(o runOptions, getenv func(string) string) (runPlan, *config.Config, string, error) {
	fail := func(format string, args ...any) (runPlan, *config.Config, string, error) {
		return runPlan{}, nil, "", fmt.Errorf("run: "+format, args...)
	}
	if !o.Local {
		return fail("only --local is supported; use the web app or the API")
	}
	repo := strings.TrimSpace(o.Repo)
	if repo == "" {
		return fail("--repo is required (owner/name, for example acme/web)")
	}
	if !validRepo(repo) {
		return fail("--repo %q is not owner/name (for example acme/web)", repo)
	}
	issue := strings.TrimSpace(o.Issue)
	if issue != "" {
		issueRepo, _, err := github.ParseIssueURL(issue)
		if err != nil {
			return fail("--issue: %v", err)
		}
		if !strings.EqualFold(issueRepo, repo) {
			return fail("--issue is in %s, not in --repo %s", issueRepo, repo)
		}
	}
	task := strings.TrimSpace(o.Task)
	if task == "" && issue == "" {
		return fail("give the task as arguments, or an --issue to resolve")
	}

	token := strings.TrimSpace(getenv("GITHUB_TOKEN"))
	if token == "" {
		perms := "Contents and Pull requests (read and write)"
		if o.NoPR {
			perms = "Contents (read)"
		}
		return fail("GITHUB_TOKEN is not set: export a personal access token with %s on %s", perms, repo)
	}

	cfg, err := config.LoadFrom(getenv)
	if err != nil {
		return runPlan{}, nil, "", fmt.Errorf("run: %w", err)
	}
	provider := strings.TrimSpace(o.Provider)
	if provider == "" {
		provider = cfg.DefaultLLMProvider
	}
	if provider != config.LLMProviderAnthropic && provider != config.LLMProviderGemini {
		return fail("--provider %q: want %s or %s", provider, config.LLMProviderAnthropic, config.LLMProviderGemini)
	}
	if !cfg.LLMProviderConfigured(provider) {
		key := "ANTHROPIC_API_KEY"
		if provider == config.LLMProviderGemini {
			key = "GEMINI_API_KEY"
		}
		return fail("%s is not set (needed for the %s provider; choose another with --provider)", key, provider)
	}
	model := strings.TrimSpace(o.Model)
	if model == "" {
		model = cfg.DefaultModelFor(provider)
	}
	if model == "" {
		return fail("no default model for %s: pass --model or set DEFAULT_GEMINI_MODEL", provider)
	}

	opts := worker.OptionsFromConfig(cfg)
	opts.EgressAllowlist = nil // docker cannot enforce one
	opts.SandboxLabels = map[string]string{localLabel: "1"}
	opts.NoPR = o.NoPR
	// Nothing would reuse a snapshot of a one-off local sandbox.
	opts.NoSnapshot = true
	return runPlan{
		Repo: repo, IssueURL: issue, Task: task, Provider: provider, Model: model,
		Install: strings.TrimSpace(o.Install), Test: strings.TrimSpace(o.Test),
		Keep: o.Keep, JSON: o.JSON, Options: opts,
	}, cfg, token, nil
}

// newCLILogger logs to w at WARN, or INFO when verbose, without
// timestamps.
func newCLILogger(w io.Writer, verbose bool) *slog.Logger {
	level := slog.LevelWarn
	if verbose {
		level = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) == 0 && a.Key == slog.TimeKey {
				return slog.Attr{}
			}
			return a
		},
	}))
}

// localDeps are what a local run needs from outside the process; tests
// swap in memsandbox, llmtest and fake shells.
type localDeps struct {
	Sandboxes sandbox.Provider
	LLM       func(provider string) (llm.Provider, error)
	GitHub    worker.GitHub
	// GitPush is nil for pushing to GIT_BASE_URL over git's smart HTTP
	// protocol, from this process (never from the sandbox).
	GitPush worker.GitPusher
	// NewShells is nil for tmux shells.
	NewShells worker.ShellsFactory
	Logger    *slog.Logger
}

// localRun is one in-process session: the worker over an in-memory store,
// event log, bus, lease and control channel.
type localRun struct {
	plan runPlan
	deps localDeps
	ms   *memstore.Store
	evs  *memevents.Store
	bus  *memevents.Bus
	ctl  *control.Mem
	w    *worker.Worker
	sess store.Session

	p *eventPrinter
}

// newLocalRun builds the worker and stores the session.
func newLocalRun(plan runPlan, d localDeps) (*localRun, error) {
	r := &localRun{
		plan: plan, deps: d, ms: memstore.New(), evs: memevents.NewStore(), bus: memevents.NewBus(), ctl: control.NewMem(),
	}
	w, err := worker.New(worker.Deps{
		Store: r.ms, Events: r.evs, Bus: r.bus, Leases: worker.NewMemLeaser(), Control: r.ctl,
		Sandboxes: d.Sandboxes, GitHub: d.GitHub, GitPush: d.GitPush, LLM: d.LLM, NewShells: d.NewShells, Logger: d.Logger,
	}, plan.Options)
	if err != nil {
		return nil, err
	}
	r.w = w

	trigger, err := json.Marshal(struct {
		Kind     string `json:"kind"`
		IssueURL string `json:"issue_url,omitempty"`
	}{"cli", plan.IssueURL})
	if err != nil {
		return nil, err
	}
	repoID := uuid.NewString()
	r.sess = r.ms.AddSession(store.Session{
		ID:        uuid.NewString(),
		OrgID:     uuid.NewString(),
		CreatedBy: uuid.NewString(),
		Prompt:    plan.Task,
		Provider:  plan.Provider,
		Model:     plan.Model,
		Trigger:   trigger,
		// The worker finds the default branch from origin/HEAD after
		// cloning; the installation ID is ignored with a PAT.
		Repos: []store.SessionRepo{{RepoID: repoID, FullName: plan.Repo, InstallationID: 1}},
	})
	if plan.Install != "" || plan.Test != "" {
		r.ms.SetProfile(store.RepoProfile{RepoID: repoID, InstallCmd: plan.Install, TestCmd: plan.Test})
	}
	return r, nil
}

// Run runs the session to a terminal status, asking for answers on stdin
// while it waits, then prints the summary and deletes the sandbox. Events
// go to out; messages, prompts and the summary to info (nil: out).
// Interrupts are Ctrl-C presses. It returns an error if the session did
// not complete.
func (r *localRun) Run(ctx context.Context, stdin io.Reader, out, info io.Writer, interrupts <-chan os.Signal) error {
	r.p = newEventPrinter(out, info, r.plan.JSON)
	id := r.sess.ID

	sctx, stopStream := context.WithCancel(ctx)
	live, err := followLocal(sctx, r.evs, r.bus, id, r.p)
	if err != nil {
		stopStream()
		return err
	}
	defer func() {
		stopStream()
		<-live.done
	}()

	pr := "pull request at the end"
	if r.plan.Options.NoPR {
		pr = "no pull request"
	}
	r.p.Say("Session %s on %s (%s %s, %s)", id, r.plan.Repo, r.plan.Provider, r.plan.Model, pr)

	step := func(hctx context.Context) error {
		return r.w.HandleRunSession(hctx, queue.RunSessionPayload{SessionID: id})
	}
	var result error
	var lines <-chan string
	cancelSent := false
	for {
		res := r.handle(ctx, interrupts, step)
		live.sync(ctx)
		if res.forceQuit {
			msg := "run: quit before the worker finished"
			if sb := r.ms.Session(id).SandboxID; sb != "" {
				msg += "; the sandbox " + sb + " may still exist"
			}
			return errors.New(msg)
		}
		if res.aborted {
			result = errors.New("run: aborted; the session did not finish")
			break
		}
		if res.err != nil {
			result = fmt.Errorf("run: %w", res.err)
			break
		}
		sess := r.ms.Session(id)
		if sess.Status != events.StatusWaitingForUser || cancelSent {
			break
		}
		if lines == nil {
			lines = readLines(stdin)
		}
		text, ok := r.ask(ctx, r.question(sess.OpenQuestionID), lines, interrupts, isTerminal(stdin))
		if !ok {
			r.p.Say("Stopping the session.")
			if err := r.ms.RequestCancel(id); err != nil {
				result = err
				break
			}
			cancelSent = true
			step = func(hctx context.Context) error {
				return r.w.HandleResumeSession(hctx, queue.ResumeSessionPayload{SessionID: id, Reason: queue.ResumeCancel})
			}
			continue
		}
		in, err := r.ms.AddInput(id, store.InputAnswer, text, sess.OpenQuestionID, r.sess.CreatedBy)
		if err != nil {
			result = err
			break
		}
		step = func(hctx context.Context) error {
			return r.w.HandleResumeSession(hctx, queue.ResumeSessionPayload{SessionID: id, Reason: queue.ResumeAnswer, InputID: in.ID})
		}
	}

	sess := r.ms.Session(id)
	r.summary(sess)
	r.cleanup(ctx, sess)
	if result != nil {
		return result
	}
	switch sess.Status {
	case events.StatusCompleted, events.StatusNoChanges:
		return nil
	case events.StatusFailed:
		return fmt.Errorf("run: the session failed (%s)", sess.StatusReason)
	case events.StatusCancelled:
		return errors.New("run: the session was cancelled")
	}
	return fmt.Errorf("run: the session stopped in status %s", sess.Status)
}

type handleResult struct {
	err error
	// aborted: a second Ctrl-C cancelled the handler.
	aborted bool
	// forceQuit: a third Ctrl-C; the handler may still be running.
	forceQuit bool
}

// handle runs one worker task. The first Ctrl-C requests a graceful stop
// (the cancel flag plus a control signal, like the API's Stop), the
// second cancels the task's context, the third gives up waiting.
func (r *localRun) handle(ctx context.Context, interrupts <-chan os.Signal, fn func(context.Context) error) handleResult {
	hctx, abort := context.WithCancel(ctx)
	defer abort()
	errc := make(chan error, 1)
	go func() { errc <- fn(hctx) }()
	presses := 0
	for {
		select {
		case err := <-errc:
			return handleResult{err: err, aborted: presses >= 2 && err != nil}
		case <-interrupts:
			presses++
			switch presses {
			case 1:
				r.p.Say("^C Stopping the session (press Ctrl-C again to abort)...")
				_ = r.ms.RequestCancel(r.sess.ID)
				_ = r.ctl.Publish(ctx, r.sess.ID, control.SignalCancel)
			case 2:
				r.p.Say("^C Aborting; parking the sandbox (press Ctrl-C again to quit at once)...")
				abort()
			default:
				return handleResult{forceQuit: true}
			}
		}
	}
}

// question returns the open question.asked (or the latest one).
func (r *localRun) question(openID string) *events.QuestionAskedPayload {
	all := r.evs.All(r.sess.ID)
	for i := len(all) - 1; i >= 0; i-- {
		if all[i].Type != events.TypeQuestionAsked {
			continue
		}
		var q events.QuestionAskedPayload
		if json.Unmarshal(all[i].Payload, &q) == nil && (openID == "" || q.QuestionID == openID) {
			return &q
		}
	}
	return nil
}

// ask prompts until a usable answer is read. It returns false on EOF,
// Ctrl-C or ctx's end. echoed says whether the terminal shows what the
// user types.
func (r *localRun) ask(ctx context.Context, q *events.QuestionAskedPayload, lines <-chan string, interrupts <-chan os.Signal, echoed bool) (string, bool) {
	var choices []string
	if q != nil {
		choices = q.Choices
	}
	prompt := "answer (Ctrl-D to stop)> "
	if len(choices) > 0 {
		prompt = fmt.Sprintf("answer (1-%d or text, Ctrl-D to stop)> ", len(choices))
	}
	for {
		r.p.Prompt(prompt)
		select {
		case line, ok := <-lines:
			if !ok {
				r.p.EndPrompt("\n")
				return "", false
			}
			if echoed {
				r.p.EndPrompt("") // the terminal showed the line and its newline
			} else {
				r.p.EndPrompt(line + "\n")
			}
			if ans, ok := parseAnswer(line, choices); ok {
				return ans, true
			}
		case <-interrupts:
			r.p.EndPrompt("\n")
			return "", false
		case <-ctx.Done():
			r.p.EndPrompt("\n")
			return "", false
		}
	}
}

// parseAnswer turns a line into an answer: a number picks that choice,
// anything else is free text. Blank lines are not answers.
func parseAnswer(line string, choices []string) (string, bool) {
	s := strings.TrimSpace(line)
	if s == "" {
		return "", false
	}
	if n, err := strconv.Atoi(s); err == nil && n >= 1 && n <= len(choices) {
		return choices[n-1], true
	}
	return s, true
}

// readLines sends r's lines (without line endings) until EOF, then
// closes the channel.
func readLines(r io.Reader) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		br := bufio.NewReader(r)
		for {
			line, err := br.ReadString('\n')
			if line != "" {
				ch <- strings.TrimRight(line, "\r\n")
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}

// isTerminal reports whether r is a terminal (which echoes typed input).
func isTerminal(r io.Reader) bool {
	f, ok := r.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// summary prints the final status, pull requests and diffs.
func (r *localRun) summary(sess store.Session) {
	status := string(sess.Status)
	if sess.StatusReason != "" {
		status += " (" + sess.StatusReason + ")"
	}
	r.p.Say("")
	r.p.Say("=== Session %s ===", status)
	for _, repo := range sess.Repos {
		if repo.PRURL != "" {
			r.p.Say("Pull request for %s: %s", repo.FullName, repo.PRURL)
		}
	}
	if sess.Status == events.StatusFailed {
		if hint := failureHint(sess.StatusReason, r.lastError(), sess.Repos); hint != "" {
			r.p.Say("Hint: %s", hint)
		}
	}
	for _, d := range r.ms.Diffs(sess.ID) {
		r.p.Say("--- diff %s ---", d.FullName)
		if d.Diff == "" {
			r.p.Say("(no changes)")
			continue
		}
		r.p.Text(d.Diff)
		if d.Truncated {
			r.p.Say("(diff truncated at %d bytes)", store.MaxDiffBytes)
		}
	}
}

// lastError returns the message of the session's last error event.
func (r *localRun) lastError() string {
	all := r.evs.All(r.sess.ID)
	for i := len(all) - 1; i >= 0; i-- {
		if all[i].Type == events.TypeError {
			var e events.ErrorPayload
			if json.Unmarshal(all[i].Payload, &e) == nil {
				return e.Message
			}
		}
	}
	return ""
}

var imageMissingRE = regexp.MustCompile(`(?i)unable to find image|no such image|pull access denied|repository does not exist`)

// failureHint suggests a fix for common local failures.
func failureHint(reason, msg string, repos []store.SessionRepo) string {
	repo := "the repo"
	if len(repos) > 0 {
		repo = repos[0].FullName
	}
	switch {
	case strings.Contains(msg, "executable file not found") && strings.Contains(msg, "docker"):
		return "Docker is not installed or not on PATH."
	case strings.Contains(msg, "Cannot connect to the Docker daemon") || strings.Contains(msg, "docker daemon running"):
		return "the Docker daemon is not reachable; start Docker and try again."
	case imageMissingRE.MatchString(msg) && strings.Contains(msg, docker.DefaultImage):
		return "build the sandbox image first:\n  docker build -f sandbox/base.Dockerfile -t " + docker.DefaultImage + " sandbox"
	case imageMissingRE.MatchString(msg):
		return "the sandbox image was not found; check SANDBOX_TEMPLATE, or unset it to use " + docker.DefaultImage + "."
	case reason == worker.ReasonCloneFailed:
		return "check that " + repo + " exists and that GITHUB_TOKEN can read it (Contents: read)."
	case reason == worker.ReasonShipFailed:
		return "GITHUB_TOKEN needs Contents and Pull requests (read and write) on " + repo + "; or rerun with --no-pr."
	case reason == worker.ReasonLLMUnavailable:
		return "check the model provider's API key and --model."
	}
	return ""
}

// cleanup deletes the sandbox unless --keep. (Local runs take no repo
// snapshots: Options.NoSnapshot.)
func (r *localRun) cleanup(ctx context.Context, sess store.Session) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Minute)
	defer cancel()
	if id := sess.SandboxID; id != "" {
		if r.plan.Keep {
			if r.deps.Sandboxes.Name() == docker.Name {
				r.p.Say("Kept the sandbox %s (paused). Look inside with: docker unpause %s && docker exec -it %s bash; remove it with: docker rm -f %s", id, id, id, id)
			} else {
				r.p.Say("Kept the sandbox %s (paused).", id)
			}
		} else {
			sb, err := r.deps.Sandboxes.Get(ctx, id)
			if err == nil {
				err = sb.Delete(ctx)
			}
			if err != nil && !errors.Is(err, sandbox.ErrNotFound) {
				r.p.Say("Could not delete the sandbox %s: %v", id, err)
			}
		}
	}
}

// liveEvents prints a local session's events as they happen: Replay's
// stored-then-live stream, plus sync to catch up from the store whenever
// the caller needs everything printed (the bus is best-effort).
type liveEvents struct {
	p       *eventPrinter
	store   events.Store
	session string
	ch      <-chan events.Event
	syncc   chan chan struct{}
	done    chan struct{}
}

func followLocal(ctx context.Context, st events.Store, bus events.Bus, sessionID string, p *eventPrinter) (*liveEvents, error) {
	ch, err := events.Replay(ctx, st, bus, sessionID, 0)
	if err != nil {
		return nil, err
	}
	l := &liveEvents{p: p, store: st, session: sessionID, ch: ch, syncc: make(chan chan struct{}), done: make(chan struct{})}
	go l.run(ctx)
	return l, nil
}

func (l *liveEvents) run(ctx context.Context) {
	defer close(l.done)
	ch := l.ch
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				ch = nil
				continue
			}
			l.p.Print(ev)
		case reply := <-l.syncc:
		drain:
			for ch != nil {
				select {
				case ev, ok := <-ch:
					if !ok {
						ch = nil
						break drain
					}
					l.p.Print(ev)
				default:
					break drain
				}
			}
			if evs, err := l.store.After(ctx, l.session, l.p.Last(), 0); err == nil {
				for _, ev := range evs {
					l.p.Print(ev)
				}
			}
			close(reply)
		case <-ctx.Done():
			return
		}
	}
}

// sync returns once every event stored so far has been printed.
func (l *liveEvents) sync(ctx context.Context) {
	reply := make(chan struct{})
	select {
	case l.syncc <- reply:
	case <-l.done:
		return
	case <-ctx.Done():
		return
	}
	select {
	case <-reply:
	case <-l.done:
	case <-ctx.Done():
	}
}

// eventPrinter renders session events for a terminal: assistant text
// streams inline as it arrives, each tool call gets a start and a finish
// line with its raw terminal output in between, and statuses, questions,
// pull requests and errors get a line each. With asJSON it prints each
// event as a JSON line instead. Stored events are printed once each, in
// the order received. It is safe for concurrent use.
type eventPrinter struct {
	mu   sync.Mutex
	w    io.Writer
	info io.Writer
	// infoIsW: messages share w, so they must start on a fresh line.
	infoIsW bool
	json    bool

	last      int64
	lineStart bool
	// streaming is the message whose deltas are being printed.
	streaming string
	streamed  map[string]string
	done      map[string]bool
	status    events.SessionStatus
}

// newEventPrinter prints events to w and messages to info (nil: w).
func newEventPrinter(w, info io.Writer, asJSON bool) *eventPrinter {
	p := &eventPrinter{w: w, info: info, json: asJSON, lineStart: true, streamed: map[string]string{}, done: map[string]bool{}}
	if info == nil {
		p.info, p.infoIsW = w, true
	}
	return p
}

// Last returns the highest stored seq printed.
func (p *eventPrinter) Last() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.last
}

// Status returns the latest session.status seen.
func (p *eventPrinter) Status() events.SessionStatus {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.status
}

// Say prints a message line.
func (p *eventPrinter) Say(format string, args ...any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.toInfo(fmt.Sprintf(format, args...) + "\n")
}

// Text prints a block of text on its own lines.
func (p *eventPrinter) Text(s string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !strings.HasSuffix(s, "\n") {
		s += "\n"
	}
	p.toInfo(s)
}

// Prompt prints s without a newline.
func (p *eventPrinter) Prompt(s string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.toInfo(s)
}

// EndPrompt finishes a prompt line with s (what the terminal did not
// show of the input, ending in a newline, or "" when it showed it all).
func (p *eventPrinter) EndPrompt(s string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	io.WriteString(p.info, s)
	if p.infoIsW {
		p.lineStart = true
	}
}

func (p *eventPrinter) toInfo(s string) {
	if p.infoIsW {
		p.newline()
		p.write(s)
		return
	}
	io.WriteString(p.info, s)
}

func (p *eventPrinter) write(s string) {
	if s == "" {
		return
	}
	io.WriteString(p.w, s)
	p.lineStart = strings.HasSuffix(s, "\n")
}

func (p *eventPrinter) newline() {
	if !p.lineStart {
		p.write("\n")
	}
}

func (p *eventPrinter) line(format string, args ...any) {
	p.newline()
	p.write(fmt.Sprintf(format, args...) + "\n")
}

// Print renders one event.
func (p *eventPrinter) Print(ev events.Event) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if ev.Seq > 0 {
		if ev.Seq <= p.last {
			return
		}
		p.last = ev.Seq
	}
	v, err := events.Decode(ev)
	if st, ok := v.(*events.SessionStatusPayload); ok && err == nil {
		p.status = st.Status
	}
	if p.json {
		if b, err := json.Marshal(ev); err == nil {
			p.write(string(b) + "\n")
		}
		return
	}
	if err != nil {
		p.line("· %s %s", ev.Type, oneLine(string(ev.Payload), 200))
		return
	}
	if ev.Type != events.TypeMessageDelta && ev.Type != events.TypeMessageAssistant && ev.Type != events.TypeToolCallOutput {
		p.streaming = ""
	}
	switch x := v.(type) {
	case *events.MessageDeltaPayload:
		if x.Text == "" || p.done[x.MessageID] {
			return
		}
		if p.streaming != x.MessageID {
			p.newline()
			p.streaming = x.MessageID
		}
		p.write(x.Text)
		p.streamed[x.MessageID] += x.Text
	case *events.MessageAssistantPayload:
		prev := p.streamed[x.MessageID]
		delete(p.streamed, x.MessageID)
		p.done[x.MessageID] = true
		if prev != "" && p.streaming == x.MessageID && strings.HasPrefix(x.Text, prev) {
			p.write(x.Text[len(prev):])
		} else if x.Text != "" {
			p.newline()
			p.write(x.Text)
		}
		p.streaming = ""
		p.newline()
	case *events.MessageUserPayload:
		p.line("› %s", strings.ReplaceAll(strings.TrimRight(x.Text, "\n"), "\n", "\n  "))
	case *events.SessionStatusPayload:
		if x.Reason != "" {
			p.line("● %s (%s)", x.Status, x.Reason)
		} else {
			p.line("● %s", x.Status)
		}
	case *events.ToolCallStartedPayload:
		if in := shortInput(x.Input); in != "" {
			p.line("▶ %s  %s", x.Tool, in)
		} else {
			p.line("▶ %s", x.Tool)
		}
	case *events.ToolCallOutputPayload:
		p.write(x.Chunk)
	case *events.ToolCallFinishedPayload:
		mark := "✓"
		if x.IsError || x.ExitCode != 0 || x.TimedOut {
			mark = "✗"
		}
		var parts []string
		if x.ExitCode != 0 {
			parts = append(parts, fmt.Sprintf("exit %d", x.ExitCode))
		}
		if x.TimedOut {
			parts = append(parts, "timed out")
		}
		parts = append(parts, formatMS(x.DurationMS))
		s := fmt.Sprintf("%s %s (%s)", mark, x.Tool, strings.Join(parts, ", "))
		if x.Summary != "" {
			s += " " + oneLine(x.Summary, 200)
		}
		p.line("%s", s)
	case *events.ShellCreatedPayload:
		// The output that follows says enough.
	case *events.FileChangedPayload:
		p.line("✎ %s %s", x.Action, x.Path)
	case *events.QuestionAskedPayload:
		p.line("? %s", x.Question)
		for i, c := range x.Choices {
			p.line("  %d) %s", i+1, c)
		}
	case *events.PROpenedPayload:
		p.line("⇪ pull request %s#%d opened: %s", x.Repo, x.Number, x.URL)
	case *events.ErrorPayload:
		if x.Code != "" {
			p.line("✗ error (%s): %s", x.Code, x.Message)
		} else {
			p.line("✗ error: %s", x.Message)
		}
	default:
		p.line("· %s %s", ev.Type, oneLine(string(ev.Payload), 200))
	}
}

// shortInput summarizes a tool call's input: its command, path, pattern
// or question, else the compact JSON.
func shortInput(raw json.RawMessage) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return oneLine(string(raw), 120)
	}
	if len(m) == 0 {
		return ""
	}
	for _, k := range []string{"command", "path", "pattern", "question", "title", "text", "name", "shell"} {
		if s, ok := m[k].(string); ok && strings.TrimSpace(s) != "" {
			return oneLine(s, 120)
		}
	}
	if repo, ok := m["repo"].(string); ok {
		if n, ok := m["number"].(float64); ok {
			return fmt.Sprintf("%s#%d", repo, int(n))
		}
		return repo
	}
	b, _ := json.Marshal(m)
	return oneLine(string(b), 120)
}

// oneLine returns s's first line, cut to max runes.
func oneLine(s string, max int) string {
	s = strings.TrimSpace(s)
	line, _, more := strings.Cut(s, "\n")
	line = strings.TrimSpace(line)
	if utf8.RuneCountInString(line) > max {
		r := []rune(line)
		return string(r[:max-1]) + "…"
	}
	if more {
		return line + " …"
	}
	return line
}

// formatMS formats a duration in milliseconds for a tool call line.
func formatMS(ms int64) string {
	d := time.Duration(ms) * time.Millisecond
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return d.Round(time.Second).String()
}
