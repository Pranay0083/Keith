package worker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/store"
)

// prepareRun builds the agent Runner and Run for the environment, and
// loads (or creates) the session's agent state.
func (s *session) prepareRun(ctx context.Context) error {
	if s.runner != nil {
		return nil
	}
	prov, err := s.w.d.LLM(s.sess.Provider)
	if err != nil {
		return failf(ReasonLLMUnavailable, "The %s model provider is not available: %v", s.sess.Provider, err)
	}
	s.prov = prov
	tools := agent.DefaultTools(agent.ToolOptions{})
	if s.w.o.NoPR {
		tools = slices.DeleteFunc(tools, func(t agent.Tool) bool { return t.Spec().Name == "open_pr" })
	}
	s.runner, err = agent.NewRunner(agent.Config{
		Provider:    prov,
		Model:       s.sess.Model,
		Tools:       tools,
		Effort:      llm.EffortHigh,
		MaxSteps:    s.w.o.MaxSteps,
		TokenBudget: s.w.o.MaxSessionTokens,
		Inputs:      s.inputs,
		ShouldStop:  s.stopRequested,
		Delivered:   s.delivered,
		Checkpoint:  func(ctx context.Context, st *agent.State) error { return s.saveState(ctx) },
		Now:         s.w.d.Now,
	})
	if err != nil {
		return failf(ReasonLLMUnavailable, "Could not start the agent: %v", err)
	}

	repos := make([]agent.RepoContext, len(s.repos))
	for i, r := range s.repos {
		repos[i] = r.context()
	}
	run := &agent.Run{
		SessionID:       s.id,
		Sandbox:         s.sb,
		Shells:          s.shells,
		Emit:            s.em.Emit,
		Repos:           repos,
		Host:            s.host,
		AfterFileChange: s.afterFileChange,
		OnPROpened:      s.onPROpened,
		Redact:          s.em.redact,
	}
	s.agentRun.Store(run)
	return s.loadState(ctx, repos)
}

// loadState loads the saved agent state, or builds a new one with the
// system prompt (built once per session, never rebuilt).
func (s *session) loadState(ctx context.Context, repos []agent.RepoContext) error {
	as, err := s.w.d.Store.LoadAgentState(ctx, s.id)
	if err == nil {
		st, err := agent.UnmarshalState(as.LLMMessages, as.RawMessages)
		if err != nil {
			return failf(ReasonStateUnreadable, "The saved agent state could not be read: %v", err)
		}
		if st.System != "" {
			s.state = st
			return nil
		}
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}
	in := agent.PromptInput{Task: s.sess.Prompt, Repos: repos}
	if s.issue != nil {
		is, err := s.host.ReadIssue(ctx, s.issue.Repo, s.issue.Number)
		switch {
		case err == nil:
			in.Issue = &is
		case ctx.Err() != nil:
			return errInterrupted
		default:
			s.log.Warn("read the session's issue; the agent can retry with read_issue", "issue", s.issue.URL, "err", err)
			if strings.TrimSpace(in.Task) == "" {
				in.Task = "Resolve the GitHub issue " + s.issue.URL + " (read it with read_issue)."
			}
		}
	}
	if s.w.o.NoPR {
		in.Task = strings.TrimSpace(in.Task + "\n\n(This is a local dry run: pull requests are disabled. Leave your changes in the working tree and summarize them.)")
	}
	s.state = agent.NewState(agent.BuildSystemPrompt(in))
	return s.saveState(ctx)
}

// saveState writes the agent state (agent_state). The session's provider
// is pinned, so the state is saved under it.
func (s *session) saveState(ctx context.Context) error {
	if s.state == nil {
		return nil
	}
	if !s.lease.Valid() {
		return errLeaseLost
	}
	msgs, aux, err := s.state.MarshalColumns()
	if err != nil {
		return err
	}
	if m, a := scrubNUL(msgs), scrubNUL(aux); len(m) != len(msgs) || len(a) != len(aux) || !bytes.Equal(m, msgs) || !bytes.Equal(a, aux) {
		// The agent keeps NUL out of what it adds; this is a last resort
		// so the state can still be saved.
		s.log.Warn("agent state held NUL characters; saved with U+FFFD instead")
		msgs, aux = m, a
	}
	return s.w.d.Store.SaveAgentState(ctx, store.AgentState{
		SessionID: s.id, Provider: s.sess.Provider, LLMMessages: msgs, RawMessages: aux,
	})
}

// inputs is the Runner's Inputs: worker-made inputs, then the session's
// pending inputs in order. Inputs the state already delivered (a crash
// between saving and acking) are acked here, since the Runner skips them.
func (s *session) inputs(ctx context.Context) ([]agent.Input, error) {
	out := s.extra
	s.extra = nil
	pending, err := s.w.d.Store.ListPendingInputs(ctx, s.id)
	if err != nil {
		return nil, err
	}
	var stale []string
	for _, in := range pending {
		if s.state != nil && slices.Contains(s.state.Aux.Delivered, in.ID) {
			stale = append(stale, in.ID)
			continue
		}
		out = append(out, agent.Input{ID: in.ID, UserID: in.UserID, Text: in.Text, QuestionID: in.QuestionID})
	}
	if len(stale) > 0 {
		if _, err := s.w.d.Store.MarkInputsConsumed(ctx, s.id, stale); err != nil {
			return nil, err
		}
		if len(out) == 0 && s.answering && s.state != nil && s.state.OpenQuestion() == nil {
			// The answer is already in the state (a worker stopped
			// between saving it and setting running), so Delivered
			// will not run for it: set running now.
			if err := s.toRunning(ctx); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// toRunning moves a session that was waiting for an answer back to
// running, clearing the open question first.
func (s *session) toRunning(ctx context.Context) error {
	s.answering = false
	if err := s.w.d.Store.ClearOpenQuestion(ctx, s.id); err != nil {
		return err
	}
	return s.setStatus(ctx, events.StatusRunning, "")
}

// delivered is the Runner's Delivered: the inputs are in the state, so
// save it, ack them, and after an answer move the session back to
// running (docs/api.md: message.user, then session.status running).
func (s *session) delivered(ctx context.Context, ins []agent.Input) error {
	if err := s.saveState(ctx); err != nil {
		return err
	}
	var ids []string
	for _, in := range ins {
		if !strings.HasPrefix(in.ID, syntheticInputPrefix) {
			ids = append(ids, in.ID)
		}
	}
	if _, err := s.w.d.Store.MarkInputsConsumed(ctx, s.id, ids); err != nil {
		return err
	}
	if s.sess.Status != events.StatusRunning {
		return s.toRunning(ctx)
	}
	return nil
}

// agentLoop runs the agent and handles its outcome. It writes the
// session's next status (or returns errInterrupted).
func (s *session) agentLoop(ctx context.Context) error {
	if s.sess.Status != events.StatusRunning && !s.answering {
		if err := s.setStatus(ctx, events.StatusRunning, ""); err != nil {
			return s.interruptedOr(ctx, err)
		}
	}
	cycles := 0
	for {
		run := s.agentRun.Load()
		out := s.runner.Run(ctx, run, s.state)
		s.em.flush(ctx)
		sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), stateSaveTimeout)
		serr := s.saveState(sctx)
		cancel()
		if ctx.Err() != nil || !s.lease.Valid() ||
			(out.Kind == agent.OutcomeFailed && (out.Reason == agent.ReasonInterrupted || errors.Is(out.Err, errLeaseLost))) {
			return errInterrupted
		}
		if serr != nil {
			return serr
		}
		s.log.Info("agent run ended", "outcome", out.Kind, "reason", out.Reason, "steps", out.Steps,
			"input_tokens", out.Usage.InputTokens, "output_tokens", out.Usage.OutputTokens)

		switch out.Kind {
		case agent.OutcomeEndTurn:
			if s.stop.Load() {
				return s.stopped(ctx)
			}
			if s.w.o.MaxVerifyCycles > 0 && cycles < s.w.o.MaxVerifyCycles {
				fb, err := s.verify(ctx)
				if err != nil {
					return s.endOfTurnFailed(ctx, err)
				}
				if s.stop.Load() {
					return s.stopped(ctx)
				}
				if fb != nil {
					cycles++
					s.extra = append(s.extra, *fb)
					continue
				}
			}
			return s.ship(ctx, out.Text)
		case agent.OutcomePaused:
			// The open question was recorded before question.asked.
			return s.setStatus(ctx, events.StatusWaitingForUser, "")
		case agent.OutcomeCancelled:
			return s.stopped(ctx)
		case agent.OutcomeStepCap:
			return s.fail(ctx, ReasonStepCap,
				fmt.Sprintf("The agent made %d model calls without finishing and was stopped. Send a message to let it continue.", out.Steps))
		case agent.OutcomeBudgetExceeded:
			return s.fail(ctx, ReasonBudgetExceeded,
				"The session used up its token budget. Send a message to continue, or raise MAX_SESSION_TOKENS.")
		default:
			msg := "The agent stopped: " + out.Reason
			if out.Err != nil {
				msg += ": " + out.Err.Error()
			}
			if out.Reason == agent.ReasonRefusal && out.Text != "" {
				msg += "\n\n" + out.Text
			}
			return s.fail(ctx, out.Reason, msg)
		}
	}
}

// endOfTurnFailed fails the session for an error at the end of a turn
// (verify or ship). Retrying the task would meet the same git state, and
// once its retries ran out the session would stay running with nothing
// to run it. An interruption is still retried.
func (s *session) endOfTurnFailed(ctx context.Context, err error) error {
	if ierr := s.interruptedOr(ctx, err); errors.Is(ierr, errInterrupted) {
		return ierr
	}
	var fe *failure
	if errors.As(err, &fe) {
		return s.fail(ctx, fe.reason, fe.msg)
	}
	return s.fail(ctx, ReasonShipFailed,
		"The worker could not finish the turn: "+s.em.redact(err.Error())+". Send a message to try again.")
}

// stopped records why the run stopped: the time limit, or a Stop.
func (s *session) stopped(ctx context.Context) error {
	if s.timeLimit.Load() {
		return s.fail(ctx, ReasonTimeLimit,
			fmt.Sprintf("The session ran for its time limit (%s) and was stopped. Send a message to continue.", s.w.o.MaxSessionDuration))
	}
	return s.finishCancelled(ctx)
}

// afterFileChange saves the changed repo's diff before file.changed is
// emitted (the web app fetches the diff when it sees file.changed).
func (s *session) afterFileChange(ctx context.Context, ch agent.FileChange) error {
	if ch.Repo.FullName == "" {
		return nil
	}
	for _, r := range s.repos {
		if r.FullName == ch.Repo.FullName {
			return s.saveDiff(ctx, r)
		}
	}
	return nil
}

// saveDiff stores the repo's diff against the commit the session branch
// started from (its merge base with origin/<default>), untracked files
// included, redacted and capped at store.MaxDiffBytes. A git failure is
// logged, not fatal.
func (s *session) saveDiff(ctx context.Context, r *repoState) error {
	script := fmt.Sprintf(`set -e
def=%s
base=$(git merge-base HEAD "origin/$def" 2>/dev/null || git rev-parse "origin/$def")
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cp "$(git rev-parse --git-dir)/index" "$tmp" 2>/dev/null || rm -f "$tmp"
GIT_INDEX_FILE="$tmp" git add -A
GIT_INDEX_FILE="$tmp" git diff --cached --no-color --no-ext-diff "$base" | head -c %d`,
		shellQuote(r.DefaultBranch), store.MaxDiffBytes+1)
	res, err := s.exec(ctx, r.Path, script, gitCmdTimeout)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		s.log.Warn("compute diff", "repo", r.FullName, "err", err)
		return nil
	}
	if res.ExitCode != 0 {
		s.log.Warn("compute diff", "repo", r.FullName, "exit", res.ExitCode, "stderr", s.em.redact(tail(res.Stderr, 500)))
		return nil
	}
	diff := res.Stdout
	truncated := len(diff) > store.MaxDiffBytes
	if truncated {
		diff = diff[:store.MaxDiffBytes]
	}
	return s.w.d.Store.SetSessionRepoDiff(ctx, s.id, r.RepoID, s.em.redact(diff), truncated)
}

// onPROpened records the agent's pull request before pr.opened.
func (s *session) onPROpened(ctx context.Context, rc agent.RepoContext, pr agent.PR) error {
	for _, r := range s.repos {
		if r.FullName == rc.FullName {
			r.PRNumber, r.PRURL = pr.Number, pr.URL
			return s.w.d.Store.SetSessionRepoPR(ctx, s.id, r.RepoID, pr.Number, pr.URL)
		}
	}
	return nil
}

// repoChanges is what a repo holds beyond its starting point.
type repoChanges struct {
	dirty    bool // uncommitted changes, untracked files included
	ahead    int  // commits beyond origin/<default>
	unpushed int  // commits beyond the pushed session branch (or ahead, if never pushed)
}

func (c repoChanges) any() bool { return c.dirty || c.ahead > 0 }

func (s *session) changes(ctx context.Context, r *repoState) (repoChanges, error) {
	script := fmt.Sprintf(`set -e
def=%s
branch=%s
git status --porcelain | head -c 1 | wc -c
git rev-list --count "origin/$def..HEAD"
if git rev-parse --verify -q "refs/remotes/origin/$branch" >/dev/null; then
	git rev-list --count "origin/$branch..HEAD"
else
	git rev-list --count "origin/$def..HEAD"
fi`, shellQuote(r.DefaultBranch), shellQuote(r.Branch))
	out, err := s.execOK(ctx, r.Path, script, gitCmdTimeout)
	if err != nil {
		return repoChanges{}, fmt.Errorf("git status in %s: %w", r.FullName, err)
	}
	f := strings.Fields(out)
	if len(f) != 3 {
		return repoChanges{}, fmt.Errorf("git status in %s: unexpected output %q", r.FullName, out)
	}
	n := make([]int, 3)
	for i, v := range f {
		if n[i], err = strconv.Atoi(v); err != nil {
			return repoChanges{}, fmt.Errorf("git status in %s: unexpected output %q", r.FullName, out)
		}
	}
	return repoChanges{dirty: n[0] > 0, ahead: n[1], unpushed: n[2]}, nil
}

// ship delivers the work at the end of a turn: for each repo with
// changes, commit and push what the agent left, and open a pull request
// if the agent did not. No changes anywhere means no_changes.
func (s *session) ship(ctx context.Context, finalText string) error {
	anything := false
	for _, r := range s.repos {
		ch, err := s.changes(ctx, r)
		if err != nil {
			return s.endOfTurnFailed(ctx, err)
		}
		_, hasPR := s.state.Aux.PRs[r.FullName]
		hasPR = hasPR || r.PRNumber > 0
		if !ch.any() && !hasPR {
			continue
		}
		anything = true
		if s.w.o.NoPR {
			// Nothing is pushed; save the final diff, which also covers
			// changes made with shell commands.
			if err := s.saveDiff(ctx, r); err != nil {
				return s.endOfTurnFailed(ctx, err)
			}
			continue
		}
		if err := s.shipRepo(ctx, r, ch, hasPR, finalText); err != nil {
			return s.endOfTurnFailed(ctx, err)
		}
	}
	if !anything {
		return s.setStatus(ctx, events.StatusNoChanges, ReasonNoChanges)
	}
	return s.setStatus(ctx, events.StatusCompleted, "")
}

func (s *session) shipRepo(ctx context.Context, r *repoState, ch repoChanges, hasPR bool, finalText string) error {
	if err := agent.ValidateAgentBranch(r.Branch); err != nil {
		return failf(ReasonShipFailed, "Refusing to push %s: %v", r.FullName, err)
	}
	title := s.prTitle()
	if ch.dirty || ch.unpushed > 0 || !hasPR {
		// Commit and push only on the session branch: the agent may have
		// checked out another branch, or left HEAD detached.
		out, err := s.execOK(ctx, r.Path, "git rev-parse --abbrev-ref HEAD", gitCmdTimeout)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return failf(ReasonShipFailed, "Could not read the current branch of %s: %s", r.FullName, err.Error())
		}
		if head := strings.TrimSpace(out); head != r.Branch {
			return failf(ReasonShipFailed,
				"%s is on %q, not the session branch %s, so the worker did not commit or push it. Send a message asking the agent to check out %s (keeping its work) and finish.",
				r.FullName, head, r.Branch, r.Branch)
		}
	}
	if ch.dirty {
		a := agent.DefaultGitAuthor
		cmd := fmt.Sprintf("git add -A && git -c user.name=%s -c user.email=%s commit -q -m %s",
			shellQuote(a.Name), shellQuote(a.Email), shellQuote(title))
		if _, err := s.execOK(ctx, r.Path, cmd, gitCmdTimeout); err != nil {
			return failf(ReasonShipFailed, "Could not commit the changes in %s: %s", r.FullName, s.em.redact(err.Error()))
		}
		ch.unpushed++
	}
	if ch.unpushed > 0 || !hasPR {
		if _, err := s.execOK(ctx, r.Path, "git push -q -u origin "+shellQuote(r.Branch), 5*time.Minute); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return failf(ReasonShipFailed, "Could not push %s to %s: %s", r.Branch, r.FullName, s.em.redact(err.Error()))
		}
	}
	if err := s.saveDiff(ctx, r); err != nil {
		return err
	}
	if hasPR {
		return nil
	}
	if s.host == nil {
		return failf(ReasonShipFailed, "Opening pull requests is not available.")
	}
	pr, err := s.host.OpenPR(ctx, agent.PRRequest{
		Repo: r.FullName, Head: r.Branch, Base: r.DefaultBranch, Title: title, Body: s.prBody(r, finalText),
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return failf(ReasonShipFailed, "Pushed %s, but opening the pull request on %s failed: %s", r.Branch, r.FullName, s.em.redact(err.Error()))
	}
	if err := s.onPROpened(ctx, r.context(), pr); err != nil {
		return err
	}
	if s.state.Aux.PRs == nil {
		s.state.Aux.PRs = map[string]agent.PR{}
	}
	s.state.Aux.PRs[r.FullName] = pr
	if err := s.saveState(ctx); err != nil {
		return err
	}
	err = s.em.Emit(ctx, events.TypePROpened, events.PROpenedPayload{
		Repo: r.FullName, Number: pr.Number, URL: pr.URL, Title: title, Branch: r.Branch,
	})
	if errors.Is(err, events.ErrPublish) {
		return nil
	}
	return err
}

// prTitle is the first line of the task, or the issue it refers to,
// redacted: it becomes the commit subject and the PR title on GitHub.
func (s *session) prTitle() string {
	line, _, _ := strings.Cut(strings.TrimSpace(s.em.redact(s.sess.Prompt)), "\n")
	line = strings.TrimSpace(line)
	if line == "" && s.issue != nil {
		line = fmt.Sprintf("Resolve %s#%d", s.issue.Repo, s.issue.Number)
	}
	if line == "" {
		line = "Changes from Keith session " + s.id[:min(8, len(s.id))]
	}
	if r := []rune(line); len(r) > 72 {
		line = strings.TrimSpace(string(r[:71])) + "…"
	}
	return line
}

// prBody is the agent's final message plus the issue reference,
// redacted: the model may repeat a secret it saw in tool output.
func (s *session) prBody(r *repoState, finalText string) string {
	var b strings.Builder
	if t := strings.TrimSpace(s.em.redact(finalText)); t != "" {
		b.WriteString(t)
		b.WriteString("\n\n")
	}
	if s.issue != nil && strings.EqualFold(s.issue.Repo, r.FullName) {
		fmt.Fprintf(&b, "Fixes #%d\n\n", s.issue.Number)
	}
	fmt.Fprintf(&b, "_Opened by Keith for session %s._", s.id)
	return b.String()
}
