package worker

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
	"github.com/Pranay0083/keith/internal/store"
)

// lockfiles are the dependency lockfiles whose content decides whether a
// snapshot's installed dependencies are still current. They are matched
// anywhere in the repo, as tracked by git.
var lockfiles = []string{
	"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock",
	"go.sum", "Cargo.lock", "poetry.lock", "Pipfile.lock", "uv.lock", "requirements*.txt",
	"Gemfile.lock", "composer.lock", "mix.lock", "pubspec.lock", "gradle.lockfile", "packages.lock.json",
}

// lockHash fingerprints a repo's dependencies: the install command plus
// the git blob IDs of its tracked lockfiles, so it is cheap and does not
// read the files.
func (s *session) lockHash(ctx context.Context, r *repoState) (string, error) {
	args := make([]string, len(lockfiles))
	for i, f := range lockfiles {
		args[i] = shellQuote(":(glob)**/" + f)
	}
	out, err := s.execOK(ctx, r.Path, "git ls-files -s -- "+strings.Join(args, " "), gitCmdTimeout)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	fmt.Fprintf(h, "install:%s\n", r.Profile.InstallCmd)
	h.Write([]byte(out))
	return "v1:" + hex.EncodeToString(h.Sum(nil)), nil
}

// setup runs each repo's install command (status setting_up). A sandbox
// started from a snapshot skips it while the lockfile hash is unchanged.
// After a successful install of a single-repo session that has no
// current snapshot, it snapshots the sandbox for later sessions. A
// failing install does not fail the session: the agent is told and can
// fix it.
func (s *session) setup(ctx context.Context, snap store.Snapshot) error {
	if err := s.setStatus(ctx, events.StatusSettingUp, ""); err != nil {
		return err
	}
	allOK := true
	ran := false
	var hash string
	for _, r := range s.repos {
		cmd := strings.TrimSpace(r.Profile.InstallCmd)
		if cmd == "" {
			continue
		}
		if s.stopRequested(ctx) {
			// A Stop during an earlier install: start no other.
			return nil
		}
		if len(s.repos) == 1 {
			h, err := s.lockHash(ctx, r)
			if err != nil {
				s.log.Warn("lockfile hash", "repo", r.FullName, "err", err)
			}
			hash = h
			if snap.ID != "" && h != "" && h == snap.LockHash {
				s.log.Info("started from the repo snapshot; dependencies are current", "repo", r.FullName)
				continue
			}
		}
		ran = true
		res, err := s.syntheticCommand(ctx, setupToolName, setupShell, r, cmd, s.w.o.SetupTimeout)
		if err != nil {
			return s.provisionErr(ctx, ReasonProvisionFailed, "Setup failed in "+r.FullName, err)
		}
		if res.ExitCode != 0 {
			allOK = false
			s.extra = append(s.extra, agent.Input{
				ID: newSyntheticID(),
				Text: fmt.Sprintf("Setup note from the worker: the install command `%s` for %s %s. Output tail:\n\n```\n%s\n```\n\nFix the setup if the task needs it, then carry on with the task described in the system prompt.",
					cmd, r.FullName, exitPhrase(res), tail(s.em.redact(res.OutputTail), maxFeedbackOutput)),
			})
		}
	}
	if s.stop.Load() {
		return nil
	}
	if !s.w.o.NoSnapshot && len(s.repos) == 1 && ran && allOK && hash != "" && hash != snap.LockHash &&
		s.pristine(ctx, s.repos[0]) {
		s.snapshot(ctx, s.repos[0], hash)
	}
	return nil
}

// pristine reports whether r's sandbox may become the repo's shared
// snapshot: the session's first provision (the agent has never run, so
// nothing in the sandbox is its work), with the session branch checked
// out from origin/<default>. Re-provisioning a session checks out the
// agent's pushed branch, whose lockfiles and install scripts the agent
// wrote; snapshotting that would hand them to every later session.
func (s *session) pristine(ctx context.Context, r *repoState) bool {
	if !r.fromDefault {
		return false
	}
	_, err := s.w.d.Store.LoadAgentState(ctx, s.id)
	if err == nil {
		return false
	}
	if !errors.Is(err, store.ErrNotFound) {
		s.log.Warn("snapshot: check for agent state; not snapshotting", "err", err)
		return false
	}
	return true
}

func exitPhrase(res sandbox.ShellResult) string {
	if res.TimedOut {
		return "did not finish in time"
	}
	return "failed with exit code " + strconv.Itoa(res.ExitCode)
}

// snapshot saves the freshly set-up sandbox as the repo's template. The
// token file and the shell state are removed first so neither lands in
// the image (the docker provider also blanks Spec.Env, the secrets), and
// the token is written back afterwards. Token refreshes wait meanwhile.
func (s *session) snapshot(ctx context.Context, r *repoState, hash string) {
	_ = s.shells.Close(ctx, setupShell)
	if tf := s.tokenFile; tf != nil {
		tf.write.Lock()
		defer func() {
			v := tf.current()
			if err := s.writeToken(ctx, v); err != nil {
				s.log.Warn("snapshot: restore the token file", "err", err)
			} else {
				tf.setWritten(v)
			}
			tf.write.Unlock()
		}()
	}
	cleanup := "rm -f " + gitTokenPath + " " + gitTokenPath + ".tmp; rm -rf " + tmuxshell.DefaultStateDir
	if _, err := s.exec(ctx, "/", cleanup, time.Minute); err != nil {
		s.log.Warn("snapshot: clean up", "err", err)
		return
	}
	id, err := s.sb.Snapshot(ctx, "keith-"+r.Name)
	if err != nil {
		if !errors.Is(err, sandbox.ErrNotSupported) {
			s.log.Warn("snapshot the sandbox", "repo", r.FullName, "err", err)
		}
		return
	}
	snap := store.Snapshot{ID: id, Provider: s.w.d.Sandboxes.Name(), LockHash: hash}
	if err := s.w.d.Store.SetRepoSnapshot(ctx, r.RepoID, snap); err != nil {
		s.log.Warn("record the repo snapshot", "repo", r.FullName, "err", err)
		return
	}
	r.Profile.Snapshot = snap
	s.log.Info("snapshotted the set-up sandbox", "repo", r.FullName, "snapshot", id)
}

// syntheticCommand runs cmd in r's directory in a worker-owned shell,
// shown in the UI as a tool call (tool_call.started, shell.created,
// streamed tool_call.output, tool_call.finished), and waits for it to
// finish, up to timeout or a Stop.
func (s *session) syntheticCommand(ctx context.Context, tool, shell string, r *repoState, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
	callID := "keith_" + tool + "_" + randomHex(6)
	input, _ := json.Marshal(map[string]string{"repo": r.FullName, "command": cmd})
	if err := s.em.Emit(ctx, events.TypeToolCallStarted, events.ToolCallStartedPayload{
		CallID: callID, Tool: tool, Input: input,
	}); err != nil {
		return sandbox.ShellResult{}, err
	}
	s.synthMu.Lock()
	s.synth[shell] = callID
	s.synthMu.Unlock()
	defer func() {
		s.synthMu.Lock()
		delete(s.synth, shell)
		s.synthMu.Unlock()
	}()
	start := s.w.d.Now()
	res, err := s.runInShell(ctx, shell, callID, "cd "+shellQuote(r.Path)+" && ( "+cmd+"\n)", timeout)
	finished := events.ToolCallFinishedPayload{
		CallID: callID, Tool: tool, DurationMS: s.w.d.Now().Sub(start).Milliseconds(),
	}
	switch {
	case err != nil && ctx.Err() != nil:
		finished.IsError, finished.Summary = true, "interrupted"
	case err != nil:
		finished.IsError, finished.Summary = true, "internal error"
	default:
		finished.ExitCode, finished.TimedOut = res.ExitCode, res.TimedOut
		finished.IsError = res.ExitCode != 0
		finished.Summary = fmt.Sprintf("%s: %s", r.FullName, summaryFor(res))
	}
	// On shutdown the call still finishes in the timeline: emit on a
	// short context detached from the cancellation.
	ectx := ctx
	if ctx.Err() != nil {
		var cancel context.CancelFunc
		ectx, cancel = context.WithTimeout(context.WithoutCancel(ctx), finishEmitTimeout)
		defer cancel()
	}
	if eerr := s.em.Emit(ectx, events.TypeToolCallFinished, finished); eerr != nil && err == nil &&
		!errors.Is(eerr, events.ErrPublish) {
		err = eerr
	}
	return res, err
}

func summaryFor(res sandbox.ShellResult) string {
	switch {
	case res.TimedOut:
		return "timed out"
	case res.ExitCode == 0:
		return "ok"
	}
	return "exit " + strconv.Itoa(res.ExitCode)
}

var exitedRE = regexp.MustCompile(`\[command exited with code (-?\d+)\]\s*$`)

// runInShell runs cmd in shell (creating it, with shell.created) and
// waits until it finishes, the timeout passes (then it is interrupted)
// or a Stop arrives.
func (s *session) runInShell(ctx context.Context, shell, callID, cmd string, timeout time.Duration) (sandbox.ShellResult, error) {
	if err := s.shells.Create(ctx, shell); err == nil {
		if err := s.em.Emit(ctx, events.TypeShellCreated, events.ShellCreatedPayload{ShellID: shell, CallID: callID}); err != nil &&
			!errors.Is(err, events.ErrPublish) {
			return sandbox.ShellResult{}, err
		}
	} else if !errors.Is(err, sandbox.ErrShellExists) {
		return sandbox.ShellResult{}, err
	}
	start := s.w.d.Now()
	res, err := s.shells.Exec(ctx, shell, cmd, min(timeout, defaultCommandStepTimout))
	if errors.Is(err, sandbox.ErrShellBusy) {
		// A previous command is stuck there: stop it and retry once.
		_ = s.shells.Interrupt(ctx, shell)
		res, err = s.shells.Exec(ctx, shell, cmd, min(timeout, defaultCommandStepTimout))
	}
	if err != nil {
		return res, err
	}
	out := res.OutputTail
	for res.TimedOut {
		if s.stop.Load() || s.w.d.Now().Sub(start) >= timeout {
			_ = s.shells.Interrupt(ctx, shell)
			res.ExitCode = -1
			break
		}
		select {
		case <-ctx.Done():
			return res, ctx.Err()
		case <-time.After(2 * time.Second):
		}
		more, err := s.shells.Read(ctx, shell)
		if err != nil {
			return res, err
		}
		out += more
		if m := exitedRE.FindStringSubmatch(more); m != nil {
			code, _ := strconv.Atoi(m[1])
			res.ExitCode, res.TimedOut = code, false
			out = exitedRE.ReplaceAllString(out, "")
		}
	}
	res.OutputTail = tail(out, 16<<10)
	res.Duration = s.w.d.Now().Sub(start)
	return res, nil
}

// verify runs each changed repo's test command at the end of a turn. It
// returns feedback for the model when a test run failed, or nil when
// every run passed (or there was nothing to run).
func (s *session) verify(ctx context.Context) (*agent.Input, error) {
	var failures []string
	for _, r := range s.repos {
		cmd := strings.TrimSpace(r.Profile.TestCmd)
		if cmd == "" {
			continue
		}
		if s.stopRequested(ctx) {
			return nil, nil
		}
		ch, err := s.changes(ctx, r)
		if err != nil {
			return nil, err
		}
		if !ch.any() {
			continue
		}
		res, err := s.syntheticCommand(ctx, verifyToolName, verifyShell, r, cmd, s.w.o.TestTimeout)
		if err != nil {
			return nil, err
		}
		if s.stop.Load() {
			return nil, nil
		}
		if res.ExitCode != 0 || res.TimedOut {
			failures = append(failures, fmt.Sprintf("%s: `%s` %s. Output tail:\n\n```\n%s\n```",
				r.FullName, cmd, exitPhrase(res), tail(s.em.redact(res.OutputTail), maxFeedbackOutput)))
		}
	}
	if len(failures) == 0 {
		return nil, nil
	}
	return &agent.Input{
		ID: newSyntheticID(),
		Text: "Automatic check from the worker: the test command failed after your changes.\n\n" +
			strings.Join(failures, "\n\n") +
			"\n\nFix the failures you caused and run the tests again. If they fail for reasons unrelated to your change, say so in the pull request description.",
	}, nil
}

func newSyntheticID() string { return syntheticInputPrefix + randomHex(8) }

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
