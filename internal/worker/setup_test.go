package worker

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
	"github.com/Pranay0083/keith/internal/store"
)

// stopDuringInstall returns an onExec that sends Stop while the install
// marked by marker runs and waits for the watcher's interrupt, as a real
// interrupted install would.
func (e *tenv) stopDuringInstall(t *testing.T, sessionID, marker string) func(string, string, time.Duration) (sandbox.ShellResult, error) {
	return func(name, cmd string, _ time.Duration) (sandbox.ShellResult, error) {
		if !strings.Contains(cmd, marker) {
			return sandbox.ShellResult{OutputTail: "ok\n"}, nil
		}
		_ = e.st.RequestCancel(sessionID)
		select {
		case <-e.lastShells().interrupts:
		case <-time.After(5 * time.Second):
			t.Error("install not interrupted")
		}
		return sandbox.ShellResult{OutputTail: "^C", ExitCode: 130}, nil
	}
}

// A Stop while the install runs: the interrupted install "fails" and
// leaves a setup note for the agent, but the session is cancelled and
// the agent never starts (the note must not restart it).
func TestStopDuringSetupDoesNotStartTheAgent(t *testing.T) {
	e := newEnv(t)
	s := e.newSession()
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	e.onExec = e.stopDuringInstall(t, s.ID, "npm ci")
	modelCalled := false
	e.llm.PushFunc(func(llm.Request) (*llm.Response, error) {
		modelCalled = true
		return llmtest.Text("working after Stop"), nil
	})
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if modelCalled {
		t.Error("the agent ran after the Stop")
	}
	if x := e.st.Session(s.ID); x.Status != events.StatusCancelled || x.StatusReason != ReasonCancelled {
		t.Errorf("session = %s/%s", x.Status, x.StatusReason)
	}
	wantSeq(t, "statuses", e.statuses(s.ID), []string{"provisioning", "setting_up", "cancelled"})
	if c, _ := e.st.IsCancelRequested(t.Context(), s.ID); c {
		t.Error("cancel flag left set")
	}

	// A follow-up after the Stop runs the agent normally.
	e.llm.Push(llmtest.Text("Done."))
	if _, err := e.st.AddInput(s.ID, store.InputMessage, "go on", "", testUser); err != nil {
		t.Fatal(err)
	}
	e.onExec = nil
	if err := e.resume(s.ID); err != nil {
		t.Fatal(err)
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusNoChanges {
		t.Errorf("after follow-up: %s", st)
	}
}

// A Stop during the first repo's install: setup starts no further
// install (nothing would interrupt it: the watcher stops watching once
// it has seen the Stop).
func TestStopDuringSetupStartsNoFurtherInstall(t *testing.T) {
	e := newEnv(t)
	other := store.SessionRepo{RepoID: testRepo2, FullName: "acme/api", InstallationID: 1, DefaultBranch: "main"}
	s := e.newSession(withRepos(webRepo(), other))
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci # web"})
	e.profile(store.RepoProfile{RepoID: testRepo2, InstallCmd: "npm ci # api"})
	stop := e.stopDuringInstall(t, s.ID, "# web")
	secondStarted := false
	e.onExec = func(name, cmd string, d time.Duration) (sandbox.ShellResult, error) {
		if strings.Contains(cmd, "# api") {
			secondStarted = true
		}
		return stop(name, cmd, d)
	}
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	if secondStarted {
		t.Error("setup started the next install after the Stop")
	}
	if st := e.st.Session(s.ID).Status; st != events.StatusCancelled {
		t.Errorf("status = %s", st)
	}
	if n := len(e.events(s.ID, events.TypeToolCallStarted)); n != 1 {
		t.Errorf("setup tool calls = %d, want 1", n)
	}
}

// Re-provisioning a reaped sandbox checks out the agent's pushed branch
// and runs the install on it (agent-written lockfiles and scripts). That
// sandbox must never become the repo's shared snapshot, and neither may
// any re-provision: only a session's first provision, from the default
// branch, is snapshotted.
func TestReprovisionNeverReplacesTheSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name, start string
	}{
		{"agent branch", "origin/agent/"},
		{"default branch", "origin/main"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := newEnv(t)
			e.opts.IdleTimeout = time.Minute
			e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
			s := e.newSession()
			e.llm.Push(llmtest.Text("one"))
			if err := e.run(s.ID); err != nil {
				t.Fatal(err)
			}
			prof, _ := e.st.GetRepoProfile(context.Background(), testRepo)
			snap1 := prof.Snapshot
			if snap1.ID == "" {
				t.Fatal("the first provision was not snapshotted")
			}
			// The sandbox is reaped while idle; the agent may have pushed
			// agent/<id> with a changed lockfile.
			e.st.Backdate(s.ID, 2*time.Minute)
			if n, err := e.worker().Reap(context.Background()); err != nil || n != 1 {
				t.Fatalf("reap %d %v", n, err)
			}
			start := tc.start
			if strings.HasSuffix(start, "/") {
				start += s.ID
			}
			e.cloneOut = "default=main\nstart=" + start + "\n"
			e.lockOut = "100644 0badc0de 0\tpackage-lock.json\n"
			if _, err := e.st.AddInput(s.ID, store.InputMessage, "one more thing", "", testUser); err != nil {
				t.Fatal(err)
			}
			e.llm.Push(llmtest.Text("two"))
			if err := e.resume(s.ID); err != nil {
				t.Fatal(err)
			}
			sb := e.sandbox(s.ID)
			if len(e.execs(sb, "git clone")) != 1 {
				t.Fatal("not re-provisioned")
			}
			prof, _ = e.st.GetRepoProfile(context.Background(), testRepo)
			if callsHave(sb, memsandbox.OpSnapshot) || prof.Snapshot.ID != snap1.ID {
				t.Errorf("the re-provisioned sandbox replaced the repo's snapshot (%s -> %s)", snap1.ID, prof.Snapshot.ID)
			}
		})
	}
}

// A token refresh landing between the snapshot's removal of the token
// file and the snapshot itself must not write the token into the image:
// the rewrite waits until the snapshot is taken.
func TestTokenRefreshDuringSnapshotIsNotBakedIn(t *testing.T) {
	e := newEnv(t)
	e.profile(store.RepoProfile{RepoID: testRepo, InstallCmd: "npm ci"})
	s := e.newSession()
	const rotated = "ghs_rotated_during_snapshot"
	e.sbRules = append(e.sbRules, memsandbox.On(memsandbox.Contains(tmuxshell.DefaultStateDir), func(req sandbox.ExecRequest) (*sandbox.ExecResult, error) {
		sb, _ := e.sbx.Sandbox(e.st.Session(s.ID).SandboxID)
		e.gh.lastSandbox().rotate(rotated)
		// Give the token writer time to write, as a refresh racing the
		// snapshot would.
		deadline := time.Now().Add(300 * time.Millisecond)
		for time.Now().Before(deadline) && len(writesOf(sb, rotated)) == 0 {
			time.Sleep(5 * time.Millisecond)
		}
		return &sandbox.ExecResult{}, nil
	}))
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	sb := e.sandbox(s.ID)
	snapAt, writeAt := -1, -1
	for i, c := range sb.Calls() {
		switch {
		case c.Op == memsandbox.OpSnapshot:
			snapAt = i
		case c.Op == memsandbox.OpWrite && c.Path == gitTokenPath+".tmp" && string(c.Data) == rotated+"\n" && writeAt < 0:
			writeAt = i
		}
	}
	if snapAt < 0 || writeAt < 0 {
		t.Fatalf("snapshot at %d, rotated token written at %d", snapAt, writeAt)
	}
	if writeAt < snapAt {
		t.Error("the refreshed token was written before the snapshot was taken")
	}
}

// writesOf returns the calls that wrote value (plus a newline) to the
// token file's temporary path.
func writesOf(sb *memsandbox.Sandbox, value string) []memsandbox.Call {
	var out []memsandbox.Call
	for _, c := range sb.Calls() {
		if c.Op == memsandbox.OpWrite && c.Path == gitTokenPath+".tmp" && string(c.Data) == value+"\n" {
			out = append(out, c)
		}
	}
	return out
}
