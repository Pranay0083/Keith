package worker

import (
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
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
