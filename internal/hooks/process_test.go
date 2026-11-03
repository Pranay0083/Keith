package hooks_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/hibiken/asynq"

	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/hooks"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/queue/queuetest"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

type fakeGitHub struct {
	repos []github.Repo
	err   error
	calls int
}

func (f *fakeGitHub) ListInstallationRepos(context.Context, int64) ([]github.Repo, error) {
	f.calls++
	return f.repos, f.err
}

func TestMalformedAndIgnored(t *testing.T) {
	d := hooks.Deps{} // nothing is touched
	ctx := context.Background()
	for _, ev := range []string{github.EventInstallation, github.EventInstallationRepositories} {
		if err := hooks.ProcessGitHub(ctx, d, ev, []byte(`not json`)); !errors.Is(err, hooks.ErrMalformed) {
			t.Errorf("%s: %v", ev, err)
		}
		if err := hooks.ProcessGitHub(ctx, d, ev, []byte(`{"action":"created"}`)); !errors.Is(err, hooks.ErrMalformed) {
			t.Errorf("%s without installation: %v", ev, err)
		}
	}
	if err := hooks.ProcessGitHub(ctx, d, "push", []byte(`{}`)); err != nil {
		t.Errorf("push: %v", err)
	}
	// The queue handler archives malformed payloads.
	err := hooks.Handler(d)(ctx, queue.GitHubWebhookPayload{DeliveryID: "d", Event: "installation", Body: []byte(`x`)})
	if !errors.Is(err, asynq.SkipRetry) {
		t.Errorf("handler: %v", err)
	}
}

type fixture struct {
	st   *store.Store
	d    hooks.Deps
	gh   *fakeGitHub
	q    *queuetest.Recorder
	ctl  *control.Mem
	f    storetest.Fixture
	inst int64
}

func setup(t *testing.T) *fixture {
	t.Helper()
	st := storetest.New(t)
	fx := &fixture{st: st, gh: &fakeGitHub{}, q: &queuetest.Recorder{}, ctl: control.NewMem(), f: storetest.Seed(t, st)}
	fx.inst = fx.f.Installation.InstallationID
	fx.d = hooks.Deps{Store: st, GitHub: fx.gh, Queue: fx.q, Control: fx.ctl}
	return fx
}

func (fx *fixture) process(t *testing.T, event, body string) {
	t.Helper()
	if err := hooks.ProcessGitHub(context.Background(), fx.d, event, []byte(body)); err != nil {
		t.Fatal(err)
	}
}

func installationEvent(action string, id int64) string {
	return fmt.Sprintf(`{"action":%q,"installation":{"id":%d,"account":{"login":"acme-renamed","type":"Organization"}},"sender":{"login":"octocat"}}`, action, id)
}

func TestInstallationCreatedIT(t *testing.T) {
	fx := setup(t)
	ctx := context.Background()
	// Not linked: ignored, GitHub not called.
	unknown := rand.Int64N(1<<40) + 1
	fx.process(t, github.EventInstallation, installationEvent("created", unknown))
	if _, err := fx.st.GetInstallation(ctx, unknown); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("unlinked installation created: %v", err)
	}
	if fx.gh.calls != 0 {
		t.Fatal("GitHub called for an unlinked installation")
	}

	// Linked: refreshed and repos synced (the old repo is gone).
	fx.gh.repos = []github.Repo{{ID: 777001, FullName: "acme/new", DefaultBranch: "trunk"}}
	fx.process(t, github.EventInstallation, installationEvent("created", fx.inst))
	inst, _ := fx.st.GetInstallation(ctx, fx.inst)
	if inst.AccountLogin != "acme-renamed" || inst.Status != store.InstallationActive || inst.OrgID != fx.f.Org.ID {
		t.Fatalf("installation %+v", inst)
	}
	repos, _ := fx.st.ListRepos(ctx, fx.f.Org.ID, true)
	got := map[string]bool{}
	for _, r := range repos {
		got[r.FullName] = r.Accessible
		if r.FullName == "acme/new" && r.DefaultBranch != "trunk" {
			t.Errorf("default branch %q", r.DefaultBranch)
		}
	}
	if !got["acme/new"] || got["acme/api"] {
		t.Fatalf("repos %v", got)
	}

	// A GitHub failure is retried.
	fx.gh.err = errors.New("github down")
	if err := hooks.ProcessGitHub(ctx, fx.d, github.EventInstallation, []byte(installationEvent("created", fx.inst))); err == nil {
		t.Fatal("GitHub failure swallowed")
	}
}

func TestInstallationDeletedIT(t *testing.T) {
	fx := setup(t)
	ctx := context.Background()
	newSession := func(status events.SessionStatus) store.Session {
		x, err := fx.st.CreateSession(ctx, store.CreateSessionParams{
			OrgID: fx.f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{fx.f.Repo.ID},
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := fx.st.SetSessionStatus(ctx, x.ID, status, ""); err != nil {
			t.Fatal(err)
		}
		return x
	}
	queued := fx.f.Session
	running := newSession(events.StatusRunning)
	waiting := newSession(events.StatusWaitingForUser)
	done := newSession(events.StatusCompleted)

	fx.process(t, github.EventInstallation, installationEvent("deleted", fx.inst))

	inst, _ := fx.st.GetInstallation(ctx, fx.inst)
	if inst.Status != store.InstallationDeleted {
		t.Fatalf("status %s", inst.Status)
	}
	if r, _ := fx.st.GetRepo(ctx, fx.f.Repo.ID); r.Accessible {
		t.Fatal("repo still accessible")
	}
	for _, x := range []store.Session{queued, running, waiting} {
		if ok, _ := fx.st.IsCancelRequested(ctx, x.ID); !ok {
			t.Errorf("session %s (%s) not cancel-requested", x.ID, x.Status)
		}
	}
	if ok, _ := fx.st.IsCancelRequested(ctx, done.ID); ok {
		t.Error("finished session cancel-requested")
	}
	signalled := map[string]bool{}
	for _, p := range fx.ctl.Published() {
		if p.Signal == control.SignalCancel {
			signalled[p.SessionID] = true
		}
	}
	if !signalled[queued.ID] || !signalled[running.ID] || !signalled[waiting.ID] || signalled[done.ID] {
		t.Fatalf("signals %v", signalled)
	}
	// Only sessions no worker holds get a resume.
	resumed := map[string]bool{}
	for _, r := range fx.q.Resumes() {
		if r.Reason != queue.ResumeCancel {
			t.Errorf("resume %+v", r)
		}
		resumed[r.SessionID] = true
	}
	if !resumed[queued.ID] || !resumed[waiting.ID] || resumed[running.ID] || len(resumed) != 2 {
		t.Fatalf("resumes %v", resumed)
	}

	// Idempotent; an unknown installation is ignored.
	fx.process(t, github.EventInstallation, installationEvent("deleted", fx.inst))
	fx.process(t, github.EventInstallation, installationEvent("deleted", 1))
}

func TestInstallationSuspendIT(t *testing.T) {
	fx := setup(t)
	ctx := context.Background()
	fx.process(t, github.EventInstallation, installationEvent("suspend", fx.inst))
	if inst, _ := fx.st.GetInstallation(ctx, fx.inst); inst.Status != store.InstallationSuspended {
		t.Fatalf("suspend: %s", inst.Status)
	}
	fx.process(t, github.EventInstallation, installationEvent("unsuspend", fx.inst))
	if inst, _ := fx.st.GetInstallation(ctx, fx.inst); inst.Status != store.InstallationActive {
		t.Fatalf("unsuspend: %s", inst.Status)
	}
	fx.process(t, github.EventInstallation, installationEvent("suspend", 1))
	fx.process(t, github.EventInstallation, installationEvent("new_permissions_accepted", fx.inst))
}

func TestInstallationRepositoriesIT(t *testing.T) {
	fx := setup(t)
	ctx := context.Background()
	ghID := fx.f.Repo.GitHubRepoID

	// added: a full sync from GitHub.
	fx.gh.repos = []github.Repo{{ID: ghID, FullName: "acme/api", DefaultBranch: "main"}, {ID: 888001, FullName: "acme/added", DefaultBranch: "dev", Private: true}}
	fx.process(t, github.EventInstallationRepositories,
		fmt.Sprintf(`{"action":"added","installation":{"id":%d},"repositories_added":[{"id":888001,"full_name":"acme/added","private":true}]}`, fx.inst))
	added, err := fx.st.GetRepoByFullName(ctx, fx.f.Org.ID, "acme/added")
	if err != nil || !added.Accessible || added.DefaultBranch != "dev" {
		t.Fatalf("added repo %+v %v", added, err)
	}

	// removed: marked inaccessible, history kept.
	fx.process(t, github.EventInstallationRepositories,
		fmt.Sprintf(`{"action":"removed","installation":{"id":%d},"repositories_removed":[{"id":%d,"full_name":"acme/api"}]}`, fx.inst, ghID))
	if r, _ := fx.st.GetRepo(ctx, fx.f.Repo.ID); r.Accessible {
		t.Fatal("removed repo still accessible")
	}
	if r, _ := fx.st.GetRepo(ctx, added.ID); !r.Accessible {
		t.Fatal("other repo marked inaccessible")
	}
	// An unlinked installation is ignored without calling GitHub.
	calls := fx.gh.calls
	fx.process(t, github.EventInstallationRepositories, `{"action":"added","installation":{"id":5}}`)
	if fx.gh.calls != calls {
		t.Fatal("GitHub called for an unlinked installation")
	}
}

func TestStopSessionIT(t *testing.T) {
	fx := setup(t)
	ctx := context.Background()
	if err := hooks.StopSession(ctx, fx.d, fx.f.Session.ID); err != nil {
		t.Fatal(err)
	}
	if ok, _ := fx.st.IsCancelRequested(ctx, fx.f.Session.ID); !ok {
		t.Fatal("not flagged")
	}
	if err := fx.st.SetSessionStatus(ctx, fx.f.Session.ID, events.StatusCancelled, "stopped"); err != nil {
		t.Fatal(err)
	}
	if err := hooks.StopSession(ctx, fx.d, fx.f.Session.ID); !errors.Is(err, hooks.ErrSessionFinished) {
		t.Fatalf("finished: %v", err)
	}
	if err := hooks.StopSession(ctx, fx.d, "00000000-0000-4000-8000-000000000000"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("unknown: %v", err)
	}
}
