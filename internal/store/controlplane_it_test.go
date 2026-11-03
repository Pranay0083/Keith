package store_test

import (
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

func TestSessionRepoOrder(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	// Names sort a, b, c; the session keeps the order it was given.
	ra := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: f.Installation.InstallationID, GitHubRepoID: 9001, FullName: "acme/a"}))
	rc := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: f.Installation.InstallationID, GitHubRepoID: 9003, FullName: "acme/c"}))
	x := must(s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m",
		RepoIDs: []string{rc.ID, f.Repo.ID, rc.ID, ra.ID},
	}))
	want := []string{"acme/c", "acme/api", "acme/a"}
	check := func(what string, repos []store.SessionRepo) {
		t.Helper()
		var got []string
		for _, r := range repos {
			got = append(got, r.FullName)
		}
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("%s order = %v, want %v", what, got, want)
		}
	}
	check("create", x.Repos)
	check("get", must(s.GetSession(ctx, x.ID)).Repos)
	page := must(s.ListSessions(ctx, store.ListSessionsParams{OrgID: f.Org.ID}))
	check("list", page.Sessions[0].Repos)

	diffs := must(s.ListSessionDiffs(ctx, x.ID))
	if len(diffs) != 3 || diffs[0].FullName != "acme/c" || diffs[2].FullName != "acme/a" {
		t.Fatalf("diff order %+v", diffs)
	}
}

func TestSessionInputs(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	sid := f.Session.ID

	m1 := must(s.CreateInput(ctx, store.CreateInputParams{SessionID: sid, Kind: store.InputMessage, UserID: f.User.ID, Text: "first"}))
	if !store.ValidInputID(m1.ID) || m1.Kind != store.InputMessage || m1.UserID != f.User.ID || m1.ConsumedAt != nil {
		t.Fatalf("message %+v", m1)
	}
	a1 := must(s.CreateInput(ctx, store.CreateInputParams{
		SessionID: sid, Kind: store.InputAnswer, QuestionID: "toolu_1", Text: "yes",
	}))
	if a1.QuestionID != "toolu_1" || a1.UserID != "" {
		t.Fatalf("answer %+v", a1)
	}

	// Invalid inputs.
	for _, p := range []store.CreateInputParams{
		{SessionID: sid, Kind: "note", Text: "x"},
		{SessionID: sid, Kind: store.InputAnswer, Text: "x"},
		{SessionID: sid, Kind: store.InputMessage, QuestionID: "q", Text: "x"},
		{SessionID: sid, Kind: store.InputMessage, ID: "bad id", Text: "x"},
	} {
		if _, err := s.CreateInput(ctx, p); err == nil {
			t.Errorf("CreateInput(%+v) accepted", p)
		}
	}
	_, err := s.CreateInput(ctx, store.CreateInputParams{SessionID: missingID, Kind: store.InputMessage, Text: "x"})
	isErr(t, err, store.ErrNotFound)
	_, err = s.CreateInput(ctx, store.CreateInputParams{ID: m1.ID, SessionID: sid, Kind: store.InputMessage, Text: "dup"})
	isErr(t, err, store.ErrConflict)

	pending := must(s.ListPendingInputs(ctx, sid))
	if len(pending) != 2 || pending[0].ID != m1.ID || pending[1].ID != a1.ID {
		t.Fatalf("pending %+v", pending)
	}
	if got := must(s.GetInput(ctx, sid, a1.ID)); got.Text != "yes" {
		t.Fatalf("get input %+v", got)
	}
	_, err = s.GetInput(ctx, missingID, a1.ID)
	isErr(t, err, store.ErrNotFound)

	if n := must(s.MarkInputsConsumed(ctx, sid, []string{m1.ID, "m_unknown"})); n != 1 {
		t.Fatalf("consumed %d, want 1", n)
	}
	if n := must(s.MarkInputsConsumed(ctx, sid, []string{m1.ID})); n != 0 {
		t.Fatalf("re-consume changed %d rows", n)
	}
	pending = must(s.ListPendingInputs(ctx, sid))
	if len(pending) != 1 || pending[0].ID != a1.ID {
		t.Fatalf("pending after consume %+v", pending)
	}
	if got := must(s.GetInput(ctx, sid, m1.ID)); got.ConsumedAt == nil {
		t.Fatal("consumed input has no consumed_at")
	}
	// Another session's inputs are untouched.
	other := must(s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{f.Repo.ID},
	}))
	if n := must(s.MarkInputsConsumed(ctx, other.ID, []string{a1.ID})); n != 0 {
		t.Fatal("consumed another session's input")
	}
}

func TestOpenQuestionAndLocks(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	sid := f.Session.ID

	if got := must(s.GetSession(ctx, sid)); got.OpenQuestionID != "" {
		t.Fatalf("new session open question %q", got.OpenQuestionID)
	}
	ok(t, s.SetOpenQuestion(ctx, sid, "toolu_9"))
	if got := must(s.GetSession(ctx, sid)); got.OpenQuestionID != "toolu_9" {
		t.Fatalf("open question %q", got.OpenQuestionID)
	}
	ok(t, s.ClearOpenQuestion(ctx, sid))
	if got := must(s.GetSession(ctx, sid)); got.OpenQuestionID != "" {
		t.Fatalf("cleared open question %q", got.OpenQuestionID)
	}
	isErr(t, s.SetOpenQuestion(ctx, missingID, "q"), store.ErrNotFound)
	isErr(t, s.ClearOpenQuestion(ctx, missingID), store.ErrNotFound)
	if err := s.SetOpenQuestion(ctx, sid, ""); err == nil {
		t.Error("empty question ID accepted")
	}

	// GetSessionForUpdate and LockOrg work inside a transaction.
	ok(t, s.WithTx(ctx, func(tx *store.Store) error {
		x, err := tx.GetSessionForUpdate(ctx, sid)
		if err != nil || x.ID != sid {
			t.Errorf("for update: %+v %v", x, err)
		}
		return tx.LockOrg(ctx, f.Org.ID)
	}))
	_, err := s.GetSessionForUpdate(ctx, missingID)
	isErr(t, err, store.ErrNotFound)
	isErr(t, s.WithTx(ctx, func(tx *store.Store) error { return tx.LockOrg(ctx, missingID) }), store.ErrNotFound)
}

func TestActiveAndLiveSessions(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	newSession := func(repoIDs ...string) store.Session {
		return must(s.CreateSession(ctx, store.CreateSessionParams{
			OrgID: f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: repoIDs,
		}))
	}
	// f.Session is queued. Add running, waiting and completed ones.
	running, waiting, done := newSession(f.Repo.ID), newSession(f.Repo.ID), newSession(f.Repo.ID)
	ok(t, s.SetSessionStatus(ctx, running.ID, events.StatusRunning, ""))
	ok(t, s.SetSessionStatus(ctx, waiting.ID, events.StatusWaitingForUser, ""))
	ok(t, s.SetSessionStatus(ctx, done.ID, events.StatusCompleted, ""))

	if n := must(s.CountActiveSessions(ctx, f.Org.ID)); n != 2 {
		t.Fatalf("active = %d, want 2 (queued + running)", n)
	}
	if n := must(s.CountActiveSessions(ctx, missingID)); n != 0 {
		t.Fatalf("active in unknown org = %d", n)
	}

	// A session on another installation is not listed.
	inst2 := must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: f.Org.ID, InstallationID: f.Installation.InstallationID + 1, AccountLogin: "b"}))
	r2 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst2.InstallationID, GitHubRepoID: 4242, FullName: "b/r"}))
	elsewhere := newSession(r2.ID)

	live := must(s.ListLiveSessionsByInstallation(ctx, f.Installation.InstallationID))
	var ids []string
	for _, x := range live {
		ids = append(ids, x.ID)
	}
	want := []string{f.Session.ID, running.ID, waiting.ID}
	if strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Fatalf("live by installation = %v, want %v", ids, want)
	}
	byRepo := must(s.ListLiveSessionsByRepos(ctx, []string{r2.ID}))
	if len(byRepo) != 1 || byRepo[0].ID != elsewhere.ID {
		t.Fatalf("live by repo = %+v", byRepo)
	}
	if got := must(s.ListLiveSessionsByRepos(ctx, nil)); got != nil {
		t.Fatalf("no repos = %+v", got)
	}
}

func TestSessionDiffs(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	sid := f.Session.ID

	d := must(s.ListSessionDiffs(ctx, sid))
	if len(d) != 1 || d[0].Diff != "" || d[0].UpdatedAt != nil || d[0].Truncated || d[0].FullName != "acme/api" {
		t.Fatalf("empty diff %+v", d)
	}
	ok(t, s.SetSessionRepoDiff(ctx, sid, f.Repo.ID, "diff --git a/x b/x\n+\x00", false))
	d = must(s.ListSessionDiffs(ctx, sid))
	if d[0].Diff != "diff --git a/x b/x\n+�" || d[0].UpdatedAt == nil || d[0].Truncated {
		t.Fatalf("saved diff %+v", d[0])
	}
	big := strings.Repeat("x", store.MaxDiffBytes+10)
	ok(t, s.SetSessionRepoDiff(ctx, sid, f.Repo.ID, big, false))
	d = must(s.ListSessionDiffs(ctx, sid))
	if len(d[0].Diff) != store.MaxDiffBytes || !d[0].Truncated {
		t.Fatalf("big diff len %d truncated %v", len(d[0].Diff), d[0].Truncated)
	}
	ok(t, s.SetSessionRepoDiff(ctx, sid, f.Repo.ID, "small", true))
	if d = must(s.ListSessionDiffs(ctx, sid)); !d[0].Truncated {
		t.Fatal("caller's truncated flag lost")
	}
	isErr(t, s.SetSessionRepoDiff(ctx, sid, missingID, "x", false), store.ErrNotFound)
	isErr(t, s.SetSessionRepoDiff(ctx, missingID, f.Repo.ID, "x", false), store.ErrNotFound)
}

func TestRepoAndUserLookups(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	inst := f.Installation.InstallationID

	got := must(s.GetRepoByFullName(ctx, f.Org.ID, "ACME/Api"))
	if got.ID != f.Repo.ID {
		t.Fatalf("by name %+v", got)
	}
	_, err := s.GetRepoByFullName(ctx, f.Org.ID, "acme/none")
	isErr(t, err, store.ErrNotFound)
	org2 := must(s.CreateOrg(ctx, "Two", f.User.ID))
	_, err = s.GetRepoByFullName(ctx, org2.ID, "acme/api")
	isErr(t, err, store.ErrNotFound)

	r2 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst, GitHubRepoID: 5001, FullName: "acme/two"}))
	r3 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst, GitHubRepoID: 5002, FullName: "acme/three"}))
	if n := must(s.SetInstallationReposInaccessibleExcept(ctx, inst, []int64{f.Repo.GitHubRepoID, 5001})); n != 1 {
		t.Fatalf("marked %d, want 1", n)
	}
	if must(s.GetRepo(ctx, r3.ID)).Accessible || !must(s.GetRepo(ctx, r2.ID)).Accessible {
		t.Fatal("wrong repos marked")
	}
	if n := must(s.SetInstallationReposInaccessibleExcept(ctx, inst, nil)); n != 2 {
		t.Fatalf("nil keep marked %d, want 2", n)
	}

	users := must(s.GetUsers(ctx, []string{f.User.ID, missingID, "bad"}))
	if len(users) != 1 || users[f.User.ID].GitHubLogin != "octocat" {
		t.Fatalf("users %+v", users)
	}
	if got := must(s.GetUsers(ctx, nil)); len(got) != 0 {
		t.Fatalf("no ids -> %+v", got)
	}

	// Profiles keyed by repo.
	must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{RepoID: f.Repo.ID, TestCmd: "go test"}))
	profiles := must(s.ListRepoProfiles(ctx, f.Org.ID))
	if len(profiles) != 1 || profiles[f.Repo.ID].TestCmd != "go test" {
		t.Fatalf("profiles %+v", profiles)
	}
	if len(must(s.ListRepoProfiles(ctx, org2.ID))) != 0 {
		t.Fatal("other org sees profiles")
	}
}

func TestListOrgSecrets(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	put := func(repoID, name string) {
		t.Helper()
		must(s.PutSecret(ctx, store.PutSecretParams{OrgID: f.Org.ID, RepoID: repoID, Name: name, Ciphertext: []byte("ct")}))
	}
	put(f.Repo.ID, "DATABASE_URL")
	put("", "DATABASE_URL")
	put("", "API_KEY")
	got := must(s.ListOrgSecrets(ctx, f.Org.ID))
	type row struct{ name, repo, full string }
	var rows []row
	for _, x := range got {
		rows = append(rows, row{x.Name, x.RepoID, x.RepoFullName})
	}
	want := []row{{"API_KEY", "", ""}, {"DATABASE_URL", "", ""}, {"DATABASE_URL", f.Repo.ID, "acme/api"}}
	if len(rows) != len(want) {
		t.Fatalf("secrets %+v", rows)
	}
	for i := range want {
		if rows[i] != want[i] {
			t.Errorf("secret %d = %+v, want %+v", i, rows[i], want[i])
		}
	}
	if got := must(s.ListOrgSecrets(ctx, missingID)); len(got) != 0 {
		t.Fatalf("unknown org secrets %+v", got)
	}
}

func TestEnsurePersonalOrg(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	u := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 424242, GitHubLogin: "hubot"}))
	if !must(s.EnsurePersonalOrg(ctx, u.ID, "hubot")) {
		t.Fatal("no org created on first sign-in")
	}
	if must(s.EnsurePersonalOrg(ctx, u.ID, "hubot")) {
		t.Fatal("second org created")
	}
	orgs := must(s.ListOrgsForUser(ctx, u.ID))
	if len(orgs) != 1 || orgs[0].Org.Name != "hubot" || orgs[0].Role != store.RoleOwner {
		t.Fatalf("orgs %+v", orgs)
	}
	_, err := s.EnsurePersonalOrg(ctx, missingID, "x")
	isErr(t, err, store.ErrNotFound)

	// Concurrent first sign-ins create one org.
	u2 := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 434343, GitHubLogin: "race"}))
	errs := make(chan error, 8)
	for range 8 {
		go func() {
			_, err := s.EnsurePersonalOrg(ctx, u2.ID, "race")
			errs <- err
		}()
	}
	for range 8 {
		ok(t, <-errs)
	}
	if n := len(must(s.ListOrgsForUser(ctx, u2.ID))); n != 1 {
		t.Fatalf("concurrent sign-ins made %d orgs", n)
	}
}
