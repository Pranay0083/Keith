package store_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
	"github.com/Pranay0083/keith/migrations"
)

// Integration tests: KEITH_IT=1, against DATABASE_URL (see storetest).

const missingID = "00000000-0000-4000-8000-000000000000"

func ctxT(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)
	return ctx
}

// must returns v, panicking on err (Go cannot pass a multi-value call
// alongside t). The panic fails the test with a stack trace to the line.
func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func ok(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func isErr(t *testing.T, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
}

func TestMigrateFromEmpty(t *testing.T) {
	storetest.Skip(t)
	ctx := ctxT(t)
	s, err := store.Open(ctx, storetest.ScratchDatabase(t))
	ok(t, err)
	defer s.Close()

	ok(t, s.Migrate(ctx))
	ok(t, s.Migrate(ctx)) // idempotent

	var tables []string
	rows, err := s.Pool().Query(ctx, `SELECT table_name FROM information_schema.tables
		WHERE table_schema = 'public' AND table_name <> 'goose_db_version' ORDER BY table_name`)
	ok(t, err)
	for rows.Next() {
		var n string
		ok(t, rows.Scan(&n))
		tables = append(tables, n)
	}
	ok(t, rows.Err())
	want := []string{"agent_state", "events", "github_installations", "org_members", "orgs",
		"repo_profiles", "repos", "secrets", "session_inputs", "session_repos", "sessions", "users"}
	if !slices.Equal(tables, want) {
		t.Errorf("tables = %v, want %v", tables, want)
	}

	var version int64
	ok(t, s.Pool().QueryRow(ctx, `SELECT max(version_id) FROM goose_db_version WHERE is_applied`).Scan(&version))
	if want := latestMigration(t); version != want {
		t.Errorf("goose version = %d, want %d", version, want)
	}
	var hasReason bool
	ok(t, s.Pool().QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM information_schema.columns
		WHERE table_name = 'sessions' AND column_name = 'status_reason')`).Scan(&hasReason))
	if !hasReason {
		t.Error("sessions.status_reason missing after migrate")
	}
}

// latestMigration is the highest version among the embedded migration
// files (NNNNN_name.sql).
func latestMigration(t *testing.T) int64 {
	t.Helper()
	names, err := fs.Glob(migrations.FS, "*.sql")
	ok(t, err)
	var latest int64
	for _, n := range names {
		prefix, _, _ := strings.Cut(n, "_")
		v, err := strconv.ParseInt(prefix, 10, 64)
		ok(t, err)
		latest = max(latest, v)
	}
	return latest
}

func TestOpenRejectsBadURL(t *testing.T) {
	_, err := store.Open(context.Background(), "postgres://user:hunter2@[bad")
	if err == nil {
		t.Fatal("want error")
	}
	if bytes.Contains([]byte(err.Error()), []byte("hunter2")) {
		t.Errorf("error leaks the password: %v", err)
	}
}

func TestUsers(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)

	u := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 42, GitHubLogin: "old", Name: "A"}))
	if u.ID == "" || u.Email != "" || u.GitHubLogin != "old" {
		t.Fatalf("created %+v", u)
	}
	u2 := must(s.UpsertUser(ctx, store.UpsertUserParams{
		GitHubUserID: 42, GitHubLogin: "new", Email: "a@example.com", Name: "B", AvatarURL: "https://x/a.png",
	}))
	if u2.ID != u.ID || u2.GitHubLogin != "new" || u2.Email != "a@example.com" || u2.AvatarURL != "https://x/a.png" {
		t.Fatalf("upserted %+v, want same ID %s with new profile", u2, u.ID)
	}
	if !u2.UpdatedAt.After(u.UpdatedAt) && !u2.UpdatedAt.Equal(u.UpdatedAt) {
		t.Errorf("updated_at went backwards")
	}

	got := must(s.GetUser(ctx, u.ID))
	if got.GitHubLogin != "new" {
		t.Errorf("GetUser = %+v", got)
	}
	got = must(s.GetUserByGitHubID(ctx, 42))
	if got.ID != u.ID {
		t.Errorf("GetUserByGitHubID = %+v", got)
	}
	_, err := s.GetUser(ctx, missingID)
	isErr(t, err, store.ErrNotFound)
	_, err = s.GetUser(ctx, "not-a-uuid")
	isErr(t, err, store.ErrNotFound)
	_, err = s.GetUserByGitHubID(ctx, 43)
	isErr(t, err, store.ErrNotFound)
}

func TestOrgsAndMembers(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	owner := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 1, GitHubLogin: "owner"}))
	other := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 2, GitHubLogin: "other"}))

	org := must(s.CreateOrg(ctx, "Zeta", owner.ID))
	org2 := must(s.CreateOrg(ctx, "Alpha", owner.ID))
	if org.ID == "" || org.Name != "Zeta" || org.DefaultLLMProvider != "" {
		t.Fatalf("org %+v", org)
	}

	// A missing owner rolls the whole thing back.
	_, err := s.CreateOrg(ctx, "Ghost", missingID)
	isErr(t, err, store.ErrNotFound)
	var n int
	ok(t, s.Pool().QueryRow(ctx, `SELECT count(*) FROM orgs WHERE name = 'Ghost'`).Scan(&n))
	if n != 0 {
		t.Errorf("CreateOrg left %d orphan orgs", n)
	}

	ok(t, s.SetOrgDefaultLLMProvider(ctx, org.ID, "gemini"))
	if got := must(s.GetOrg(ctx, org.ID)); got.DefaultLLMProvider != "gemini" {
		t.Errorf("provider = %q", got.DefaultLLMProvider)
	}
	ok(t, s.SetOrgDefaultLLMProvider(ctx, org.ID, ""))
	if got := must(s.GetOrg(ctx, org.ID)); got.DefaultLLMProvider != "" {
		t.Errorf("provider = %q, want cleared", got.DefaultLLMProvider)
	}
	isErr(t, s.SetOrgDefaultLLMProvider(ctx, missingID, "x"), store.ErrNotFound)
	_, err = s.GetOrg(ctx, missingID)
	isErr(t, err, store.ErrNotFound)

	list := must(s.ListOrgsForUser(ctx, owner.ID))
	if len(list) != 2 || list[0].Org.Name != "Alpha" || list[1].Org.Name != "Zeta" ||
		list[0].Role != store.RoleOwner || list[0].Org.ID != org2.ID {
		t.Fatalf("ListOrgsForUser = %+v", list)
	}
	if l := must(s.ListOrgsForUser(ctx, other.ID)); len(l) != 0 {
		t.Errorf("other user orgs = %+v", l)
	}

	if has := must(s.HasOrgRole(ctx, org.ID, other.ID, store.RoleMember)); has {
		t.Error("non-member has member role")
	}
	_, err = s.GetOrgRole(ctx, org.ID, other.ID)
	isErr(t, err, store.ErrNotFound)

	ok(t, s.AddOrgMember(ctx, org.ID, other.ID, store.RoleMember))
	if r := must(s.GetOrgRole(ctx, org.ID, other.ID)); r != store.RoleMember {
		t.Errorf("role = %q", r)
	}
	if has := must(s.HasOrgRole(ctx, org.ID, other.ID, store.RoleAdmin)); has {
		t.Error("member has admin")
	}
	ok(t, s.AddOrgMember(ctx, org.ID, other.ID, store.RoleAdmin))
	if has := must(s.HasOrgRole(ctx, org.ID, other.ID, store.RoleAdmin)); !has {
		t.Error("admin lacks admin")
	}
	if has := must(s.HasOrgRole(ctx, org.ID, other.ID, store.RoleOwner)); has {
		t.Error("admin has owner")
	}
	if err := s.AddOrgMember(ctx, org.ID, other.ID, "root"); err == nil {
		t.Error("invalid role accepted")
	}
	isErr(t, s.AddOrgMember(ctx, org.ID, missingID, store.RoleMember), store.ErrNotFound)

	ok(t, s.RemoveOrgMember(ctx, org.ID, other.ID))
	isErr(t, s.RemoveOrgMember(ctx, org.ID, other.ID), store.ErrNotFound)
	if has := must(s.HasOrgRole(ctx, org.ID, other.ID, store.RoleMember)); has {
		t.Error("removed member still has role")
	}
}

func TestInstallationsAndRepos(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	u := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 1, GitHubLogin: "u"}))
	org := must(s.CreateOrg(ctx, "A", u.ID))
	org2 := must(s.CreateOrg(ctx, "B", u.ID))

	inst := must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: org.ID, InstallationID: 100, AccountLogin: "acme",
	}))
	if inst.Status != store.InstallationActive || inst.OrgID != org.ID {
		t.Fatalf("installation %+v", inst)
	}
	inst2 := must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: org.ID, InstallationID: 100, AccountLogin: "acme-renamed", Status: store.InstallationSuspended,
	}))
	if inst2.ID != inst.ID || inst2.AccountLogin != "acme-renamed" || inst2.Status != store.InstallationSuspended {
		t.Fatalf("re-upsert %+v", inst2)
	}
	_, err := s.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: org2.ID, InstallationID: 100, AccountLogin: "hijack",
	})
	isErr(t, err, store.ErrConflict)
	if got := must(s.GetInstallation(ctx, 100)); got.OrgID != org.ID || got.AccountLogin != "acme-renamed" {
		t.Fatalf("conflicting upsert changed the row: %+v", got)
	}
	_, err = s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: missingID, InstallationID: 101})
	isErr(t, err, store.ErrNotFound)
	_, err = s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: org.ID, InstallationID: 101, Status: "bogus"})
	if err == nil {
		t.Error("invalid status accepted")
	}

	ok(t, s.SetInstallationStatus(ctx, 100, store.InstallationActive))
	if got := must(s.GetInstallation(ctx, 100)); got.Status != store.InstallationActive {
		t.Errorf("status = %q", got.Status)
	}
	isErr(t, s.SetInstallationStatus(ctx, 999, store.InstallationDeleted), store.ErrNotFound)
	_, err = s.GetInstallation(ctx, 999)
	isErr(t, err, store.ErrNotFound)
	must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: org.ID, InstallationID: 200, AccountLogin: "acme2"}))
	if l := must(s.ListInstallations(ctx, org.ID)); len(l) != 2 || l[0].InstallationID != 100 {
		t.Errorf("ListInstallations = %+v", l)
	}

	// Repos.
	r1 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 100, GitHubRepoID: 1, FullName: "acme/web"}))
	if r1.OrgID != org.ID || r1.DefaultBranch != "main" || !r1.Accessible || r1.Private {
		t.Fatalf("repo %+v", r1)
	}
	r1b := must(s.UpsertRepo(ctx, store.UpsertRepoParams{
		InstallationID: 100, GitHubRepoID: 1, FullName: "acme/web2", DefaultBranch: "trunk", Private: true,
	}))
	if r1b.ID != r1.ID || r1b.FullName != "acme/web2" || r1b.DefaultBranch != "trunk" || !r1b.Private {
		t.Fatalf("re-upsert repo %+v", r1b)
	}
	r2 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 100, GitHubRepoID: 2, FullName: "acme/api"}))
	r3 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 200, GitHubRepoID: 3, FullName: "acme/cli"}))
	_, err = s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 999, GitHubRepoID: 4, FullName: "x/y"})
	isErr(t, err, store.ErrNotFound)

	names := func(rs []store.Repo) []string {
		var out []string
		for _, r := range rs {
			out = append(out, r.FullName)
		}
		return out
	}
	if got := names(must(s.ListRepos(ctx, org.ID, false))); !slices.Equal(got, []string{"acme/api", "acme/cli", "acme/web2"}) {
		t.Fatalf("ListRepos = %v", got)
	}
	if got := must(s.GetRepo(ctx, r2.ID)); got.FullName != "acme/api" {
		t.Errorf("GetRepo = %+v", got)
	}
	_, err = s.GetRepo(ctx, missingID)
	isErr(t, err, store.ErrNotFound)

	// Removing repos 2 and 3 from installation 100 touches only repo 2.
	if n := must(s.SetReposInaccessible(ctx, 100, []int64{2, 3})); n != 1 {
		t.Errorf("SetReposInaccessible changed %d, want 1", n)
	}
	if got := names(must(s.ListRepos(ctx, org.ID, false))); !slices.Equal(got, []string{"acme/cli", "acme/web2"}) {
		t.Fatalf("ListRepos after removal = %v", got)
	}
	if got := names(must(s.ListRepos(ctx, org.ID, true))); len(got) != 3 {
		t.Fatalf("ListRepos(include) = %v", got)
	}
	// Re-adding makes it accessible again.
	must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 100, GitHubRepoID: 2, FullName: "acme/api"}))
	if got := must(s.GetRepo(ctx, r2.ID)); !got.Accessible {
		t.Error("re-added repo still inaccessible")
	}
	// Uninstall.
	if n := must(s.SetInstallationReposInaccessible(ctx, 100)); n != 2 {
		t.Errorf("SetInstallationReposInaccessible changed %d, want 2", n)
	}
	if got := names(must(s.ListRepos(ctx, org.ID, false))); !slices.Equal(got, []string{"acme/cli"}) {
		t.Fatalf("ListRepos after uninstall = %v", got)
	}
	_ = r3
}

func TestRepoProfiles(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)

	_, err := s.GetRepoProfile(ctx, f.Repo.ID)
	isErr(t, err, store.ErrNotFound)

	p := must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{
		RepoID: f.Repo.ID, InstallCmd: "npm ci", TestCmd: "npm test", Knowledge: "uses pnpm workspaces",
	}))
	if p.InstallCmd != "npm ci" || p.Snapshot != (store.Snapshot{}) {
		t.Fatalf("profile %+v", p)
	}
	snap := store.Snapshot{ID: "snap-1", Provider: "daytona", LockHash: "abc"}
	ok(t, s.SetRepoSnapshot(ctx, f.Repo.ID, snap))
	if got := must(s.GetRepoProfile(ctx, f.Repo.ID)); got.Snapshot != snap || got.TestCmd != "npm test" {
		t.Fatalf("after snapshot %+v", got)
	}
	// Same install command keeps the snapshot.
	p = must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{
		RepoID: f.Repo.ID, InstallCmd: "npm ci", TestCmd: "npm run test:ci",
	}))
	if p.Snapshot != snap || p.TestCmd != "npm run test:ci" || p.Knowledge != "" {
		t.Fatalf("same install cmd %+v", p)
	}
	// A new install command clears it.
	p = must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{RepoID: f.Repo.ID, InstallCmd: "pnpm i"}))
	if p.Snapshot != (store.Snapshot{}) {
		t.Fatalf("new install cmd kept snapshot %+v", p.Snapshot)
	}
	ok(t, s.SetRepoSnapshot(ctx, f.Repo.ID, snap))
	ok(t, s.SetRepoSnapshot(ctx, f.Repo.ID, store.Snapshot{}))
	if got := must(s.GetRepoProfile(ctx, f.Repo.ID)); got.Snapshot != (store.Snapshot{}) || got.InstallCmd != "pnpm i" {
		t.Fatalf("cleared snapshot %+v", got)
	}

	// A snapshot before onboarding creates an empty profile.
	r2 := must(s.UpsertRepo(ctx, store.UpsertRepoParams{
		InstallationID: f.Installation.InstallationID, GitHubRepoID: f.Repo.GitHubRepoID + 1, FullName: "acme/two",
	}))
	ok(t, s.SetRepoSnapshot(ctx, r2.ID, snap))
	if got := must(s.GetRepoProfile(ctx, r2.ID)); got.Snapshot != snap || got.InstallCmd != "" {
		t.Fatalf("snapshot-first profile %+v", got)
	}
	isErr(t, s.SetRepoSnapshot(ctx, missingID, snap), store.ErrNotFound)
	_, err = s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{RepoID: missingID})
	isErr(t, err, store.ErrNotFound)
}

func TestSessions(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)

	x := f.Session
	if x.Status != events.StatusQueued || x.FinishedAt != nil || x.CancelRequestedAt != nil || x.CreatedBy != f.User.ID {
		t.Fatalf("seeded session %+v", x)
	}
	if len(x.Repos) != 1 || x.Repos[0].RepoID != f.Repo.ID || x.Repos[0].Branch != "agent/"+x.ID ||
		x.Repos[0].FullName != "acme/api" || x.Repos[0].InstallationID != f.Installation.InstallationID ||
		x.Repos[0].DefaultBranch != "main" {
		t.Fatalf("session repos %+v", x.Repos)
	}

	// Repos of another org, unknown repos and inaccessible repos are refused
	// and leave nothing behind.
	org2 := must(s.CreateOrg(ctx, "Other", f.User.ID))
	inst2 := must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: org2.ID, InstallationID: 7777, AccountLogin: "o"}))
	foreign := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst2.InstallationID, GitHubRepoID: 7777, FullName: "o/r"}))
	gone := must(s.UpsertRepo(ctx, store.UpsertRepoParams{
		InstallationID: f.Installation.InstallationID, GitHubRepoID: 8888, FullName: "acme/gone",
	}))
	must(s.SetReposInaccessible(ctx, f.Installation.InstallationID, []int64{8888}))
	for _, ids := range [][]string{{f.Repo.ID, foreign.ID}, {missingID}, {gone.ID}, {"bad"}} {
		_, err := s.CreateSession(ctx, store.CreateSessionParams{
			OrgID: f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: ids,
		})
		isErr(t, err, store.ErrNotFound)
	}
	var n int
	ok(t, s.Pool().QueryRow(ctx, `SELECT count(*) FROM sessions`).Scan(&n))
	if n != 1 {
		t.Fatalf("refused creates left %d sessions, want 1", n)
	}
	if _, err := s.CreateSession(ctx, store.CreateSessionParams{OrgID: f.Org.ID, Prompt: "p"}); err == nil {
		t.Error("session without provider accepted")
	}

	// Trigger, routine and duplicate repo IDs.
	trig := json.RawMessage(`{"kind":"issue","number":12}`)
	routine := "11111111-2222-4333-8444-555555555555"
	y := must(s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: f.Org.ID, Prompt: "from routine", Provider: "gemini", Model: "g",
		RepoIDs: []string{f.Repo.ID, f.Repo.ID}, RoutineID: routine, Trigger: trig,
	}))
	if y.CreatedBy != "" || y.RoutineID != routine || len(y.Repos) != 1 {
		t.Fatalf("routine session %+v", y)
	}
	var gotTrig map[string]any
	ok(t, json.Unmarshal(y.Trigger, &gotTrig))
	if gotTrig["kind"] != "issue" {
		t.Errorf("trigger = %s", y.Trigger)
	}
	if must(s.GetSession(ctx, x.ID)).Trigger != nil {
		t.Error("nil trigger came back non-nil")
	}

	// Status.
	ok(t, s.SetSessionStatus(ctx, x.ID, events.StatusRunning, ""))
	got := must(s.GetSession(ctx, x.ID))
	if got.Status != events.StatusRunning || got.FinishedAt != nil {
		t.Fatalf("running %+v", got)
	}
	ok(t, s.SetSessionStatus(ctx, x.ID, events.StatusFailed, "sandbox_provision_failed"))
	got = must(s.GetSession(ctx, x.ID))
	if got.Status != events.StatusFailed || got.StatusReason != "sandbox_provision_failed" || got.FinishedAt == nil {
		t.Fatalf("failed %+v", got)
	}
	ok(t, s.SetSessionStatus(ctx, x.ID, events.StatusRunning, ""))
	if got := must(s.GetSession(ctx, x.ID)); got.FinishedAt != nil || got.StatusReason != "" {
		t.Fatalf("resumed %+v", got)
	}
	if err := s.SetSessionStatus(ctx, x.ID, "exploded", ""); err == nil {
		t.Error("invalid status accepted")
	}
	isErr(t, s.SetSessionStatus(ctx, missingID, events.StatusRunning, ""), store.ErrNotFound)

	// Sandbox.
	ok(t, s.SetSessionSandbox(ctx, x.ID, "sb-1", "docker"))
	if got := must(s.GetSession(ctx, x.ID)); got.SandboxID != "sb-1" || got.SandboxProvider != "docker" {
		t.Fatalf("sandbox %+v", got)
	}
	ok(t, s.SetSessionSandbox(ctx, x.ID, "", ""))
	if got := must(s.GetSession(ctx, x.ID)); got.SandboxID != "" {
		t.Fatalf("sandbox not cleared %+v", got)
	}
	isErr(t, s.SetSessionSandbox(ctx, missingID, "a", "b"), store.ErrNotFound)

	// Cancel.
	if must(s.IsCancelRequested(ctx, x.ID)) {
		t.Fatal("cancel requested before Stop")
	}
	ok(t, s.RequestCancel(ctx, x.ID))
	first := must(s.GetSession(ctx, x.ID)).CancelRequestedAt
	ok(t, s.RequestCancel(ctx, x.ID))
	if !must(s.IsCancelRequested(ctx, x.ID)) {
		t.Fatal("cancel not requested after Stop")
	}
	if again := must(s.GetSession(ctx, x.ID)).CancelRequestedAt; first == nil || !again.Equal(*first) {
		t.Errorf("second RequestCancel moved the time: %v -> %v", first, again)
	}
	ok(t, s.ClearCancelRequest(ctx, x.ID))
	if must(s.IsCancelRequested(ctx, x.ID)) {
		t.Fatal("cancel still requested after clear")
	}
	isErr(t, s.RequestCancel(ctx, missingID), store.ErrNotFound)
	_, err := s.IsCancelRequested(ctx, missingID)
	isErr(t, err, store.ErrNotFound)

	// Provider and model: allowed until agent state exists.
	ok(t, s.SetSessionModel(ctx, x.ID, "gemini", "gemini-3-pro"))
	if got := must(s.GetSession(ctx, x.ID)); got.Provider != "gemini" || got.Model != "gemini-3-pro" {
		t.Fatalf("model %+v", got)
	}
	ok(t, s.SaveAgentState(ctx, store.AgentState{SessionID: x.ID, Provider: "gemini"}))
	isErr(t, s.SetSessionModel(ctx, x.ID, "anthropic", "claude-opus-5-5"), store.ErrConflict)
	isErr(t, s.SetSessionModel(ctx, missingID, "a", "b"), store.ErrNotFound)

	// Session repos.
	ok(t, s.SetSessionRepoBranch(ctx, x.ID, f.Repo.ID, "agent/custom"))
	ok(t, s.SetSessionRepoPR(ctx, x.ID, f.Repo.ID, 17, "https://github.com/acme/api/pull/17"))
	repos := must(s.ListSessionRepos(ctx, x.ID))
	if len(repos) != 1 || repos[0].Branch != "agent/custom" || repos[0].PRNumber != 17 ||
		repos[0].PRURL != "https://github.com/acme/api/pull/17" {
		t.Fatalf("session repos %+v", repos)
	}
	isErr(t, s.SetSessionRepoBranch(ctx, x.ID, foreign.ID, "b"), store.ErrNotFound)
	isErr(t, s.SetSessionRepoPR(ctx, missingID, f.Repo.ID, 1, "u"), store.ErrNotFound)

	_, err = s.GetSession(ctx, missingID)
	isErr(t, err, store.ErrNotFound)
	_, err = s.GetSession(ctx, "nope")
	isErr(t, err, store.ErrNotFound)
}

func TestListSessionsPaging(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	want := []string{f.Session.ID}
	for i := 0; i < 6; i++ {
		x := must(s.CreateSession(ctx, store.CreateSessionParams{
			OrgID: f.Org.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{f.Repo.ID},
		}))
		want = append(want, x.ID)
	}
	// Two sessions with the same created_at must still page stably.
	_, err := s.Pool().Exec(ctx, `UPDATE sessions SET created_at = $2 WHERE id = ANY($1::uuid[])`,
		want[3:5], time.Now().Add(-time.Hour))
	ok(t, err)

	// Expected order: created_at DESC, id DESC.
	var expect []string
	rows, err := s.Pool().Query(ctx, `SELECT id FROM sessions WHERE org_id = $1 ORDER BY created_at DESC, id DESC`, f.Org.ID)
	ok(t, err)
	for rows.Next() {
		var id string
		ok(t, rows.Scan(&id))
		expect = append(expect, id)
	}
	ok(t, rows.Err())

	var got []string
	cursor, pages := "", 0
	for {
		page := must(s.ListSessions(ctx, store.ListSessionsParams{OrgID: f.Org.ID, Limit: 3, Cursor: cursor}))
		pages++
		for _, x := range page.Sessions {
			got = append(got, x.ID)
			if len(x.Repos) != 1 {
				t.Errorf("session %s has %d repos in list", x.ID, len(x.Repos))
			}
		}
		if page.NextCursor == "" {
			break
		}
		cursor = page.NextCursor
		if pages > 10 {
			t.Fatal("paging does not terminate")
		}
	}
	if !slices.Equal(got, expect) || pages != 3 {
		t.Fatalf("paged %v in %d pages, want %v in 3", got, pages, expect)
	}

	all := must(s.ListSessions(ctx, store.ListSessionsParams{OrgID: f.Org.ID}))
	if len(all.Sessions) != 7 || all.NextCursor != "" {
		t.Fatalf("default page: %d sessions, cursor %q", len(all.Sessions), all.NextCursor)
	}
	_, err = s.ListSessions(ctx, store.ListSessionsParams{OrgID: f.Org.ID, Cursor: "garbage"})
	isErr(t, err, store.ErrInvalidCursor)
	if p := must(s.ListSessions(ctx, store.ListSessionsParams{OrgID: missingID})); len(p.Sessions) != 0 {
		t.Error("sessions listed for a missing org")
	}
}

func TestAgentState(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	id := f.Session.ID

	_, err := s.LoadAgentState(ctx, id)
	isErr(t, err, store.ErrNotFound)

	ok(t, s.SaveAgentState(ctx, store.AgentState{SessionID: id, Provider: "anthropic"}))
	st := must(s.LoadAgentState(ctx, id))
	if string(st.LLMMessages) != "[]" || string(st.RawMessages) != "[]" || st.Provider != "anthropic" {
		t.Fatalf("empty state %+v", st)
	}

	msgs := json.RawMessage(`[{"role":"user","content":[{"type":"text","text":"hi"}]}]`)
	raw := json.RawMessage(`[{"id":"msg_1","content":[{"type":"thinking","signature":"sig=="}]}]`)
	ok(t, s.SaveAgentState(ctx, store.AgentState{
		SessionID: id, Provider: "anthropic", LLMMessages: msgs, RawMessages: raw, Summary: "said hi",
	}))
	st = must(s.LoadAgentState(ctx, id))
	if !jsonEqual(t, st.LLMMessages, msgs) || !jsonEqual(t, st.RawMessages, raw) || st.Summary != "said hi" {
		t.Fatalf("state %+v", st)
	}

	isErr(t, s.SaveAgentState(ctx, store.AgentState{SessionID: id, Provider: "gemini"}), store.ErrProviderMismatch)
	if st2 := must(s.LoadAgentState(ctx, id)); st2.Summary != "said hi" {
		t.Error("mismatched save overwrote state")
	}
	isErr(t, s.SaveAgentState(ctx, store.AgentState{SessionID: missingID, Provider: "anthropic"}), store.ErrNotFound)
	if err := s.SaveAgentState(ctx, store.AgentState{SessionID: id, Provider: "anthropic", LLMMessages: json.RawMessage(`[`)}); err == nil {
		t.Error("invalid JSON accepted")
	}
}

func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var x, y any
	ok(t, json.Unmarshal(a, &x))
	ok(t, json.Unmarshal(b, &y))
	ja, _ := json.Marshal(x)
	jb, _ := json.Marshal(y)
	return bytes.Equal(ja, jb)
}

func TestSecrets(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	kr := must(secrets.NewKeyring(bytes.Repeat([]byte{7}, secrets.KeySize)))

	put := func(repoID, name, value string) store.SecretMeta {
		t.Helper()
		ct := must(kr.Seal([]byte(value), store.SecretAAD(f.Org.ID, repoID, name)))
		return must(s.PutSecret(ctx, store.PutSecretParams{
			OrgID: f.Org.ID, RepoID: repoID, Name: name, Ciphertext: ct, CreatedBy: f.User.ID,
		}))
	}
	orgTok := put("", "API_TOKEN", "org-value")
	put("", "SHARED", "org-shared")
	repoTok := put(f.Repo.ID, "SHARED", "repo-shared")
	if orgTok.RepoID != "" || repoTok.RepoID != f.Repo.ID || orgTok.CreatedBy != f.User.ID {
		t.Fatalf("metas %+v %+v", orgTok, repoTok)
	}
	// Put again replaces the value in place.
	again := put("", "API_TOKEN", "org-value-2")
	if again.ID != orgTok.ID {
		t.Errorf("re-put made a new row: %s vs %s", again.ID, orgTok.ID)
	}

	names := func(ms []store.SecretMeta) []string {
		var out []string
		for _, m := range ms {
			out = append(out, m.Name)
		}
		return out
	}
	if got := names(must(s.ListSecretNames(ctx, f.Org.ID, ""))); !slices.Equal(got, []string{"API_TOKEN", "SHARED"}) {
		t.Errorf("org secrets = %v", got)
	}
	if got := names(must(s.ListSecretNames(ctx, f.Org.ID, f.Repo.ID))); !slices.Equal(got, []string{"SHARED"}) {
		t.Errorf("repo secrets = %v", got)
	}

	// Tenancy: a repo of another org, bad names, empty ciphertext.
	org2 := must(s.CreateOrg(ctx, "Other", f.User.ID))
	_, err := s.PutSecret(ctx, store.PutSecretParams{OrgID: org2.ID, RepoID: f.Repo.ID, Name: "X", Ciphertext: []byte{1}})
	isErr(t, err, store.ErrNotFound)
	_, err = s.PutSecret(ctx, store.PutSecretParams{OrgID: missingID, Name: "X", Ciphertext: []byte{1}})
	isErr(t, err, store.ErrNotFound)
	if _, err := s.PutSecret(ctx, store.PutSecretParams{OrgID: f.Org.ID, Name: "BAD-NAME", Ciphertext: []byte{1}}); err == nil {
		t.Error("invalid name accepted")
	}
	if _, err := s.PutSecret(ctx, store.PutSecretParams{OrgID: f.Org.ID, Name: "EMPTY"}); err == nil {
		t.Error("empty ciphertext accepted")
	}
	if got := must(s.ListSecretNames(ctx, org2.ID, "")); len(got) != 0 {
		t.Errorf("other org sees %v", names(got))
	}

	// Load: org secrets first, then repo secrets; open each with its AAD
	// and apply in order so the repo value wins.
	cts := must(s.LoadSecretCiphertexts(ctx, f.Org.ID, f.Repo.ID))
	env := map[string]string{}
	var order []string
	for _, c := range cts {
		v := must(kr.Open(c.Ciphertext, c.AAD()))
		env[c.Name] = string(v)
		order = append(order, c.RepoID+"/"+c.Name)
	}
	if !slices.Equal(order, []string{"/API_TOKEN", "/SHARED", f.Repo.ID + "/SHARED"}) {
		t.Errorf("load order = %v", order)
	}
	if env["API_TOKEN"] != "org-value-2" || env["SHARED"] != "repo-shared" {
		t.Errorf("env = %v", env)
	}
	// A ciphertext moved to another row's identity does not open.
	if _, err := kr.Open(cts[0].Ciphertext, store.SecretAAD(f.Org.ID, "", "SHARED")); !errors.Is(err, secrets.ErrDecrypt) {
		t.Errorf("moved ciphertext opened: %v", err)
	}
	if got := must(s.LoadSecretCiphertexts(ctx, f.Org.ID)); len(got) != 2 {
		t.Errorf("org-only load = %d rows, want 2", len(got))
	}
	if got := must(s.LoadSecretCiphertexts(ctx, org2.ID, f.Repo.ID)); len(got) != 0 {
		t.Errorf("other org loaded %d secrets", len(got))
	}

	ok(t, s.DeleteSecret(ctx, f.Org.ID, f.Repo.ID, "SHARED"))
	isErr(t, s.DeleteSecret(ctx, f.Org.ID, f.Repo.ID, "SHARED"), store.ErrNotFound)
	if got := names(must(s.ListSecretNames(ctx, f.Org.ID, ""))); len(got) != 2 {
		t.Errorf("deleting the repo secret touched org secrets: %v", got)
	}
	ok(t, s.DeleteSecret(ctx, f.Org.ID, "", "SHARED"))
	isErr(t, s.DeleteSecret(ctx, org2.ID, "", "API_TOKEN"), store.ErrNotFound)
}

func TestWithTx(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	boom := errors.New("boom")

	err := s.WithTx(ctx, func(tx *store.Store) error {
		ok(t, tx.SetSessionStatus(ctx, f.Session.ID, events.StatusRunning, ""))
		if _, err := tx.Events().Append(ctx, f.Session.ID, events.TypeSessionStatus,
			events.SessionStatusPayload{Status: events.StatusRunning}); err != nil {
			return err
		}
		return boom
	})
	isErr(t, err, boom)
	if got := must(s.GetSession(ctx, f.Session.ID)); got.Status != events.StatusQueued {
		t.Errorf("rolled-back status = %q", got.Status)
	}
	if evs := must(s.Events().After(ctx, f.Session.ID, 0, 0)); len(evs) != 0 {
		t.Errorf("rolled-back events = %d", len(evs))
	}

	// Nested: the inner failure rolls back to its savepoint only.
	ok(t, s.WithTx(ctx, func(tx *store.Store) error {
		ok(t, tx.SetSessionStatus(ctx, f.Session.ID, events.StatusRunning, ""))
		inner := tx.WithTx(ctx, func(tx2 *store.Store) error {
			ok(t, tx2.SetSessionSandbox(ctx, f.Session.ID, "sb", "docker"))
			return boom
		})
		isErr(t, inner, boom)
		return nil
	}))
	got := must(s.GetSession(ctx, f.Session.ID))
	if got.Status != events.StatusRunning || got.SandboxID != "" {
		t.Errorf("after nested tx: status %q sandbox %q", got.Status, got.SandboxID)
	}

	// A panic rolls back and propagates.
	func() {
		defer func() {
			if recover() == nil {
				t.Error("panic swallowed")
			}
		}()
		_ = s.WithTx(ctx, func(tx *store.Store) error {
			ok(t, tx.SetSessionStatus(ctx, f.Session.ID, events.StatusFailed, "x"))
			panic("boom")
		})
	}()
	if got := must(s.GetSession(ctx, f.Session.ID)); got.Status != events.StatusRunning {
		t.Errorf("status after panic = %q", got.Status)
	}
}
