package store_test

import (
	"testing"

	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

// A repo row follows its GitHub repo to whichever installation last
// listed it, even one linked to another org (an uninstall from org A,
// then an install into org B). Org A's sessions must then see the repo as
// inaccessible, with no installation to mint a token on, and org B must
// not inherit org A's profile or snapshot.
func TestRepoMovedToAnotherOrg(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	u := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 1, GitHubLogin: "u"}))
	orgA := must(s.CreateOrg(ctx, "A", u.ID))
	orgB := must(s.CreateOrg(ctx, "B", u.ID))
	must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: orgA.ID, InstallationID: 100, AccountLogin: "acme"}))
	must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: orgA.ID, InstallationID: 101, AccountLogin: "acme-two"}))
	must(s.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: orgB.ID, InstallationID: 200, AccountLogin: "acme"}))

	repo := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 100, GitHubRepoID: 7, FullName: "acme/web"}))
	must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{RepoID: repo.ID, InstallCmd: "npm ci", Knowledge: "org A notes"}))
	ok(t, s.SetRepoSnapshot(ctx, repo.ID, store.Snapshot{ID: "snap-a", Provider: "docker", LockHash: "h"}))
	sessA := must(s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: orgA.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{repo.ID},
	}))
	sessionRepo := func(id string) store.SessionRepo {
		t.Helper()
		x := must(s.GetSession(ctx, id))
		if len(x.Repos) != 1 {
			t.Fatalf("session repos %+v", x.Repos)
		}
		return x.Repos[0]
	}
	if sr := sessionRepo(sessA.ID); sr.Inaccessible || sr.InstallationID != 100 {
		t.Fatalf("fresh session repo %+v", sr)
	}

	// A transfer between two installations of the same org keeps the
	// profile and the session's access.
	must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 101, GitHubRepoID: 7, FullName: "acme-two/web"}))
	if p := must(s.GetRepoProfile(ctx, repo.ID)); p.Knowledge != "org A notes" || p.Snapshot.ID != "snap-a" {
		t.Fatalf("profile after a same-org transfer %+v", p)
	}
	if sr := sessionRepo(sessA.ID); sr.Inaccessible || sr.InstallationID != 101 {
		t.Fatalf("session repo after a same-org transfer %+v", sr)
	}

	// Removed from the installation: inaccessible, same installation.
	must(s.SetInstallationReposInaccessible(ctx, 101))
	if sr := sessionRepo(sessA.ID); !sr.Inaccessible || sr.InstallationID != 101 {
		t.Fatalf("session repo after uninstall %+v", sr)
	}

	// Installed into org B: the row moves, and is accessible again, but
	// not to org A's session. The profile and snapshot are gone.
	moved := must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 200, GitHubRepoID: 7, FullName: "acme/web"}))
	if moved.ID != repo.ID || moved.OrgID != orgB.ID || !moved.Accessible {
		t.Fatalf("moved repo %+v", moved)
	}
	if sr := sessionRepo(sessA.ID); !sr.Inaccessible || sr.InstallationID != 0 {
		t.Fatalf("org A's session repo after the move %+v: it must not reach org B's installation", sr)
	}
	page := must(s.ListSessions(ctx, store.ListSessionsParams{OrgID: orgA.ID}))
	if len(page.Sessions) != 1 || len(page.Sessions[0].Repos) != 1 || !page.Sessions[0].Repos[0].Inaccessible || page.Sessions[0].Repos[0].InstallationID != 0 {
		t.Fatalf("ListSessions after the move %+v", page.Sessions)
	}
	if _, err := s.GetRepoProfile(ctx, repo.ID); err == nil {
		t.Fatal("org B inherited org A's repo profile and snapshot")
	} else {
		isErr(t, err, store.ErrNotFound)
	}
	sessB := must(s.CreateSession(ctx, store.CreateSessionParams{
		OrgID: orgB.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{repo.ID},
	}))
	if sr := sessionRepo(sessB.ID); sr.Inaccessible || sr.InstallationID != 200 {
		t.Fatalf("org B's session repo %+v", sr)
	}
	// Re-syncing it under org B keeps org B's own profile.
	must(s.UpsertRepoProfile(ctx, store.UpsertRepoProfileParams{RepoID: repo.ID, Knowledge: "org B notes"}))
	must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 200, GitHubRepoID: 7, FullName: "acme/web"}))
	if p := must(s.GetRepoProfile(ctx, repo.ID)); p.Knowledge != "org B notes" {
		t.Fatalf("profile after a same-org re-sync %+v", p)
	}

	// Moving back flips both sessions, and drops org B's profile.
	must(s.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: 100, GitHubRepoID: 7, FullName: "acme/web"}))
	if sr := sessionRepo(sessA.ID); sr.Inaccessible || sr.InstallationID != 100 {
		t.Fatalf("org A's session repo after moving back %+v", sr)
	}
	if sr := sessionRepo(sessB.ID); !sr.Inaccessible || sr.InstallationID != 0 {
		t.Fatalf("org B's session repo after moving back %+v", sr)
	}
	if _, err := s.GetRepoProfile(ctx, repo.ID); err == nil {
		t.Fatal("org A got org B's profile back")
	}
}
