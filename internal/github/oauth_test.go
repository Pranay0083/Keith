package github

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestAuthCodeURL(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	raw, err := a.AuthCodeURL("st4te")
	if err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := u.Scheme+"://"+u.Host+u.Path, f.srv.URL+"/login/oauth/authorize"; got != want {
		t.Errorf("endpoint = %s, want %s", got, want)
	}
	want := url.Values{
		"client_id":    {testClientID},
		"state":        {"st4te"},
		"redirect_uri": {"https://keith.example/auth/github/callback"},
	}
	if got := u.Query(); !reflect.DeepEqual(got, want) {
		t.Errorf("query = %v, want %v", got, want)
	}

	if _, err := a.AuthCodeURL(""); err == nil {
		t.Error("AuthCodeURL(\"\") succeeded")
	}
	noClient := f.app(t, func(c *AppConfig) { c.ClientSecret = "" })
	if _, err := noClient.AuthCodeURL("s"); err == nil {
		t.Error("AuthCodeURL without a client secret succeeded")
	}
}

func TestInstallURL(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	got, err := a.InstallURL("keith-dev", "abc")
	if err != nil {
		t.Fatal(err)
	}
	if want := f.srv.URL + "/apps/keith-dev/installations/new?state=abc"; got != want {
		t.Errorf("InstallURL = %s, want %s", got, want)
	}
	for _, slug := range []string{"", "../x", "Keith", "a/b"} {
		if _, err := a.InstallURL(slug, "abc"); err == nil {
			t.Errorf("InstallURL(%q) succeeded", slug)
		}
	}
	if _, err := a.InstallURL("keith-dev", ""); err == nil {
		t.Error("InstallURL without state succeeded")
	}
}

func TestSlug(t *testing.T) {
	f := newFakeGitHub(t)
	f.mux.HandleFunc("GET /app", func(w http.ResponseWriter, r *http.Request) {
		if _, _, err := f.parseAppJWT(r); err != nil {
			writeJSON(w, http.StatusUnauthorized, nil)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": testAppID, "slug": "keith-dev"})
	})
	got, err := f.app(t).Slug(context.Background())
	if err != nil || got != "keith-dev" {
		t.Fatalf("Slug = %q, %v", got, err)
	}
}

func TestExchange(t *testing.T) {
	f := newFakeGitHub(t)
	f.mux.HandleFunc("POST /login/oauth/access_token", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "application/json" {
			http.Error(w, "want json", http.StatusNotAcceptable)
			return
		}
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		want := url.Values{
			"client_id":     {testClientID},
			"client_secret": {testSecret},
			"code":          {r.PostForm.Get("code")},
			"redirect_uri":  {"https://keith.example/auth/github/callback"},
		}
		if !reflect.DeepEqual(r.PostForm, want) {
			t.Errorf("form = %v, want %v", r.PostForm, want)
		}
		switch r.PostForm.Get("code") {
		case "good":
			writeJSON(w, http.StatusOK, map[string]any{
				"access_token": "ghu_user", "token_type": "bearer", "scope": "",
				"expires_in": 28800, "refresh_token": "ghr_refresh", "refresh_token_expires_in": 15897600,
			})
		case "broken":
			w.WriteHeader(http.StatusInternalServerError)
		default:
			writeJSON(w, http.StatusOK, map[string]any{
				"error": "bad_verification_code", "error_description": "The code passed is incorrect or expired.",
			})
		}
	})
	a := f.app(t)
	ctx := context.Background()

	tok, err := a.Exchange(ctx, "good")
	if err != nil {
		t.Fatal(err)
	}
	now := f.clock.Now()
	want := UserToken{
		AccessToken: "ghu_user", TokenType: "bearer", RefreshToken: "ghr_refresh",
		ExpiresAt: now.Add(8 * time.Hour), RefreshTokenExpiresAt: now.Add(15897600 * time.Second),
	}
	if !reflect.DeepEqual(tok, want) {
		t.Errorf("token = %+v, want %+v", tok, want)
	}
	if s := fmt.Sprintf("%v %#v", tok, tok); strings.Contains(s, "ghu_user") || strings.Contains(s, "ghr_refresh") {
		t.Errorf("UserToken formatting leaked a token: %s", s)
	}

	if _, err := a.Exchange(ctx, "stale"); err == nil || !strings.Contains(err.Error(), "bad_verification_code") {
		t.Errorf("bad code: err = %v", err)
	}
	if _, err := a.Exchange(ctx, "broken"); err == nil {
		t.Error("500 response: no error")
	}
	if _, err := a.Exchange(ctx, ""); err == nil {
		t.Error("empty code: no error")
	}
}

func TestGetUser(t *testing.T) {
	f := newFakeGitHub(t)
	f.mux.HandleFunc("GET /user", func(w http.ResponseWriter, r *http.Request) {
		if bearer(r) != "ghu_user" {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "Bad credentials"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": 12, "login": "octo", "name": "Octo Cat", "email": "octo@example.com",
			"avatar_url": "https://avatars.example/12",
		})
	})
	a := f.app(t)
	u, err := a.GetUser(context.Background(), "ghu_user")
	if err != nil {
		t.Fatal(err)
	}
	want := User{ID: 12, Login: "octo", Name: "Octo Cat", Email: "octo@example.com", AvatarURL: "https://avatars.example/12"}
	if u != want {
		t.Errorf("user = %+v, want %+v", u, want)
	}
	if _, err := a.GetUser(context.Background(), "ghu_wrong"); err == nil {
		t.Error("bad token: no error")
	}
}

// serveUserInstallations serves /user/installations for token ghu_user,
// two per page, and counts requests.
func serveUserInstallations(f *fakeGitHub, installs []map[string]any) *atomic.Int32 {
	calls := new(atomic.Int32)
	f.mux.HandleFunc("GET /user/installations", func(w http.ResponseWriter, r *http.Request) {
		if bearer(r) != "ghu_user" {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "Bad credentials"})
			return
		}
		calls.Add(1)
		const per = 2
		page := pageParam(r)
		last := (len(installs) + per - 1) / per
		lo, hi := min((page-1)*per, len(installs)), min(page*per, len(installs))
		linkHeader(w, r, page, last)
		writeJSON(w, http.StatusOK, map[string]any{"total_count": len(installs), "installations": installs[lo:hi]})
	})
	return calls
}

func TestUserCanAccessInstallation(t *testing.T) {
	f := newFakeGitHub(t)
	calls := serveUserInstallations(f, []map[string]any{
		{"id": 101, "app_id": testAppID}, {"id": 102, "app_id": testAppID},
		{"id": 103, "app_id": testAppID}, {"id": 104, "app_id": 1},
		{"id": 105, "app_id": testAppID},
	})
	a := f.app(t)
	ctx := context.Background()

	ok, err := a.UserCanAccessInstallation(ctx, "ghu_user", 105)
	if err != nil || !ok {
		t.Fatalf("installation on the last page: %v, %v; want true", ok, err)
	}
	if n := calls.Load(); n != 3 {
		t.Errorf("fetched %d pages, want 3", n)
	}

	calls.Store(0)
	ok, err = a.UserCanAccessInstallation(ctx, "ghu_user", 999)
	if err != nil || ok {
		t.Errorf("unknown installation: %v, %v; want false", ok, err)
	}
	if n := calls.Load(); n != 3 {
		t.Errorf("fetched %d pages for an unknown installation, want all 3", n)
	}

	if ok, err := a.UserCanAccessInstallation(ctx, "ghu_user", 104); err != nil || ok {
		t.Errorf("another App's installation: %v, %v; want false", ok, err)
	}
	if _, err := a.UserCanAccessInstallation(ctx, "ghu_spoofed", 101); err == nil {
		t.Error("bad token: no error")
	}
}

func TestUserIsOrgAdmin(t *testing.T) {
	f := newFakeGitHub(t)
	memberships := map[string]map[string]any{
		"acme":    {"state": "active", "role": "admin"},
		"members": {"state": "active", "role": "member"},
		"pending": {"state": "pending", "role": "admin"},
	}
	var paths []string
	f.mux.HandleFunc("GET /user/memberships/orgs/{org}", func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		switch {
		case bearer(r) == "ghu_noperm":
			writeJSON(w, http.StatusForbidden, map[string]string{"message": "Resource not accessible by integration"})
		case bearer(r) != "ghu_user":
			writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "Bad credentials"})
		case memberships[r.PathValue("org")] != nil:
			writeJSON(w, http.StatusOK, memberships[r.PathValue("org")])
		default:
			writeJSON(w, http.StatusNotFound, map[string]string{"message": "Not Found"})
		}
	})
	a := f.app(t)
	ctx := context.Background()

	for org, want := range map[string]bool{"acme": true, "members": false, "pending": false, "elsewhere": false} {
		got, err := a.UserIsOrgAdmin(ctx, "ghu_user", org)
		if err != nil || got != want {
			t.Errorf("%s: %v, %v; want %v", org, got, err, want)
		}
	}
	if !slices.Contains(paths, "/user/memberships/orgs/acme") {
		t.Errorf("paths %v", paths)
	}
	// No Members permission: an error that says so, not false.
	if _, err := a.UserIsOrgAdmin(ctx, "ghu_noperm", "acme"); err == nil || !strings.Contains(err.Error(), "Members organization permission") {
		t.Errorf("403: err = %v", err)
	}
	if _, err := a.UserIsOrgAdmin(ctx, "ghu_wrong", "acme"); err == nil {
		t.Error("bad token: no error")
	}
	n := len(paths)
	for _, org := range []string{"", "../app", "acme/x", "a?b", strings.Repeat("a", 40)} {
		if _, err := a.UserIsOrgAdmin(ctx, "ghu_user", org); err == nil {
			t.Errorf("org %q: no error", org)
		}
	}
	if _, err := a.UserIsOrgAdmin(ctx, "", "acme"); err == nil {
		t.Error("empty token: no error")
	}
	if len(paths) != n {
		t.Errorf("invalid input reached GitHub: %v", paths[n:])
	}
}

func TestGetInstallation(t *testing.T) {
	f := newFakeGitHub(t)
	f.mux.HandleFunc("GET /app/installations/{id}", func(w http.ResponseWriter, r *http.Request) {
		if _, _, err := f.parseAppJWT(r); err != nil {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "bad jwt"})
			return
		}
		if r.PathValue("id") != "55" {
			writeJSON(w, http.StatusNotFound, map[string]string{"message": "Not Found"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": 55, "app_id": testAppID, "repository_selection": "selected",
			"account":      map[string]any{"id": 9, "login": "acme", "type": "Organization"},
			"suspended_at": "2026-11-01T00:00:00Z",
		})
	})
	a := f.app(t)
	in, err := a.GetInstallation(context.Background(), 55)
	if err != nil {
		t.Fatal(err)
	}
	want := Installation{ID: 55, AccountID: 9, AccountLogin: "acme", AccountType: "Organization", RepositorySelection: "selected", Suspended: true}
	if in != want {
		t.Errorf("installation = %+v, want %+v", in, want)
	}
	if _, err := a.GetInstallation(context.Background(), 56); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing installation: err = %v, want ErrNotFound", err)
	}
}

func TestListInstallationRepos(t *testing.T) {
	f := newFakeGitHub(t)
	var all []map[string]any
	for i := 1; i <= 5; i++ {
		all = append(all, map[string]any{
			"id": 1000 + i, "full_name": fmt.Sprintf("acme/r%d", i), "default_branch": "main", "private": i%2 == 0,
		})
	}
	var pages atomic.Int32
	f.mux.HandleFunc("GET /installation/repositories", func(w http.ResponseWriter, r *http.Request) {
		if !f.authorized(w, r) {
			return
		}
		pages.Add(1)
		const per = 2
		page := pageParam(r)
		last := (len(all) + per - 1) / per
		lo, hi := min((page-1)*per, len(all)), min(page*per, len(all))
		linkHeader(w, r, page, last)
		writeJSON(w, http.StatusOK, map[string]any{"total_count": len(all), "repositories": all[lo:hi]})
	})

	repos, err := f.app(t).ListInstallationRepos(context.Background(), 31)
	if err != nil {
		t.Fatal(err)
	}
	if len(repos) != 5 || pages.Load() != 3 {
		t.Fatalf("got %d repos over %d pages, want 5 over 3", len(repos), pages.Load())
	}
	if want := (Repo{ID: 1002, FullName: "acme/r2", DefaultBranch: "main", Private: true}); repos[1] != want {
		t.Errorf("repos[1] = %+v, want %+v", repos[1], want)
	}

	// The listing token is metadata-only and is revoked afterwards.
	m := f.lastMint(t)
	if want := map[string]any{"permissions": map[string]any{"metadata": "read"}}; !reflect.DeepEqual(m.Body, want) {
		t.Errorf("listing token body = %v, want %v", m.Body, want)
	}
	if got := f.revokedTokens(); !reflect.DeepEqual(got, []string{"ghs_test1"}) {
		t.Errorf("revoked %v, want the listing token", got)
	}
}
