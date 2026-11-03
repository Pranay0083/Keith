package e2e

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	jwt "github.com/golang-jwt/jwt/v4"

	"github.com/Pranay0083/keith/internal/api"
	"github.com/Pranay0083/keith/internal/github"
)

// The test GitHub App.
const (
	e2eAppID        = 314159
	e2eClientID     = "Iv1.e2eclient"
	e2eClientSecret = "e2e-client-secret"
	e2eAppSlug      = "keith-e2e"
)

var (
	appKeyOnce sync.Once
	appKey     *rsa.PrivateKey
	appKeyPEM  []byte
	appKeyErr  error
)

// testAppKey returns the App's RSA key (2048 bits, PKCS#1 PEM),
// generated once per test binary.
func testAppKey(t testing.TB) (*rsa.PrivateKey, []byte) {
	t.Helper()
	appKeyOnce.Do(func() {
		appKey, appKeyErr = rsa.GenerateKey(rand.Reader, 2048)
		if appKeyErr == nil {
			appKeyPEM = pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(appKey)})
		}
	})
	if appKeyErr != nil {
		t.Fatalf("generate App key: %v", appKeyErr)
	}
	return appKey, appKeyPEM
}

// newTestApp returns the worker's *github.App, pointed at f.
func newTestApp(t testing.TB, f *fakeGitHubAPI) *github.App {
	t.Helper()
	_, keyPEM := testAppKey(t)
	app, err := github.NewApp(github.AppConfig{
		AppID:         e2eAppID,
		ClientID:      e2eClientID,
		ClientSecret:  e2eClientSecret,
		PrivateKeyPEM: keyPEM,
		APIBaseURL:    f.URL + "/",
		WebBaseURL:    f.URL,
		HTTPClient:    f.Client(),
	})
	if err != nil {
		t.Fatalf("github.NewApp: %v", err)
	}
	return app
}

// ghMint is one installation token the fake minted.
type ghMint struct {
	InstallationID int64
	// Repositories are owner/name, resolved against the installation's
	// account; empty means the whole installation.
	Repositories []string
	// Permissions as requested, e.g. {"contents": "write"}; empty means
	// everything the installation has, as on GitHub.
	Permissions map[string]string
	Token       string
	ExpiresAt   time.Time
}

type ghToken struct {
	ghMint
	revoked bool
}

// ghIssue is a scripted issue.
type ghIssue struct {
	Title, Body, Author string
	State               string // default "open"
	Labels              []string
	Comments            []ghComment
}

type ghComment struct {
	Author, Body string
	CreatedAt    time.Time
}

// ghPR is a pull request opened through the fake.
type ghPR struct {
	Repo    string // owner/name
	Number  int
	Title   string
	Head    string
	Base    string
	Body    string
	Draft   bool
	State   string
	HTMLURL string
	// Token is the token that opened it.
	Token string
}

// ghCall is one request the fake answered.
type ghCall struct {
	Method, Path string
	Status       int
}

type ghInstall struct {
	account string
	repos   map[string]bool // lower-case owner/name
}

// fakeGitHubAPI is an httptest GitHub REST API for the worker's
// *github.App: installation tokens (App JWT checked against the test
// key), their revocation, issues, and pull requests. Tokens look like
// ghs_e2e1 and carry the permissions and repos they were minted with;
// every other endpoint checks them as GitHub would.
type fakeGitHubAPI struct {
	*httptest.Server
	t testing.TB

	mu         sync.Mutex
	installs   map[int64]*ghInstall
	tokens     map[string]*ghToken
	mintLog    []ghMint
	revokedLog []string
	issues     map[string]map[int]ghIssue // lower-case owner/name -> number
	pulls      []ghPR
	calls      []ghCall
	unexpected []string
	minted     int
	tokenTTL   time.Duration
	headExists func(repo, branch string) bool
}

func newFakeGitHubAPI(t testing.TB) *fakeGitHubAPI {
	t.Helper()
	f := &fakeGitHubAPI{
		t:        t,
		installs: map[int64]*ghInstall{},
		tokens:   map[string]*ghToken{},
		issues:   map[string]map[int]ghIssue{},
		tokenTTL: time.Hour,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /app/installations/{id}/access_tokens", f.handleMint)
	mux.HandleFunc("DELETE /installation/token", f.handleRevoke)
	mux.HandleFunc("GET /repos/{owner}/{repo}/issues/{number}", f.handleIssue)
	mux.HandleFunc("GET /repos/{owner}/{repo}/issues/{number}/comments", f.handleComments)
	mux.HandleFunc("GET /repos/{owner}/{repo}/pulls", f.handleListPulls)
	mux.HandleFunc("POST /repos/{owner}/{repo}/pulls", f.handleCreatePull)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.unexpected = append(f.unexpected, r.Method+" "+r.URL.Path)
		f.mu.Unlock()
		ghError(w, http.StatusNotFound, "Not Found")
	})
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sw := &statusWriter{ResponseWriter: w}
		mux.ServeHTTP(sw, r)
		f.mu.Lock()
		f.calls = append(f.calls, ghCall{Method: r.Method, Path: r.URL.Path, Status: sw.code()})
		f.mu.Unlock()
	}))
	t.Cleanup(f.Close)
	return f
}

// addInstallation registers an installation of the App on account and
// the repos (owner/name) it can access. Calling it again adds repos.
// Tokens for unknown installations answer 404, and tokens for repos the
// installation lacks 422.
func (f *fakeGitHubAPI) addInstallation(id int64, account string, repos ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	in := f.installs[id]
	if in == nil {
		in = &ghInstall{account: account, repos: map[string]bool{}}
		f.installs[id] = in
	}
	for _, r := range repos {
		in.repos[strings.ToLower(r)] = true
	}
}

// addIssue scripts issue number of repo (owner/name).
func (f *fakeGitHubAPI) addIssue(repo string, number int, is ghIssue) {
	f.mu.Lock()
	defer f.mu.Unlock()
	k := strings.ToLower(repo)
	if f.issues[k] == nil {
		f.issues[k] = map[int]ghIssue{}
	}
	f.issues[k][number] = is
}

// setHeadExists sets the check POST /pulls makes on head and base
// branches (422 when it returns false); nil accepts any branch.
func (f *fakeGitHubAPI) setHeadExists(fn func(repo, branch string) bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.headExists = fn
}

// setTokenTTL sets the lifetime of tokens minted from now on (default
// one hour).
func (f *fakeGitHubAPI) setTokenTTL(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tokenTTL = d
}

// mints returns every token minted, in order.
func (f *fakeGitHubAPI) mints() []ghMint {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.mintLog)
}

// revoked returns every token revoked, in order.
func (f *fakeGitHubAPI) revoked() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.revokedLog)
}

// prs returns every pull request opened, in order.
func (f *fakeGitHubAPI) prs() []ghPR {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.pulls)
}

// requests returns every request answered, in order.
func (f *fakeGitHubAPI) requests() []ghCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

// unexpectedCalls returns requests to endpoints the fake does not
// implement.
func (f *fakeGitHubAPI) unexpectedCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.unexpected)
}

// isLive reports whether token was minted, is not revoked and has not
// expired.
func (f *fakeGitHubAPI) isLive(token string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.liveLocked(token) != nil
}

// tokenAllows reports whether token is live and grants perm (contents,
// issues, pull_requests, metadata, ...): read or write when write is
// false, write when it is true.
func (f *fakeGitHubAPI) tokenAllows(token, perm string, write bool) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.allowsLocked(token, perm, write)
}

// tokenCovers reports whether token is live and scoped to repo
// (owner/name).
func (f *fakeGitHubAPI) tokenCovers(token, repo string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.coversLocked(token, repo)
}

func (f *fakeGitHubAPI) liveLocked(token string) *ghToken {
	tk := f.tokens[token]
	if tk == nil || tk.revoked || !time.Now().Before(tk.ExpiresAt) {
		return nil
	}
	return tk
}

func (f *fakeGitHubAPI) allowsLocked(token, perm string, write bool) bool {
	tk := f.liveLocked(token)
	if tk == nil {
		return false
	}
	if len(tk.Permissions) == 0 {
		return true
	}
	switch tk.Permissions[perm] {
	case "write":
		return true
	case "read":
		return !write
	}
	return false
}

func (f *fakeGitHubAPI) coversLocked(token, repo string) bool {
	tk := f.liveLocked(token)
	if tk == nil {
		return false
	}
	in := f.installs[tk.InstallationID]
	if in == nil {
		return false
	}
	owner, _, _ := strings.Cut(repo, "/")
	if !strings.EqualFold(owner, in.account) {
		return false
	}
	if len(tk.Repositories) == 0 {
		return len(in.repos) == 0 || in.repos[strings.ToLower(repo)]
	}
	return slices.ContainsFunc(tk.Repositories, func(r string) bool { return strings.EqualFold(r, repo) })
}

func ghError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"message": msg, "documentation_url": "https://docs.github.com/rest"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func bearer(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// checkAppJWT verifies the request's App JWT: RS256 with the test key,
// issued by the App's client ID (or App ID), unexpired.
func (f *fakeGitHubAPI) checkAppJWT(r *http.Request) error {
	key, _ := testAppKey(f.t)
	var claims jwt.RegisteredClaims
	p := jwt.NewParser(jwt.WithValidMethods([]string{"RS256"}))
	if _, err := p.ParseWithClaims(bearer(r), &claims, func(*jwt.Token) (any, error) { return &key.PublicKey, nil }); err != nil {
		return err
	}
	if claims.Issuer != e2eClientID && claims.Issuer != strconv.Itoa(e2eAppID) {
		return fmt.Errorf("issuer %q", claims.Issuer)
	}
	if claims.ExpiresAt == nil || claims.ExpiresAt.After(time.Now().Add(10*time.Minute)) {
		return errors.New("exp missing or more than 10 minutes ahead")
	}
	return nil
}

// handleMint is POST /app/installations/{id}/access_tokens.
func (f *fakeGitHubAPI) handleMint(w http.ResponseWriter, r *http.Request) {
	if err := f.checkAppJWT(r); err != nil {
		ghError(w, http.StatusUnauthorized, "A JSON web token could not be decoded: "+err.Error())
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		ghError(w, http.StatusNotFound, "Not Found")
		return
	}
	var body struct {
		Repositories  []string          `json:"repositories"`
		RepositoryIDs []int64           `json:"repository_ids"`
		Permissions   map[string]string `json:"permissions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		ghError(w, http.StatusBadRequest, "Problems parsing JSON")
		return
	}
	if len(body.RepositoryIDs) > 0 {
		ghError(w, http.StatusUnprocessableEntity, "the e2e fake does not support repository_ids")
		return
	}
	for k, v := range body.Permissions {
		if v != "read" && v != "write" {
			ghError(w, http.StatusUnprocessableEntity, fmt.Sprintf("invalid permission %s: %q", k, v))
			return
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	in := f.installs[id]
	if in == nil {
		ghError(w, http.StatusNotFound, "Not Found")
		return
	}
	var repos []string
	for _, name := range body.Repositories {
		full := name
		if !strings.Contains(name, "/") {
			full = in.account + "/" + name
		}
		if len(in.repos) > 0 && !in.repos[strings.ToLower(full)] {
			ghError(w, http.StatusUnprocessableEntity,
				"There is at least one repository that does not exist or is not accessible to the parent installation.")
			return
		}
		repos = append(repos, full)
	}
	f.minted++
	m := ghMint{
		InstallationID: id,
		Repositories:   repos,
		Permissions:    maps.Clone(body.Permissions),
		Token:          "ghs_e2e" + strconv.Itoa(f.minted),
		ExpiresAt:      time.Now().Add(f.tokenTTL).Truncate(time.Second),
	}
	f.tokens[m.Token] = &ghToken{ghMint: m}
	f.mintLog = append(f.mintLog, m)
	resp := map[string]any{
		"token":                m.Token,
		"expires_at":           m.ExpiresAt.UTC().Format(time.RFC3339),
		"permissions":          m.Permissions,
		"repository_selection": "selected",
	}
	if len(repos) == 0 {
		resp["repository_selection"] = "all"
	}
	writeJSON(w, http.StatusCreated, resp)
}

// handleRevoke is DELETE /installation/token.
func (f *fakeGitHubAPI) handleRevoke(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	tok := bearer(r)
	tk := f.liveLocked(tok)
	if tk == nil {
		ghError(w, http.StatusUnauthorized, "Bad credentials")
		return
	}
	tk.revoked = true
	f.revokedLog = append(f.revokedLog, tok)
	w.WriteHeader(http.StatusNoContent)
}

// authorize checks the request's token for repo and perm and answers
// 401 (not a live token), 404 (repo outside its scope) or 403 (missing
// permission). It returns the token and holds no lock.
func (f *fakeGitHubAPI) authorize(w http.ResponseWriter, r *http.Request, perm string, write bool) (string, bool) {
	tok := bearer(r)
	repo := r.PathValue("owner") + "/" + r.PathValue("repo")
	f.mu.Lock()
	live := f.liveLocked(tok) != nil
	covers := f.coversLocked(tok, repo)
	allows := f.allowsLocked(tok, perm, write)
	f.mu.Unlock()
	switch {
	case !live:
		ghError(w, http.StatusUnauthorized, "Bad credentials")
	case !covers:
		ghError(w, http.StatusNotFound, "Not Found")
	case !allows:
		ghError(w, http.StatusForbidden, "Resource not accessible by integration")
	default:
		return tok, true
	}
	return "", false
}

func repoOf(r *http.Request) string { return r.PathValue("owner") + "/" + r.PathValue("repo") }

func (f *fakeGitHubAPI) issue(r *http.Request) (ghIssue, int, bool) {
	n, err := strconv.Atoi(r.PathValue("number"))
	if err != nil {
		return ghIssue{}, 0, false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	is, ok := f.issues[strings.ToLower(repoOf(r))][n]
	return is, n, ok
}

// handleIssue is GET /repos/{owner}/{repo}/issues/{number}.
func (f *fakeGitHubAPI) handleIssue(w http.ResponseWriter, r *http.Request) {
	if _, ok := f.authorize(w, r, "issues", false); !ok {
		return
	}
	is, n, ok := f.issue(r)
	if !ok {
		ghError(w, http.StatusNotFound, "Not Found")
		return
	}
	state := is.State
	if state == "" {
		state = "open"
	}
	labels := []map[string]string{}
	for _, l := range is.Labels {
		labels = append(labels, map[string]string{"name": l})
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"number":   n,
		"title":    is.Title,
		"body":     is.Body,
		"state":    state,
		"user":     map[string]string{"login": is.Author},
		"labels":   labels,
		"html_url": "https://github.com/" + repoOf(r) + "/issues/" + strconv.Itoa(n),
		"comments": len(is.Comments),
	})
}

// handleComments is GET /repos/{owner}/{repo}/issues/{number}/comments,
// all on one page.
func (f *fakeGitHubAPI) handleComments(w http.ResponseWriter, r *http.Request) {
	if _, ok := f.authorize(w, r, "issues", false); !ok {
		return
	}
	is, _, ok := f.issue(r)
	if !ok {
		ghError(w, http.StatusNotFound, "Not Found")
		return
	}
	out := []map[string]any{}
	for i, c := range is.Comments {
		at := c.CreatedAt
		if at.IsZero() {
			at = time.Date(2026, 1, 1, 0, i, 0, 0, time.UTC)
		}
		out = append(out, map[string]any{
			"id":         i + 1,
			"user":       map[string]string{"login": c.Author},
			"body":       c.Body,
			"created_at": at.UTC().Format(time.RFC3339),
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func prJSON(p ghPR) map[string]any {
	owner, _, _ := strings.Cut(p.Repo, "/")
	ref := func(branch string) map[string]any {
		return map[string]any{"ref": branch, "label": owner + ":" + branch, "repo": map[string]any{"full_name": p.Repo}}
	}
	return map[string]any{
		"number":   p.Number,
		"state":    p.State,
		"title":    p.Title,
		"body":     p.Body,
		"draft":    p.Draft,
		"html_url": p.HTMLURL,
		"head":     ref(p.Head),
		"base":     ref(p.Base),
	}
}

// handleListPulls is GET /repos/{owner}/{repo}/pulls with the state,
// head (owner:branch) and base filters.
func (f *fakeGitHubAPI) handleListPulls(w http.ResponseWriter, r *http.Request) {
	if _, ok := f.authorize(w, r, "pull_requests", false); !ok {
		return
	}
	q := r.URL.Query()
	state := q.Get("state")
	if state == "" {
		state = "open"
	}
	repo := repoOf(r)
	headOwner, headBranch, hasOwner := strings.Cut(q.Get("head"), ":")
	if !hasOwner {
		headOwner, headBranch = "", q.Get("head")
	}
	f.mu.Lock()
	out := []map[string]any{}
	for _, p := range f.pulls {
		owner, _, _ := strings.Cut(p.Repo, "/")
		switch {
		case !strings.EqualFold(p.Repo, repo),
			state != "all" && p.State != state,
			headBranch != "" && p.Head != headBranch,
			headOwner != "" && !strings.EqualFold(headOwner, owner),
			q.Get("base") != "" && p.Base != q.Get("base"):
			continue
		}
		out = append(out, prJSON(p))
	}
	f.mu.Unlock()
	writeJSON(w, http.StatusOK, out)
}

func validationFailed(w http.ResponseWriter, errs ...map[string]string) {
	writeJSON(w, http.StatusUnprocessableEntity, map[string]any{
		"message": "Validation Failed", "errors": errs,
		"documentation_url": "https://docs.github.com/rest/pulls/pulls#create-a-pull-request",
	})
}

// handleCreatePull is POST /repos/{owner}/{repo}/pulls.
func (f *fakeGitHubAPI) handleCreatePull(w http.ResponseWriter, r *http.Request) {
	tok, ok := f.authorize(w, r, "pull_requests", true)
	if !ok {
		return
	}
	var body struct {
		Title string `json:"title"`
		Head  string `json:"head"`
		Base  string `json:"base"`
		Body  string `json:"body"`
		Draft bool   `json:"draft"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		ghError(w, http.StatusBadRequest, "Problems parsing JSON")
		return
	}
	repo := repoOf(r)
	owner := r.PathValue("owner")
	head := body.Head
	if o, b, ok := strings.Cut(head, ":"); ok {
		if !strings.EqualFold(o, owner) {
			validationFailed(w, map[string]string{"resource": "PullRequest", "field": "head", "code": "invalid"})
			return
		}
		head = b
	}
	if strings.TrimSpace(body.Title) == "" {
		validationFailed(w, map[string]string{"resource": "PullRequest", "field": "title", "code": "missing_field"})
		return
	}
	if head == "" || body.Base == "" || head == body.Base {
		validationFailed(w, map[string]string{"resource": "PullRequest", "field": "head", "code": "invalid"})
		return
	}
	f.mu.Lock()
	exists := f.headExists
	f.mu.Unlock()
	if exists != nil {
		for field, b := range map[string]string{"head": head, "base": body.Base} {
			if !exists(repo, b) {
				validationFailed(w, map[string]string{"resource": "PullRequest", "field": field, "code": "invalid"})
				return
			}
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	next := 1
	for n := range f.issues[strings.ToLower(repo)] {
		next = max(next, n+1)
	}
	for _, p := range f.pulls {
		if !strings.EqualFold(p.Repo, repo) {
			continue
		}
		if p.State == "open" && p.Head == head && p.Base == body.Base {
			validationFailed(w, map[string]string{"resource": "PullRequest", "code": "custom",
				"message": "A pull request already exists for " + owner + ":" + head + "."})
			return
		}
		next = max(next, p.Number+1)
	}
	p := ghPR{
		Repo: repo, Number: next, Title: body.Title, Head: head, Base: body.Base, Body: body.Body,
		Draft: body.Draft, State: "open", Token: tok,
		HTMLURL: "https://github.com/" + repo + "/pull/" + strconv.Itoa(next),
	}
	f.pulls = append(f.pulls, p)
	writeJSON(w, http.StatusCreated, prJSON(p))
}

// fakeSignIn is the api.GitHub the API signs users in with. Each user
// gets an OAuth code from expectLogin; Exchange turns it into a user
// token and GetUser into the user.
type fakeSignIn struct {
	mu       sync.Mutex
	users    map[string]github.User // by code
	installs map[int64]github.Installation
	repos    map[int64][]github.Repo
	calls    []string
}

var _ api.GitHub = (*fakeSignIn)(nil)

func newFakeSignIn() *fakeSignIn {
	return &fakeSignIn{
		users:    map[string]github.User{},
		installs: map[int64]github.Installation{},
		repos:    map[int64][]github.Repo{},
	}
}

// expectLogin registers u and returns the OAuth code that signs u in.
func (f *fakeSignIn) expectLogin(u github.User) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	code := "e2e-code-" + strconv.FormatInt(u.ID, 10)
	f.users[code] = u
	return code
}

// setInstallation scripts GetInstallation and ListInstallationRepos.
func (f *fakeSignIn) setInstallation(in github.Installation, repos []github.Repo) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.installs[in.ID] = in
	f.repos[in.ID] = slices.Clone(repos)
}

func (f *fakeSignIn) record(c string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, c)
}

// Calls returns the methods called, in order.
func (f *fakeSignIn) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

const fakeWebBase = "https://github.e2e.test"

func (f *fakeSignIn) AuthCodeURL(state string) (string, error) {
	f.record("AuthCodeURL")
	return fakeWebBase + "/login/oauth/authorize?" + url.Values{"client_id": {e2eClientID}, "state": {state}}.Encode(), nil
}

func (f *fakeSignIn) Exchange(_ context.Context, code string) (github.UserToken, error) {
	f.record("Exchange")
	f.mu.Lock()
	_, ok := f.users[code]
	f.mu.Unlock()
	if !ok {
		return github.UserToken{}, errors.New("github: oauth exchange: bad_verification_code")
	}
	return github.UserToken{AccessToken: "gho_" + code, TokenType: "bearer"}, nil
}

func (f *fakeSignIn) GetUser(_ context.Context, userToken string) (github.User, error) {
	f.record("GetUser")
	f.mu.Lock()
	defer f.mu.Unlock()
	u, ok := f.users[strings.TrimPrefix(userToken, "gho_")]
	if !ok {
		return github.User{}, fmt.Errorf("github: get user: %w", github.ErrNotFound)
	}
	return u, nil
}

func (f *fakeSignIn) UserCanAccessInstallation(context.Context, string, int64) (bool, error) {
	f.record("UserCanAccessInstallation")
	return true, nil
}

func (f *fakeSignIn) GetInstallation(_ context.Context, id int64) (github.Installation, error) {
	f.record("GetInstallation")
	f.mu.Lock()
	defer f.mu.Unlock()
	in, ok := f.installs[id]
	if !ok {
		return github.Installation{}, fmt.Errorf("github: get installation %d: %w", id, github.ErrNotFound)
	}
	return in, nil
}

func (f *fakeSignIn) ListInstallationRepos(_ context.Context, id int64) ([]github.Repo, error) {
	f.record("ListInstallationRepos")
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.installs[id]; !ok {
		return nil, fmt.Errorf("github: list repos of installation %d: %w", id, github.ErrNotFound)
	}
	return slices.Clone(f.repos[id]), nil
}

func (f *fakeSignIn) InstallURL(appSlug, state string) (string, error) {
	f.record("InstallURL")
	return fakeWebBase + "/apps/" + appSlug + "/installations/new?" + url.Values{"state": {state}}.Encode(), nil
}

func (f *fakeSignIn) Slug(context.Context) (string, error) {
	f.record("Slug")
	return e2eAppSlug, nil
}
