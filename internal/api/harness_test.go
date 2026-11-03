package api

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events/memevents"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue/queuetest"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

const (
	testWebOrigin = "https://app.keith.test"
	testPublicURL = "https://api.keith.test"
	testSecret    = "0123456789abcdef0123456789abcdef-session"
	testHookKey   = "webhook-secret"
)

// fakeGitHub is a scripted GitHub. Every method records its call.
type fakeGitHub struct {
	mu        sync.Mutex
	calls     []string
	token     string
	exchErr   error
	user      github.User
	userErr   error
	canAccess bool
	accessErr error
	orgAdmin  bool
	adminErr  error
	inst      github.Installation
	instErr   error
	repos     []github.Repo
	reposErr  error
	slug      string
}

func (f *fakeGitHub) record(c string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, c)
}

func (f *fakeGitHub) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}

func (f *fakeGitHub) AuthCodeURL(state string) (string, error) {
	f.record("AuthCodeURL")
	return "https://github.test/login/oauth/authorize?client_id=cid&redirect_uri=" +
		testPublicURL + "/auth/github/callback&state=" + state, nil
}

func (f *fakeGitHub) Exchange(_ context.Context, code string) (github.UserToken, error) {
	f.record("Exchange:" + code)
	if f.exchErr != nil {
		return github.UserToken{}, f.exchErr
	}
	return github.UserToken{AccessToken: f.token}, nil
}

func (f *fakeGitHub) GetUser(_ context.Context, tok string) (github.User, error) {
	f.record("GetUser:" + tok)
	return f.user, f.userErr
}

func (f *fakeGitHub) UserCanAccessInstallation(_ context.Context, tok string, id int64) (bool, error) {
	f.record("UserCanAccessInstallation")
	return f.canAccess, f.accessErr
}

func (f *fakeGitHub) UserIsOrgAdmin(_ context.Context, tok, org string) (bool, error) {
	f.record("UserIsOrgAdmin:" + org)
	return f.orgAdmin, f.adminErr
}

func (f *fakeGitHub) GetInstallation(_ context.Context, id int64) (github.Installation, error) {
	f.record("GetInstallation")
	if f.instErr != nil {
		return github.Installation{}, f.instErr
	}
	in := f.inst
	in.ID = id
	return in, nil
}

func (f *fakeGitHub) ListInstallationRepos(_ context.Context, id int64) ([]github.Repo, error) {
	f.record("ListInstallationRepos")
	return f.repos, f.reposErr
}

func (f *fakeGitHub) InstallURL(slug, state string) (string, error) {
	f.record("InstallURL")
	return "https://github.test/apps/" + slug + "/installations/new?state=" + state, nil
}

func (f *fakeGitHub) Slug(context.Context) (string, error) {
	f.record("Slug")
	if f.slug == "" {
		return "", errors.New("no slug")
	}
	return f.slug, nil
}

// memDeduper is an in-memory Deduper.
type memDeduper struct {
	mu   sync.Mutex
	seen map[string]bool
	err  error
}

func (d *memDeduper) FirstSeen(_ context.Context, key string, _ time.Duration) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.err != nil {
		return false, d.err
	}
	if d.seen == nil {
		d.seen = map[string]bool{}
	}
	if d.seen[key] {
		return false, nil
	}
	d.seen[key] = true
	return true, nil
}

func (d *memDeduper) Forget(_ context.Context, key string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	delete(d.seen, key)
	return nil
}

type harness struct {
	t      *testing.T
	srv    *Server
	h      http.Handler
	st     *store.Store
	cfg    *config.Config
	q      *queuetest.Recorder
	ctl    *control.Mem
	gh     *fakeGitHub
	bus    *memevents.Bus
	evs    *memevents.Store
	dedupe *memDeduper
	ready  error
}

func testConfig() *config.Config {
	return &config.Config{
		PublicURL:               testPublicURL,
		WebOrigin:               testWebOrigin,
		SessionSecret:           testSecret,
		GitHubWebhookSecret:     testHookKey,
		DefaultLLMProvider:      config.LLMProviderAnthropic,
		DefaultModel:            config.DefaultAnthropicModel,
		AnthropicAPIKey:         "sk-ant",
		MaxActiveSessionsPerOrg: 5,
	}
}

// newHarness builds a Server. st may be nil for tests that never reach
// the database.
func newHarness(t *testing.T, st *store.Store) *harness {
	t.Helper()
	if st == nil {
		st = store.New(nil)
	}
	key := make([]byte, 32)
	_, _ = rand.Read(key)
	kr, err := secrets.NewKeyring(key)
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{
		t: t, st: st, cfg: testConfig(), q: &queuetest.Recorder{}, ctl: control.NewMem(),
		gh:  &fakeGitHub{token: "gho_user", slug: "keith-agent", canAccess: true, orgAdmin: true},
		bus: memevents.NewBus(), evs: memevents.NewStore(), dedupe: &memDeduper{},
	}
	h.srv, err = New(Deps{
		Config: h.cfg, Store: st, Events: h.evs, Bus: h.bus, Queue: h.q, Control: h.ctl,
		GitHub: h.gh, Dedupe: h.dedupe, Keyring: kr,
		Ready:  func(context.Context) error { return h.ready },
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	h.h = h.srv.Handler()
	return h
}

type reqOpt func(*http.Request)

// as signs the request in as userID.
func (h *harness) as(userID string) reqOpt {
	return func(r *http.Request) {
		v, err := h.srv.signer.signSession(userID, time.Now(), time.Now().Add(time.Hour))
		if err != nil {
			h.t.Fatal(err)
		}
		r.AddCookie(&http.Cookie{Name: sessionCookie, Value: v})
	}
}

func withOrigin(o string) reqOpt { return func(r *http.Request) { setOrigin(r, o) } }

func setOrigin(r *http.Request, o string) {
	if o == "" {
		r.Header.Del("Origin")
	} else {
		r.Header.Set("Origin", o)
	}
}

func withHeader(k, v string) reqOpt { return func(r *http.Request) { r.Header.Set(k, v) } }

func withCookie(name, value string) reqOpt {
	return func(r *http.Request) { r.AddCookie(&http.Cookie{Name: name, Value: value}) }
}

// do sends a request. body is JSON-encoded unless it is a string (sent
// as-is). Unsafe methods carry Origin: WEB_ORIGIN unless an option
// changes it.
func (h *harness) do(method, path string, body any, opts ...reqOpt) *httptest.ResponseRecorder {
	h.t.Helper()
	var rdr io.Reader
	switch b := body.(type) {
	case nil:
	case string:
		rdr = strings.NewReader(b)
	default:
		buf, err := json.Marshal(b)
		if err != nil {
			h.t.Fatal(err)
		}
		rdr = bytes.NewReader(buf)
	}
	r := httptest.NewRequest(method, path, rdr)
	if rdr != nil {
		r.Header.Set("Content-Type", "application/json")
	}
	if unsafeMethod(method) {
		r.Header.Set("Origin", testWebOrigin)
	}
	for _, o := range opts {
		o(r)
	}
	w := httptest.NewRecorder()
	h.h.ServeHTTP(w, r)
	return w
}

// wantError checks a docs/api.md error response.
func wantError(t *testing.T, w *httptest.ResponseRecorder, status int, code string) errorDetail {
	t.Helper()
	if w.Code != status {
		t.Fatalf("status = %d, want %d; body %s", w.Code, status, w.Body)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
		t.Errorf("content type %q", ct)
	}
	var b errorBody
	if err := json.Unmarshal(w.Body.Bytes(), &b); err != nil {
		t.Fatalf("error body %s: %v", w.Body, err)
	}
	if b.Error.Code != code || b.Error.Message == "" {
		t.Fatalf("error = %+v, want code %s", b.Error, code)
	}
	return b.Error
}

func decode[T any](t *testing.T, w *httptest.ResponseRecorder, status int) T {
	t.Helper()
	if w.Code != status {
		t.Fatalf("status = %d, want %d; body %s", w.Code, status, w.Body)
	}
	var v T
	if err := json.Unmarshal(w.Body.Bytes(), &v); err != nil {
		t.Fatalf("decode %s: %v", w.Body, err)
	}
	return v
}

// world is a DB-backed fixture: an org with an owner, an admin, a member
// and an outsider (who owns another org), one installation and repos.
type world struct {
	*harness
	org, otherOrg                    store.Org
	owner, admin, member, outsider   store.User
	inst                             store.Installation
	repo, repo2, gone, otherOrgsRepo store.Repo
}

func randID() int64 {
	n, _ := rand.Int(rand.Reader, big.NewInt(1<<40))
	return n.Int64() + 1
}

func newWorld(t *testing.T) *world {
	t.Helper()
	st := storetest.New(t)
	ctx := context.Background()
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	w := &world{harness: newHarness(t, st)}
	user := func(login string) store.User {
		u, err := st.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: randID(), GitHubLogin: login, Name: strings.ToUpper(login), AvatarURL: "https://avatars.test/" + login})
		must(err)
		return u
	}
	w.owner, w.admin, w.member, w.outsider = user("owner"), user("admin"), user("member"), user("outsider")
	var err error
	w.org, err = st.CreateOrg(ctx, "acme", w.owner.ID)
	must(err)
	must(st.AddOrgMember(ctx, w.org.ID, w.admin.ID, store.RoleAdmin))
	must(st.AddOrgMember(ctx, w.org.ID, w.member.ID, store.RoleMember))
	w.otherOrg, err = st.CreateOrg(ctx, "zeta", w.outsider.ID)
	must(err)
	w.inst, err = st.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: w.org.ID, InstallationID: randID(), AccountLogin: "acme"})
	must(err)
	repo := func(inst int64, name string) store.Repo {
		r, err := st.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst, GitHubRepoID: randID(), FullName: name, DefaultBranch: "main", Private: true})
		must(err)
		return r
	}
	w.repo, w.repo2, w.gone = repo(w.inst.InstallationID, "acme/web"), repo(w.inst.InstallationID, "acme/api"), repo(w.inst.InstallationID, "acme/gone")
	_, err = st.SetReposInaccessible(ctx, w.inst.InstallationID, []int64{w.gone.GitHubRepoID})
	must(err)
	otherInst, err := st.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: w.otherOrg.ID, InstallationID: randID(), AccountLogin: "zeta"})
	must(err)
	w.otherOrgsRepo = repo(otherInst.InstallationID, "zeta/app")
	return w
}
