package e2e

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	mrand "math/rand/v2"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/api"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/redistest"
	"github.com/Pranay0083/keith/internal/sandbox/docker"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

const (
	// e2eWebOrigin is WEB_ORIGIN. Nothing serves it: the API only
	// compares Origin headers with it and redirects to it.
	e2eWebOrigin     = "http://web.keith.test"
	e2eSessionSecret = "e2e-session-secret-0123456789abcdef"
	e2eWebhookSecret = "e2e-webhook-secret"
	// defaultE2ERedisDB is the Redis database the test uses; see
	// e2eRedisDB.
	defaultE2ERedisDB = 7
)

// containerLabels are put on every container the e2e test creates
// (sandboxes and the network probe): KEITH_IT_LABEL (key=value) if set,
// else keith.test=e2e.
func containerLabels() map[string]string {
	k, v, ok := strings.Cut(os.Getenv("KEITH_IT_LABEL"), "=")
	if !ok || k == "" {
		return map[string]string{"keith.test": "e2e"}
	}
	return map[string]string{k: v}
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// randID is a random positive GitHub-style ID.
func randID() int64 { return mrand.Int64N(1<<40) + 1 }

// repoRoot is the module root.
func repoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..")
}

var (
	imageOnce sync.Once
	imageTag  string
	imageErr  error
)

// buildTestImage builds sandbox/test.Dockerfile once per test binary,
// tagged keith-sandbox-test:<hash of the Dockerfile> exactly as the
// sandbox package's sbtest.TestImage does, so the image is shared and
// reused across runs.
func buildTestImage() (string, error) {
	imageOnce.Do(func() {
		root := repoRoot()
		df := filepath.Join(root, "sandbox", "test.Dockerfile")
		b, err := os.ReadFile(df)
		if err != nil {
			imageErr = err
			return
		}
		sum := sha256.Sum256(b)
		tag := "keith-sandbox-test:" + hex.EncodeToString(sum[:])[:12]
		if exec.Command("docker", "image", "inspect", tag).Run() == nil {
			imageTag = tag
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
		defer cancel()
		out, err := exec.CommandContext(ctx, "docker", "build", "-q", "-f", df, "-t", tag, filepath.Join(root, "sandbox")).CombinedOutput()
		if err != nil {
			imageErr = fmt.Errorf("docker build: %w\n%s", err, out)
			return
		}
		imageTag = tag
	})
	return imageTag, imageErr
}

// testImage returns the sandbox test image, building it if needed.
func testImage(t testing.TB) string {
	t.Helper()
	tag, err := buildTestImage()
	if err != nil {
		t.Fatal(err)
	}
	return tag
}

// runLabel is the label key newSandboxProvider adds, unique per test.
const runLabel = "keith.e2e.run"

// newSandboxProvider returns the docker provider for e2e sandboxes: the
// test image, 1 CPU and 512 MB, no published ports, sandboxExtraHosts,
// and containerLabels() plus runLabel=<random>. When t ends, containers
// and snapshot images still carrying that runLabel are removed, so a
// worker that fails to delete its sandbox leaves nothing behind. Create
// it before starting a worker, so this cleanup runs after the worker
// has stopped.
func newSandboxProvider(t testing.TB) *docker.Provider {
	t.Helper()
	run := randomHex(6)
	labels := containerLabels()
	labels[runLabel] = run
	t.Cleanup(func() { removeLabeled(t, runLabel+"="+run) })
	return docker.New(docker.Options{
		Image:      testImage(t),
		CPUs:       1,
		MemoryMB:   512,
		Ports:      []int{},
		Labels:     labels,
		ExtraHosts: sandboxExtraHosts,
	})
}

// removeLabeled force-removes the containers, then the images, that
// carry label (key=value).
func removeLabeled(t testing.TB, label string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	for _, k := range []struct {
		what     string
		list, rm []string
	}{
		{"containers", []string{"ps", "-aq"}, []string{"rm", "-f", "-v"}},
		{"images", []string{"images", "-q"}, []string{"rmi", "-f"}},
	} {
		out, err := exec.CommandContext(ctx, "docker", append(k.list, "--filter", "label="+label)...).Output()
		if err != nil {
			t.Logf("e2e cleanup: list %s: %v", k.what, err)
			continue
		}
		ids := slices.Compact(slices.Sorted(slices.Values(strings.Fields(string(out)))))
		if len(ids) == 0 {
			continue
		}
		if out, err := exec.CommandContext(ctx, "docker", append(k.rm, ids...)...).CombinedOutput(); err != nil {
			t.Logf("e2e cleanup: remove %s %v: %v: %s", k.what, ids, err, out)
		} else {
			t.Logf("e2e cleanup: removed leftover %s %v", k.what, ids)
		}
	}
}

// logBuffer collects log output and prints it through t.Log only if
// the test failed, so passing runs stay quiet. It keeps the last
// maxLogBytes.
type logBuffer struct {
	mu  sync.Mutex
	buf []byte
}

const maxLogBytes = 4 << 20

func newLogBuffer(t testing.TB, name string) *logBuffer {
	b := &logBuffer{}
	t.Cleanup(func() {
		if t.Failed() {
			if s := b.String(); s != "" {
				t.Logf("%s:\n%s", name, s)
			}
		}
	})
	return b
}

func (b *logBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
	if over := len(b.buf) - maxLogBytes; over > 0 {
		b.buf = append([]byte("[...]\n"), b.buf[over:]...)
	}
	return len(p), nil
}

func (b *logBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(b.buf)
}

// e2eRedisDB is the Redis database number the e2e test uses:
// KEITH_E2E_REDIS_DB, default 7.
//
// asynq's queue names are fixed ("sessions", "webhooks"), so a worker
// on the database other packages' tests use (REDIS_URL's, normally 0)
// would take the run_session tasks their tests enqueue, and they ours.
// A separate database keeps the queues apart. Pub/sub (the event bus
// and control channels) ignores the database number, but its channels
// are named by session UUID, so sharing them is harmless. Two e2e runs
// on one Redis at once would still share the database.
func e2eRedisDB(t testing.TB) int {
	t.Helper()
	v := os.Getenv("KEITH_E2E_REDIS_DB")
	if v == "" {
		return defaultE2ERedisDB
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		t.Fatalf("KEITH_E2E_REDIS_DB=%q is not a database number", v)
	}
	return n
}

// e2eRedis returns a client on REDIS_URL's server, database e2eRedisDB,
// and that database's URL.
func e2eRedis(t testing.TB) (*redis.Client, string) {
	t.Helper()
	storetest.Skip(t)
	db := e2eRedisDB(t)
	base := redistest.RedisURL()
	opt, err := redis.ParseURL(base)
	if err != nil {
		t.Fatalf("REDIS_URL: %v", err)
	}
	opt.DB = db
	u, err := url.Parse(base)
	if err != nil {
		t.Fatalf("REDIS_URL: %v", err)
	}
	u.Path = "/" + strconv.Itoa(db)
	rdb := redis.NewClient(opt)
	t.Cleanup(func() { _ = rdb.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis database %d at REDIS_URL: %v", db, err)
	}
	return rdb, u.String()
}

// env is a running control plane for the end-to-end test: a scratch
// Postgres, Redis (database e2eRedisDB), the real API server over HTTP,
// a fake GitHub (REST API for the worker's App, plus the sign-in
// interface the API uses) and a git server that only accepts tokens
// the fake GitHub minted with contents permission for the repo. The
// worker is not part of it: the test starts one on Config, Store,
// Redis, App and Sandboxes.
type env struct {
	Config  *config.Config // passes Validate for both api and worker
	Store   *store.Store
	Redis   *redis.Client // database e2eRedisDB
	Queue   *queue.Client
	Bus     *redisbus.Bus
	Control *control.Redis
	Keyring *secrets.Keyring
	Logger  *slog.Logger // buffered; printed if the test fails

	API    *api.Server
	APIURL string // http://127.0.0.1:<port>

	GitHub *fakeGitHubAPI // the REST API the worker's App calls
	App    *github.App    // the worker's App, on GitHub
	SignIn *fakeSignIn    // the API's api.GitHub
	Git    *gitServer     // Config.GitBaseURL
	// Sandboxes is the docker provider for the worker
	// (newSandboxProvider).
	Sandboxes *docker.Provider

	mu       sync.Mutex
	installs map[string]int64 // org ID -> installation ID
}

// newEnv starts an env, torn down when t ends. It skips t unless
// KEITH_IT=1, and needs Docker (the git server probes the network from
// a container; see gitBindHost; and Sandboxes needs the test image).
func newEnv(t *testing.T) *env {
	t.Helper()
	storetest.Skip(t)
	ctx := context.Background()
	e := &env{installs: map[string]int64{}}
	e.Logger = slog.New(slog.NewTextHandler(newLogBuffer(t, "e2e logs"), &slog.HandlerOptions{Level: slog.LevelDebug}))

	// Postgres: a migrated scratch database whose URL goes into Config.
	dbURL := storetest.ScratchDatabase(t)
	st, err := store.Open(ctx, dbURL)
	if err != nil {
		t.Fatalf("open scratch database: %v", err)
	}
	t.Cleanup(st.Close)
	if err := st.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	e.Store = st

	var redisURL string
	e.Redis, redisURL = e2eRedis(t)
	e.Queue = queue.NewClient(e.Redis)
	t.Cleanup(func() { _ = e.Queue.Close() })
	e.Bus = redisbus.New(e.Redis)
	e.Control = control.NewRedis(e.Redis)

	key := make([]byte, config.SecretsKeySize)
	_, _ = rand.Read(key)
	if e.Keyring, err = secrets.NewKeyring(key); err != nil {
		t.Fatal(err)
	}

	e.GitHub = newFakeGitHubAPI(t)
	e.App = newTestApp(t, e.GitHub)
	e.SignIn = newFakeSignIn()
	e.Git = newGitServer(t)
	e.Git.setAllow(func(token string, push bool) bool { return e.GitHub.tokenAllows(token, "contents", push) })
	e.Git.setScope(e.GitHub.tokenCovers)
	e.GitHub.setHeadExists(func(repo, branch string) bool {
		bs, err := e.Git.listBranches(repo)
		return err == nil && slices.Contains(bs, branch)
	})
	_, keyPEM := testAppKey(t)
	e.Sandboxes = newSandboxProvider(t)

	// Delete the session tasks this env enqueued, once the worker (set
	// up after newEnv, so stopped before this runs) is gone.
	t.Cleanup(func() { e.deleteTasks(t) })

	ts := httptest.NewUnstartedServer(nil)
	e.APIURL = "http://" + ts.Listener.Addr().String()
	e.Config = &config.Config{
		DatabaseURL:             dbURL,
		RedisURL:                redisURL,
		AnthropicAPIKey:         "sk-ant-e2e-unused",
		DefaultLLMProvider:      config.LLMProviderAnthropic,
		DefaultModel:            config.DefaultAnthropicModel,
		MaxActiveSessionsPerOrg: config.DefaultMaxActiveSessionsPerOrg,
		SandboxProvider:         config.SandboxProviderDocker,
		GitHubAppID:             e2eAppID,
		GitHubAppClientID:       e2eClientID,
		GitHubAppClientSecret:   e2eClientSecret,
		GitHubWebhookSecret:     e2eWebhookSecret,
		GitHubAppSlug:           e2eAppSlug,
		GitHubAppPrivateKey:     keyPEM,
		PublicURL:               e.APIURL,
		WebOrigin:               e2eWebOrigin,
		SessionSecret:           e2eSessionSecret,
		SecretsKey:              key,
		ListenAddr:              ts.Listener.Addr().String(),
		MaxSessionDuration:      10 * time.Minute,
		MaxSessionTokens:        config.DefaultMaxSessionTokens,
		SandboxIdleTimeout:      config.DefaultSandboxIdleTimeout,
		GitBaseURL:              e.Git.BaseURL(),
		SandboxTemplate:         testImage(t),
		WorkerConcurrency:       2,
	}
	for _, role := range []config.Role{config.RoleAPI, config.RoleWorker} {
		if err := e.Config.Validate(role); err != nil {
			t.Fatalf("e2e config: %v", err)
		}
	}

	e.API, err = api.New(api.Deps{
		Config:  e.Config,
		Store:   st,
		Bus:     e.Bus,
		Queue:   e.Queue,
		Control: e.Control,
		GitHub:  e.SignIn,
		Dedupe:  api.NewRedisDeduper(e.Redis),
		Keyring: e.Keyring,
		Ready:   func(ctx context.Context) error { return e.Redis.Ping(ctx).Err() },
		Logger:  e.Logger.With("component", "api"),
	})
	if err != nil {
		t.Fatalf("api.New: %v", err)
	}
	ts.Config.Handler = e.API.Handler()
	ts.Config.ReadHeaderTimeout = 10 * time.Second
	// Cleanups run last-in first-out: stop taking requests, then close
	// open WebSockets (api.Server.Shutdown's documented order).
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := e.API.Shutdown(ctx); err != nil {
			t.Errorf("api shutdown: %v", err)
		}
	})
	ts.Start()
	t.Cleanup(ts.Close)
	return e
}

// deleteTasks deletes the tasks of this env's sessions from the
// sessions queue (in every state but active), so none outlive the test.
// Tasks of sessions in other databases (other runs) are left alone.
func (e *env) deleteTasks(t testing.TB) {
	insp := asynq.NewInspectorFromRedisClient(e.Redis)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	lists := []func(string, ...asynq.ListOption) ([]*asynq.TaskInfo, error){
		insp.ListPendingTasks, insp.ListScheduledTasks, insp.ListRetryTasks,
		insp.ListArchivedTasks, insp.ListCompletedTasks,
	}
	var ours []string
	for _, list := range lists {
		for page := 1; ; page++ {
			tasks, err := list(queue.QueueSessions, asynq.PageSize(100), asynq.Page(page))
			if errors.Is(err, asynq.ErrQueueNotFound) {
				return
			}
			if err != nil {
				t.Logf("e2e cleanup: list tasks: %v", err)
				break
			}
			for _, ti := range tasks {
				var p struct {
					SessionID string `json:"session_id"`
				}
				if json.Unmarshal(ti.Payload, &p) != nil || p.SessionID == "" {
					continue
				}
				if _, err := e.Store.GetSession(ctx, p.SessionID); err == nil {
					ours = append(ours, ti.ID)
				}
			}
			if len(tasks) < 100 {
				break
			}
		}
	}
	for _, id := range ours {
		if err := insp.DeleteTask(queue.QueueSessions, id); err != nil && !errors.Is(err, asynq.ErrTaskNotFound) {
			t.Logf("e2e cleanup: delete task %s: %v", id, err)
		}
	}
}

// login signs in GitHub user (githubUserID, login) through the API's
// real OAuth flow and returns a client with the keith_session cookie in
// its jar. The client sends Origin: WEB_ORIGIN on state-changing
// requests, as the web app's fetch does, and does not follow redirects.
func (e *env) login(t testing.TB, githubUserID int64, login string) *http.Client {
	t.Helper()
	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	c := &http.Client{
		Jar:           jar,
		Transport:     originTransport{origin: e2eWebOrigin, base: http.DefaultTransport},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		Timeout:       30 * time.Second,
	}
	code := e.SignIn.expectLogin(github.User{ID: githubUserID, Login: login, Name: login, AvatarURL: "https://avatars.e2e.test/" + login})

	// 1. /auth/github/login sets the state cookie and redirects to GitHub.
	resp, err := c.Get(e.APIURL + "/auth/github/login?next=/sessions")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("GET /auth/github/login: status %d, want 302", resp.StatusCode)
	}
	loc, err := url.Parse(resp.Header.Get("Location"))
	if err != nil || !strings.HasPrefix(loc.String(), fakeWebBase+"/login/oauth/authorize") {
		t.Fatalf("GET /auth/github/login: redirect to %q, want the authorize URL", resp.Header.Get("Location"))
	}
	state := loc.Query().Get("state")
	if state == "" {
		t.Fatalf("authorize URL %s has no state", loc)
	}

	// 2. GitHub redirects back with the code and state.
	resp, err = c.Get(e.APIURL + "/auth/github/callback?" + url.Values{"code": {code}, "state": {state}}.Encode())
	if err != nil {
		t.Fatalf("callback: %v", err)
	}
	_ = resp.Body.Close()
	if want := e2eWebOrigin + "/sessions"; resp.StatusCode != http.StatusFound || resp.Header.Get("Location") != want {
		t.Fatalf("GET /auth/github/callback: status %d, Location %q; want 302 to %s", resp.StatusCode, resp.Header.Get("Location"), want)
	}
	u, _ := url.Parse(e.APIURL)
	if !slices.ContainsFunc(jar.Cookies(u), func(c *http.Cookie) bool { return c.Name == "keith_session" && c.Value != "" }) {
		t.Fatal("callback set no keith_session cookie")
	}
	return c
}

// originTransport adds Origin to state-changing requests that lack it.
type originTransport struct {
	origin string
	base   http.RoundTripper
}

func (o originTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	switch r.Method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		if r.Header.Get("Origin") == "" {
			r = r.Clone(r.Context())
			r.Header.Set("Origin", o.origin)
		}
	}
	return o.base.RoundTrip(r)
}

// user returns the Keith user of a GitHub user ID.
func (e *env) user(t testing.TB, githubUserID int64) store.User {
	t.Helper()
	u, err := e.Store.GetUserByGitHubID(context.Background(), githubUserID)
	if err != nil {
		t.Fatalf("user for GitHub ID %d: %v", githubUserID, err)
	}
	return u
}

// org returns the first org (by name) userID belongs to: after login,
// the personal org.
func (e *env) org(t testing.TB, userID string) store.Org {
	t.Helper()
	ms, err := e.Store.ListOrgsForUser(context.Background(), userID)
	if err != nil || len(ms) == 0 {
		t.Fatalf("orgs of user %s: %v (%d)", userID, err, len(ms))
	}
	return ms[0].Org
}

// seedRepo links an installation (one per org, random ID, account =
// the repo's owner) to orgID and adds fullName to it, in the store and
// in the fake GitHub, so tokens can be minted for it. Create its git
// repo with e.Git.seed.
func (e *env) seedRepo(t testing.TB, orgID, fullName, defaultBranch string) store.Repo {
	t.Helper()
	ctx := context.Background()
	owner, _, ok := strings.Cut(fullName, "/")
	if !ok {
		t.Fatalf("seedRepo: %q is not owner/name", fullName)
	}
	e.mu.Lock()
	instID, ok := e.installs[orgID]
	if !ok {
		instID = randID()
		e.installs[orgID] = instID
	}
	e.mu.Unlock()
	if _, err := e.Store.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: orgID, InstallationID: instID, AccountLogin: owner,
	}); err != nil {
		t.Fatalf("seedRepo: upsert installation: %v", err)
	}
	r, err := e.Store.UpsertRepo(ctx, store.UpsertRepoParams{
		InstallationID: instID, GitHubRepoID: randID(), FullName: fullName, DefaultBranch: defaultBranch, Private: true,
	})
	if err != nil {
		t.Fatalf("seedRepo: upsert repo: %v", err)
	}
	e.GitHub.addInstallation(instID, owner, fullName)
	return r
}

// apiJSON sends a request to the API with c and checks the status.
// body is JSON-encoded unless it is nil, a string or []byte (sent
// as-is). It returns the response body.
func (e *env) apiJSON(t testing.TB, c *http.Client, method, path string, body any, want int) []byte {
	t.Helper()
	var rdr io.Reader
	switch b := body.(type) {
	case nil:
	case string:
		rdr = strings.NewReader(b)
	case []byte:
		rdr = bytes.NewReader(b)
	default:
		buf, err := json.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		rdr = bytes.NewReader(buf)
	}
	req, err := http.NewRequest(method, e.APIURL+path, rdr)
	if err != nil {
		t.Fatal(err)
	}
	if rdr != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("%s %s: read body: %v", method, path, err)
	}
	if resp.StatusCode != want {
		t.Fatalf("%s %s: status %d, want %d; body %s", method, path, resp.StatusCode, want, out)
	}
	return out
}

// decodeJSON unmarshals b into a T.
func decodeJSON[T any](t testing.TB, b []byte) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return v
}

// apiSession is GET /api/sessions/{id} (docs/api.md).
type apiSession struct {
	ID              string               `json:"id"`
	OrgID           string               `json:"org_id"`
	Status          events.SessionStatus `json:"status"`
	Prompt          string               `json:"prompt"`
	IssueURL        *string              `json:"issue_url"`
	Provider        string               `json:"provider"`
	Model           string               `json:"model"`
	CancelRequested bool                 `json:"cancel_requested"`
	FinishedAt      *time.Time           `json:"finished_at"`
	Repos           []struct {
		RepoID   string  `json:"repo_id"`
		FullName string  `json:"full_name"`
		Branch   string  `json:"branch"`
		PRNumber *int    `json:"pr_number"`
		PRURL    *string `json:"pr_url"`
	} `json:"repos"`
}

// session reads GET /api/sessions/{id}.
func (e *env) session(t testing.TB, c *http.Client, id string) apiSession {
	t.Helper()
	return decodeJSON[apiSession](t, e.apiJSON(t, c, http.MethodGet, "/api/sessions/"+id, nil, http.StatusOK))
}

// waitStatus polls GET /api/sessions/{id} until the status is one of
// want, and returns the session. It fails t after timeout.
func (e *env) waitStatus(t testing.TB, c *http.Client, id string, timeout time.Duration, want ...events.SessionStatus) apiSession {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		s := e.session(t, c, id)
		if slices.Contains(want, s.Status) {
			return s
		}
		if time.Now().After(deadline) {
			t.Fatalf("session %s: status %s after %s, want one of %v", id, s.Status, timeout, want)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// errWSTimeout is returned by wsReader.next when no event arrives in
// time. The connection stays usable.
var errWSTimeout = errors.New("no event before the timeout")

// wsReader reads a session's event WebSocket. A goroutine reads frames
// into a channel, so a timed-out next leaves the connection usable and
// pings are answered while the test does other work.
type wsReader struct {
	conn *websocket.Conn
	ch   chan events.Event
	done chan struct{}
	err  error // set before done is closed
	once sync.Once
	last int64 // highest stored seq seen by until
}

// events opens GET /api/sessions/{id}/events?after_seq=N with c's
// cookie and Origin: WEB_ORIGIN, and fails t unless the upgrade
// succeeds. The connection is closed when t ends.
func (e *env) events(t testing.TB, c *http.Client, sessionID string, afterSeq int64) *wsReader {
	t.Helper()
	u := "ws" + strings.TrimPrefix(e.APIURL, "http") + "/api/sessions/" + sessionID + "/events?after_seq=" + strconv.FormatInt(afterSeq, 10)
	d := websocket.Dialer{HandshakeTimeout: 10 * time.Second, Jar: c.Jar}
	conn, resp, err := d.Dial(u, http.Header{"Origin": {e2eWebOrigin}})
	if err != nil {
		if resp != nil {
			b, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			t.Fatalf("open events of %s: %v: status %d: %s", sessionID, err, resp.StatusCode, b)
		}
		t.Fatalf("open events of %s: %v", sessionID, err)
	}
	r := &wsReader{conn: conn, ch: make(chan events.Event, 1024), done: make(chan struct{}), last: afterSeq}
	go r.read()
	t.Cleanup(r.close)
	return r
}

func (r *wsReader) read() {
	defer close(r.done)
	for {
		_, b, err := r.conn.ReadMessage()
		if err != nil {
			r.err = err
			return
		}
		var ev events.Event
		if err := json.Unmarshal(b, &ev); err != nil {
			r.err = fmt.Errorf("decode event %s: %w", b, err)
			return
		}
		select {
		case r.ch <- ev:
		case <-time.After(time.Minute):
			r.err = errors.New("events channel full for a minute; is the test reading?")
			return
		}
	}
}

// next returns the next event (message.delta included), errWSTimeout
// if none arrives within timeout, or the connection's error once it
// has closed and every event before it was read.
func (r *wsReader) next(timeout time.Duration) (events.Event, error) {
	t := time.NewTimer(timeout)
	defer t.Stop()
	select {
	case ev := <-r.ch:
		return ev, nil
	case <-r.done:
		select {
		case ev := <-r.ch:
			return ev, nil
		default:
			return events.Event{}, r.err
		}
	case <-t.C:
		return events.Event{}, errWSTimeout
	}
}

// until reads stored events (seq > 0; message.delta and other live-only
// events are skipped) until pred returns true for one, and returns them
// all, the matching one last. It fails t on timeout, on a closed
// connection, and if stored seqs do not strictly increase.
func (r *wsReader) until(t testing.TB, timeout time.Duration, pred func(events.Event) bool) []events.Event {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var got []events.Event
	for {
		left := time.Until(deadline)
		if left <= 0 {
			left = time.Nanosecond
		}
		ev, err := r.next(left)
		if err != nil {
			var types []string
			for _, g := range got {
				types = append(types, strconv.FormatInt(g.Seq, 10)+":"+g.Type)
			}
			t.Fatalf("events: %v; stored events read meanwhile: %v", err, types)
		}
		if ev.Seq <= 0 {
			continue
		}
		if ev.Seq <= r.last {
			t.Fatalf("events: seq %d (%s) after seq %d", ev.Seq, ev.Type, r.last)
		}
		r.last = ev.Seq
		got = append(got, ev)
		if pred(ev) {
			return got
		}
	}
}

// close closes the connection and waits for the reader. It is safe to
// call more than once.
func (r *wsReader) close() {
	r.once.Do(func() {
		_ = r.conn.WriteControl(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(time.Second))
		_ = r.conn.Close()
		<-r.done
	})
}

// isType returns a predicate for until that matches an event type.
func isType(typ string) func(events.Event) bool {
	return func(ev events.Event) bool { return ev.Type == typ }
}

// isStatus returns a predicate for until that matches a session.status
// event with one of statuses.
func isStatus(statuses ...events.SessionStatus) func(events.Event) bool {
	return func(ev events.Event) bool {
		if ev.Type != events.TypeSessionStatus {
			return false
		}
		var p struct {
			Status events.SessionStatus `json:"status"`
		}
		return json.Unmarshal(ev.Payload, &p) == nil && slices.Contains(statuses, p.Status)
	}
}
