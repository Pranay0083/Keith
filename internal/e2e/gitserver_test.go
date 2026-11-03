package e2e

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// gitServer is a git smart-HTTP server for the end-to-end test: git
// http-backend behind net/http/cgi. Repos are bare repos
// <owner>/<name>.git under a temporary GIT_PROJECT_ROOT, served at
// /git/<owner>/<name>.git with push enabled.
//
// Every request, fetch or push, needs HTTP Basic auth with the username
// x-access-token and a password that allow accepts (and scope, when
// set, accepts for the repo). Anything else gets 401 with
// WWW-Authenticate, as GitHub answers, so git asks its credential
// helper. Every request is recorded.
//
// Sandboxes reach the server at BaseURL, http://host.docker.internal:
// <port>/git/; host-side git uses LocalURL. See gitBindHost for the
// listen address.
type gitServer struct {
	root string // GIT_PROJECT_ROOT
	home string // HOME for host-side git and http-backend
	port int
	srv  *httptest.Server
	cgi  *cgi.Handler
	logs *logBuffer // http-backend's stderr

	mu    sync.Mutex
	allow func(token string, push bool) bool
	scope func(token, repo string) bool
	reqs  []gitRequest
}

// gitRequest is one request the git server saw.
type gitRequest struct {
	Method string
	Path   string // URL path, e.g. /git/acme/web.git/info/refs
	Query  string
	// Repo is owner/name, or "" if the path names no repo.
	Repo string
	// Token is the Basic auth password sent as x-access-token, or "".
	Token string
	// Push is true for receive-pack requests (info/refs advertisement
	// and the pack upload).
	Push bool
	// Chunked is true when the body came with chunked transfer encoding
	// (git does that above http.postBuffer, 1 MiB).
	Chunked bool
	// Status is the response status: 401 when auth failed, else
	// http-backend's.
	Status int
}

// gitUser is the Basic auth username GitHub expects with an
// installation token.
const gitUser = "x-access-token"

// maxGitBody bounds a de-chunked request body.
const maxGitBody = 256 << 20

// newGitServer starts a git server, closed when t ends. Until setAllow
// is called it rejects every request.
func newGitServer(t *testing.T) *gitServer {
	t.Helper()
	host := gitBindHost(t)
	backend := gitHTTPBackend(t)
	gs := &gitServer{
		root: t.TempDir(),
		home: t.TempDir(),
		logs: newLogBuffer(t, "git http-backend stderr"),
	}
	gs.cgi = &cgi.Handler{
		Path: backend,
		Root: "/git",
		Env: []string{
			"GIT_PROJECT_ROOT=" + gs.root,
			"GIT_HTTP_EXPORT_ALL=1",
			"REMOTE_USER=" + gitUser,
			"GIT_CONFIG_NOSYSTEM=1",
			"HOME=" + gs.home,
			"XDG_CONFIG_HOME=" + gs.home,
		},
		Stderr: gs.logs,
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		t.Fatalf("git server: listen on %s: %v", host, err)
	}
	gs.port = ln.Addr().(*net.TCPAddr).Port
	gs.srv = &httptest.Server{
		Listener: ln,
		Config:   &http.Server{Handler: http.HandlerFunc(gs.serveHTTP), ReadHeaderTimeout: 10 * time.Second},
	}
	gs.srv.Start()
	t.Cleanup(gs.srv.Close)
	t.Logf("git server: %s (listening on %s)", gs.BaseURL(), ln.Addr())
	return gs
}

// BaseURL is the clone base for sandboxes (GIT_BASE_URL).
func (gs *gitServer) BaseURL() string {
	return "http://host.docker.internal:" + strconv.Itoa(gs.port) + "/git/"
}

// LocalURL is the clone base for host-side git.
func (gs *gitServer) LocalURL() string {
	return "http://127.0.0.1:" + strconv.Itoa(gs.port) + "/git/"
}

// setAllow sets the token check: allow(token, push) must return true
// for a request to reach http-backend.
func (gs *gitServer) setAllow(allow func(token string, push bool) bool) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.allow = allow
}

// setScope adds a per-repo check (repo is owner/name); nil turns it off.
func (gs *gitServer) setScope(scope func(token, repo string) bool) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.scope = scope
}

// requests returns every request so far, in order.
func (gs *gitServer) requests() []gitRequest {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	return slices.Clone(gs.reqs)
}

func (gs *gitServer) authorized(token, repo string, push bool) bool {
	gs.mu.Lock()
	allow, scope := gs.allow, gs.scope
	gs.mu.Unlock()
	if token == "" || allow == nil || !allow(token, push) {
		return false
	}
	return scope == nil || (repo != "" && scope(token, repo))
}

func (gs *gitServer) record(r gitRequest) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.reqs = append(gs.reqs, r)
}

// parseGitPath returns owner/name from /git/<owner>/<name>.git/...
func parseGitPath(p string) string {
	rest, ok := strings.CutPrefix(p, "/git/")
	if !ok {
		return ""
	}
	owner, rest, ok := strings.Cut(rest, "/")
	if !ok || owner == "" {
		return ""
	}
	name, _, ok := strings.Cut(rest, ".git")
	if !ok || name == "" || strings.Contains(name, "/") {
		return ""
	}
	return owner + "/" + name
}

func (gs *gitServer) serveHTTP(w http.ResponseWriter, r *http.Request) {
	rec := gitRequest{
		Method: r.Method,
		Path:   r.URL.Path,
		Query:  r.URL.RawQuery,
		Repo:   parseGitPath(r.URL.Path),
		Push:   strings.HasSuffix(r.URL.Path, "/git-receive-pack") || r.URL.Query().Get("service") == "git-receive-pack",
		// net/http/cgi refuses chunked bodies: they are buffered below.
		Chunked: len(r.TransferEncoding) > 0 && r.TransferEncoding[0] == "chunked",
	}
	user, pass, ok := r.BasicAuth()
	if ok && user == gitUser {
		rec.Token = pass
	}
	if !strings.HasPrefix(r.URL.Path, "/git/") {
		rec.Status = http.StatusNotFound
		gs.record(rec)
		http.NotFound(w, r)
		return
	}
	if !gs.authorized(rec.Token, rec.Repo, rec.Push) {
		rec.Status = http.StatusUnauthorized
		gs.record(rec)
		w.Header().Set("WWW-Authenticate", `Basic realm="git"`)
		http.Error(w, "Authentication failed", http.StatusUnauthorized)
		return
	}

	if rec.Chunked {
		body, err := io.ReadAll(io.LimitReader(r.Body, maxGitBody+1))
		if err != nil || len(body) > maxGitBody {
			rec.Status = http.StatusRequestEntityTooLarge
			gs.record(rec)
			http.Error(w, "request body too large or unreadable", rec.Status)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
		r.TransferEncoding = nil
	}
	r.Header.Del("Authorization") // REMOTE_USER is set instead
	sw := &statusWriter{ResponseWriter: w}
	gs.cgi.ServeHTTP(sw, r)
	rec.Status = sw.code()
	gs.record(rec)
}

// statusWriter records the response status.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *statusWriter) code() int {
	if w.status == 0 {
		return http.StatusOK
	}
	return w.status
}

// bareDir is the bare repo of owner/name.
func (gs *gitServer) bareDir(fullName string) string {
	owner, name, _ := strings.Cut(fullName, "/")
	return filepath.Join(gs.root, owner, name+".git")
}

// gitEnv is the environment for host-side git: the process environment
// without GIT_* variables, with HOME and XDG_CONFIG_HOME in a temporary
// directory and no system config, so the developer's git configuration
// never leaks in.
func gitEnv(home string) []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, "GIT_") || k == "HOME" || k == "XDG_CONFIG_HOME" {
			continue
		}
		env = append(env, kv)
	}
	return append(env, "HOME="+home, "XDG_CONFIG_HOME="+home, "GIT_CONFIG_NOSYSTEM=1", "GIT_TERMINAL_PROMPT=0")
}

// git runs host-side git in dir and returns its stdout.
func (gs *gitServer) git(dir string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	full := append([]string{"-c", "user.name=Keith E2E", "-c", "user.email=e2e@keith.test",
		"-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"}, args...)
	cmd := exec.CommandContext(ctx, "git", full...)
	cmd.Dir = dir
	cmd.Env = gitEnv(gs.home)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

func (gs *gitServer) mustGit(t testing.TB, dir string, args ...string) string {
	t.Helper()
	out, err := gs.git(dir, args...)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// seed creates the bare repo fullName (owner/name) with one commit of
// files (path -> content) on defaultBranch, and points HEAD at it.
func (gs *gitServer) seed(t testing.TB, fullName, defaultBranch string, files map[string]string) {
	t.Helper()
	if parseGitPath("/git/"+fullName+".git") != fullName {
		t.Fatalf("git server: %q is not owner/name", fullName)
	}
	bare := gs.bareDir(fullName)
	if _, err := os.Stat(bare); err == nil {
		t.Fatalf("git server: %s already exists", fullName)
	}
	if err := os.MkdirAll(filepath.Dir(bare), 0o755); err != nil {
		t.Fatal(err)
	}
	gs.mustGit(t, gs.root, "init", "--quiet", "--bare", "--initial-branch="+defaultBranch, bare)
	gs.mustGit(t, bare, "config", "http.receivepack", "true")

	work := t.TempDir()
	gs.mustGit(t, work, "init", "--quiet", "--initial-branch="+defaultBranch, ".")
	for _, p := range slices.Sorted(maps.Keys(files)) {
		dst := filepath.Join(work, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, []byte(files[p]), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	gs.mustGit(t, work, "add", "-A")
	gs.mustGit(t, work, "commit", "--quiet", "--allow-empty", "-m", "Initial commit")
	gs.mustGit(t, work, "push", "--quiet", bare, "HEAD:refs/heads/"+defaultBranch)
	gs.mustGit(t, bare, "symbolic-ref", "HEAD", "refs/heads/"+defaultBranch)
}

// listBranches returns the repo's branch names, sorted. It is safe to
// call from any goroutine.
func (gs *gitServer) listBranches(fullName string) ([]string, error) {
	bare := gs.bareDir(fullName)
	if _, err := os.Stat(bare); err != nil {
		return nil, err
	}
	out, err := gs.git(bare, "for-each-ref", "--format=%(refname)", "refs/heads/")
	if err != nil {
		return nil, err
	}
	var bs []string
	for _, l := range strings.Split(strings.TrimSpace(out), "\n") {
		if b, ok := strings.CutPrefix(l, "refs/heads/"); ok {
			bs = append(bs, b)
		}
	}
	slices.Sort(bs)
	return bs, nil
}

// branches returns the repo's branch names, sorted.
func (gs *gitServer) branches(t testing.TB, fullName string) []string {
	t.Helper()
	bs, err := gs.listBranches(fullName)
	if err != nil {
		t.Fatalf("git server: branches of %s: %v", fullName, err)
	}
	return bs
}

// show returns the content of path at ref, and false if either does
// not exist.
func (gs *gitServer) show(t testing.TB, fullName, ref, path string) (string, bool) {
	t.Helper()
	out, err := gs.git(gs.bareDir(fullName), "cat-file", "blob", ref+":"+path)
	if err != nil {
		return "", false
	}
	return out, true
}

// log returns the commit subjects reachable from ref, newest first.
func (gs *gitServer) log(t testing.TB, fullName, ref string) []string {
	t.Helper()
	out := gs.mustGit(t, gs.bareDir(fullName), "log", "--format=%s", ref, "--")
	return strings.Split(strings.TrimRight(out, "\n"), "\n")
}

// revParse returns the commit ref points at.
func (gs *gitServer) revParse(t testing.TB, fullName, ref string) string {
	t.Helper()
	return strings.TrimSpace(gs.mustGit(t, gs.bareDir(fullName), "rev-parse", "--verify", ref+"^{commit}"))
}

var (
	backendOnce sync.Once
	backendPath string
	backendErr  error
)

// gitHTTPBackend returns the path of git-http-backend.
func gitHTTPBackend(t testing.TB) string {
	t.Helper()
	backendOnce.Do(func() {
		out, err := exec.Command("git", "--exec-path").Output()
		if err != nil {
			backendErr = fmt.Errorf("git --exec-path: %w", err)
			return
		}
		p := filepath.Join(strings.TrimSpace(string(out)), "git-http-backend")
		if _, err := os.Stat(p); err != nil {
			backendErr = fmt.Errorf("git-http-backend: %w", err)
			return
		}
		backendPath = p
	})
	if backendErr != nil {
		t.Fatalf("git server: %v", backendErr)
	}
	return backendPath
}

// sandboxExtraHosts is docker.Options.ExtraHosts for e2e sandboxes.
//
// Docker Engine on Linux does not define host.docker.internal; the
// host-gateway mapping adds it (the bridge gateway). Docker Desktop
// (macOS, Windows) defines it already; adding host-gateway there is
// harmless: /etc/hosts gets the same 192.168.65.254 plus an IPv6 entry
// that is unreachable inside the container, and git (curl) falls back
// to IPv4 at once. Checked on Docker Desktop 29.7 for Mac.
var sandboxExtraHosts = []string{"host.docker.internal:host-gateway"}

var (
	bindOnce sync.Once
	bindHost string
	bindErr  error
)

// gitBindHost returns the host the git server listens on:
// KEITH_E2E_GIT_BIND if set, else 127.0.0.1 when a probe container
// (with sandboxExtraHosts) reaches a loopback listener through
// host.docker.internal, else 0.0.0.0.
//
// Docker Desktop forwards host.docker.internal to the host's loopback,
// so 127.0.0.1 works there and keeps the server off the network. Docker
// Engine on Linux routes host.docker.internal to the bridge gateway,
// which a loopback listener does not answer; there the server listens
// on all interfaces (it still requires a token on every request).
func gitBindHost(t testing.TB) string {
	t.Helper()
	bindOnce.Do(func() {
		if v := os.Getenv("KEITH_E2E_GIT_BIND"); v != "" {
			bindHost = v
			return
		}
		image, err := buildTestImage()
		if err != nil {
			bindErr = err
			return
		}
		ok, err := probeLoopback(image)
		if err != nil {
			bindErr = err
			return
		}
		bindHost = "0.0.0.0"
		if ok {
			bindHost = "127.0.0.1"
		}
	})
	if bindErr != nil {
		t.Fatalf("git server: choose listen address: %v", bindErr)
	}
	return bindHost
}

// probeLoopback reports whether a container reaches a server on the
// host's 127.0.0.1 at host.docker.internal.
func probeLoopback(image string) (bool, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return false, err
	}
	nonce := "keith-e2e-probe-" + randomHex(8)
	srv := &http.Server{
		Handler:           http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, nonce) }),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	name := "keith-e2e-probe-" + randomHex(6)
	args := []string{"run", "--rm", "--name", name}
	labels := containerLabels()
	for _, k := range slices.Sorted(maps.Keys(labels)) {
		args = append(args, "--label", k+"="+labels[k])
	}
	for _, h := range sandboxExtraHosts {
		args = append(args, "--add-host", h)
	}
	port := strconv.Itoa(ln.Addr().(*net.TCPAddr).Port)
	args = append(args, "--entrypoint", "sh", image, "-c",
		"busybox wget -T 5 -qO- http://host.docker.internal:"+port+"/ || true")
	out, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	if err != nil {
		_ = exec.Command("docker", "rm", "-f", name).Run()
		var ee *exec.ExitError
		if errors.As(err, &ee) || ctx.Err() != nil {
			return false, fmt.Errorf("docker run probe: %w: %s", err, strings.TrimSpace(string(out)))
		}
		return false, fmt.Errorf("docker run probe: %w", err)
	}
	return strings.Contains(string(out), nonce), nil
}
