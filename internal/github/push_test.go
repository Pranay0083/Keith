package github

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

const (
	pushTok = "ghs_pushTOKEN0123456789"
	readTok = "ghs_readTOKEN0123456789"
)

// testRemote is a smart-HTTP git server (git http-backend) with one bare
// repo, acme/web.git. pushTok may push; readTok may only fetch, and a
// push with it gets 403, as on GitHub.
type testRemote struct {
	t    *testing.T
	root string
	home string
	srv  *httptest.Server

	mu   sync.Mutex
	reqs []*http.Request
	auth []string // Authorization headers, in order
}

func newTestRemote(t *testing.T) *testRemote {
	t.Helper()
	gitPath, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git is not installed")
	}
	execPath, err := exec.Command(gitPath, "--exec-path").Output()
	if err != nil {
		t.Fatal(err)
	}
	backend := filepath.Join(strings.TrimSpace(string(execPath)), "git-http-backend")
	if _, err := os.Stat(backend); err != nil {
		t.Skip("git-http-backend is not installed")
	}
	r := &testRemote{t: t, root: t.TempDir(), home: t.TempDir()}
	h := &cgi.Handler{
		Path: backend,
		Root: "/git",
		Env: []string{
			"GIT_PROJECT_ROOT=" + r.root, "GIT_HTTP_EXPORT_ALL=1", "REMOTE_USER=x-access-token",
			"GIT_CONFIG_NOSYSTEM=1", "HOME=" + r.home, "XDG_CONFIG_HOME=" + r.home,
		},
		Stderr: io.Discard,
	}
	r.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.mu.Lock()
		r.reqs = append(r.reqs, req)
		r.auth = append(r.auth, req.Header.Get("Authorization"))
		r.mu.Unlock()
		user, pass, _ := req.BasicAuth()
		push := strings.HasSuffix(req.URL.Path, "/git-receive-pack") || req.URL.Query().Get("service") == "git-receive-pack"
		switch {
		case user != "x-access-token" || (pass != pushTok && pass != readTok):
			w.Header().Set("WWW-Authenticate", `Basic realm="git"`)
			http.Error(w, "Authentication failed", http.StatusUnauthorized)
			return
		case push && pass != pushTok:
			http.Error(w, "Permission to acme/web.git denied to keith[bot].", http.StatusForbidden)
			return
		}
		req.Header.Del("Authorization")
		h.ServeHTTP(w, req)
	}))
	t.Cleanup(r.srv.Close)
	r.seed()
	return r
}

func (r *testRemote) bare() string { return filepath.Join(r.root, "acme", "web.git") }

func (r *testRemote) git(dir string, args ...string) string {
	r.t.Helper()
	full := append([]string{"-c", "user.name=T", "-c", "user.email=t@example.test", "-c", "commit.gpgsign=false",
		"-c", "init.defaultBranch=main", "-c", "core.hooksPath=/dev/null"}, args...)
	cmd := exec.Command("git", full...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "HOME="+r.home, "XDG_CONFIG_HOME="+r.home, "GIT_CONFIG_NOSYSTEM=1", "GIT_TERMINAL_PROMPT=0")
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		r.t.Fatalf("git %v: %v: %s", args, err, errb.String())
	}
	return strings.TrimSpace(out.String())
}

// seed creates acme/web.git with main, agent/other and tag v1, and a
// clone of it in r.work.
func (r *testRemote) seed() {
	if err := os.MkdirAll(filepath.Dir(r.bare()), 0o755); err != nil {
		r.t.Fatal(err)
	}
	r.git(r.root, "init", "-q", "--bare", r.bare())
	r.git(r.bare(), "config", "http.receivepack", "true")
	work := r.t.TempDir()
	r.git(work, "init", "-q", ".")
	writeFile(r.t, filepath.Join(work, "README.md"), "# web\n")
	r.git(work, "add", "-A")
	r.git(work, "commit", "-qm", "init")
	r.git(work, "branch", "agent/other")
	r.git(work, "tag", "v1")
	r.git(work, "push", "-q", r.bare(), "main", "agent/other", "v1")
}

// clone returns a fresh clone of the bare repo (origin = the bare path).
func (r *testRemote) clone() string {
	dir := filepath.Join(r.t.TempDir(), "web")
	r.git(r.root, "clone", "-q", r.bare(), dir)
	return dir
}

// refs returns every ref of the bare repo as "name oid" lines.
func (r *testRemote) refs() string {
	return r.git(r.bare(), "for-each-ref", "--format=%(refname) %(objectname)")
}

func (r *testRemote) remote() GitRemote {
	gr, err := ParseGitRemote(r.srv.URL + "/git/")
	if err != nil {
		r.t.Fatal(err)
	}
	return gr
}

// bundle creates a bundle in dir of args (rev-list arguments) and parses
// it.
func (r *testRemote) bundle(dir string, args ...string) Bundle {
	r.t.Helper()
	out := filepath.Join(r.t.TempDir(), "b.bundle")
	r.git(dir, append([]string{"bundle", "create", "-q", out}, args...)...)
	data, err := os.ReadFile(out)
	if err != nil {
		r.t.Fatal(err)
	}
	b, err := ParseBundle(data)
	if err != nil {
		r.t.Fatalf("ParseBundle: %v", err)
	}
	return b
}

func writeFile(t *testing.T, p, s string) {
	t.Helper()
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}

func noToken(t *testing.T, err error) {
	t.Helper()
	if err != nil && (strings.Contains(err.Error(), pushTok) || strings.Contains(err.Error(), readTok)) {
		t.Fatalf("error leaks a token: %v", err)
	}
}

func TestReceivePackPushesOnlyTheGivenRef(t *testing.T) {
	r := newTestRemote(t)
	ctx := context.Background()
	work := r.clone()
	mainOID := r.git(work, "rev-parse", "main")
	r.git(work, "checkout", "-qb", "agent/s1")
	writeFile(t, filepath.Join(work, "fix.txt"), "fixed\n")
	r.git(work, "add", "-A")
	r.git(work, "commit", "-qm", "fix")
	tip := r.git(work, "rev-parse", "HEAD")
	before := r.refs()

	b := r.bundle(work, "refs/heads/agent/s1", "--not", mainOID)
	if len(b.Refs) != 1 || b.Refs[0] != (BundleRef{Name: "refs/heads/agent/s1", OID: tip}) ||
		len(b.Prerequisites) != 1 || b.Prerequisites[0] != mainOID {
		t.Fatalf("bundle = %+v", b)
	}

	rp, err := r.remote().OpenReceivePack(ctx, nil, "acme/web", pushTok)
	if err != nil {
		t.Fatal(err)
	}
	if rp.Ref("refs/heads/main") != mainOID || rp.Ref("refs/heads/agent/s1") != "" || rp.Ref("refs/tags/v1") != mainOID {
		t.Fatalf("refs = %v", rp.refs)
	}
	if err := rp.Update(ctx, RefUpdate{Ref: "refs/heads/agent/s1", New: tip}, b.Pack); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if got, want := r.refs(), before+"\nrefs/heads/agent/s1 "+tip; !sameLines(got, want) {
		t.Fatalf("refs after the push:\n%s\nwant\n%s", got, want)
	}
	if got := r.git(r.bare(), "show", "agent/s1:fix.txt"); got != "fixed" {
		t.Fatalf("fix.txt = %q", got)
	}

	// The token went only in the Authorization header.
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, req := range r.reqs {
		if strings.Contains(req.URL.String(), pushTok) || !strings.HasPrefix(r.auth[i], "Basic ") {
			t.Errorf("request %s %s, auth %q", req.Method, req.URL, r.auth[i])
		}
		if ua := req.Header.Get("User-Agent"); !strings.HasPrefix(ua, "git/") {
			t.Errorf("User-Agent %q", ua)
		}
	}
}

func TestReceivePackLease(t *testing.T) {
	r := newTestRemote(t)
	ctx := context.Background()
	work := r.clone()
	mainOID := r.git(work, "rev-parse", "main")
	r.git(work, "checkout", "-qb", "agent/s1")
	writeFile(t, filepath.Join(work, "a.txt"), "1\n")
	r.git(work, "add", "-A")
	r.git(work, "commit", "-qm", "one")
	first := r.git(work, "rev-parse", "HEAD")
	push := func(old, tip string, pack []byte) error {
		rp, err := r.remote().OpenReceivePack(ctx, nil, "acme/web", pushTok)
		if err != nil {
			t.Fatal(err)
		}
		err = rp.Update(ctx, RefUpdate{Ref: "refs/heads/agent/s1", Old: old, New: tip}, pack)
		noToken(t, err)
		return err
	}
	if err := push("", first, r.bundle(work, "refs/heads/agent/s1", "--not", mainOID).Pack); err != nil {
		t.Fatal(err)
	}

	// The agent rewrote its branch: a non-fast-forward update with the
	// right lease goes through.
	writeFile(t, filepath.Join(work, "a.txt"), "2\n")
	r.git(work, "commit", "-qa", "--amend", "-m", "one, amended")
	second := r.git(work, "rev-parse", "HEAD")
	pack := r.bundle(work, "refs/heads/agent/s1", "--not", mainOID).Pack
	// A stale lease (or "create" when it exists) is refused, and the ref
	// stays.
	for _, old := range []string{"", mainOID} {
		if err := push(old, second, pack); err == nil || !strings.Contains(err.Error(), "refused the update") {
			t.Fatalf("stale lease %q: %v", old, err)
		}
	}
	if got := r.git(r.bare(), "rev-parse", "agent/s1"); got != first {
		t.Fatalf("agent/s1 = %s after refused updates", got)
	}
	if err := push(first, second, pack); err != nil {
		t.Fatalf("force with lease: %v", err)
	}
	if got := r.git(r.bare(), "rev-parse", "agent/s1"); got != second {
		t.Fatalf("agent/s1 = %s, want %s", got, second)
	}

	// A commit the remote already has needs no objects.
	if err := push(second, mainOID, EmptyPack()); err != nil {
		t.Fatalf("empty pack: %v", err)
	}
	// A pack without the new commit is refused by the remote.
	writeFile(t, filepath.Join(work, "a.txt"), "3\n")
	r.git(work, "commit", "-qam", "three")
	third := r.git(work, "rev-parse", "HEAD")
	if err := push(mainOID, third, EmptyPack()); err == nil {
		t.Fatal("an update without its objects succeeded")
	}
	if got := r.git(r.bare(), "rev-parse", "agent/s1"); got != mainOID {
		t.Fatalf("agent/s1 = %s", got)
	}
}

func TestReceivePackRefusals(t *testing.T) {
	r := newTestRemote(t)
	ctx := context.Background()
	gr := r.remote()

	// A read-only token cannot open a push, and no error quotes a token.
	_, err := gr.OpenReceivePack(ctx, nil, "acme/web", readTok)
	if err == nil || !strings.Contains(err.Error(), "HTTP 403: the push token was refused") || !strings.Contains(err.Error(), "Permission to acme/web.git denied") {
		t.Fatalf("read token: %v", err)
	}
	noToken(t, err)
	_, err = gr.OpenReceivePack(ctx, nil, "acme/web", "ghs_wrong")
	if err == nil || !strings.Contains(err.Error(), "HTTP 401") || strings.Contains(err.Error(), "ghs_wrong") {
		t.Fatalf("bad token: %v", err)
	}
	if _, err := gr.OpenReceivePack(ctx, nil, "acme/nope", pushTok); err == nil {
		t.Fatal("opened a push to a missing repo")
	} else {
		noToken(t, err)
	}
	if _, err := gr.OpenReceivePack(ctx, nil, "acme/web", ""); err == nil {
		t.Fatal("opened a push without a token")
	}
	if _, err := gr.OpenReceivePack(ctx, nil, "../etc", pushTok); err == nil {
		t.Fatal("opened a push to a bad repo name")
	}

	rp, err := gr.OpenReceivePack(ctx, nil, "acme/web", pushTok)
	if err != nil {
		t.Fatal(err)
	}
	mainOID := rp.Ref("refs/heads/main")
	before := r.refs()
	for _, u := range []RefUpdate{
		{Ref: "refs/tags/v2", New: mainOID},
		{Ref: "main", New: mainOID},
		{Ref: "refs/heads/a..b", New: mainOID},
		{Ref: "refs/heads/x y", New: mainOID},
		{Ref: "refs/heads/agent/s1", New: ZeroOID},                  // a delete
		{Ref: "refs/heads/agent/s1", New: "HEAD"},                   // not an object ID
		{Ref: "refs/heads/agent/s1", Old: "main", New: mainOID},     // not an object ID
		{Ref: "refs/heads/agent/s1\nrefs/heads/main", New: mainOID}, // a second command
	} {
		if err := rp.Update(ctx, u, EmptyPack()); err == nil {
			t.Errorf("Update(%+v) succeeded", u)
		}
	}
	if err := rp.Update(ctx, RefUpdate{Ref: "refs/heads/agent/s1", New: mainOID}, []byte("not a pack")); err == nil {
		t.Error("Update with a non-pack succeeded")
	}
	if got := r.refs(); got != before {
		t.Fatalf("refs changed:\n%s\nwas\n%s", got, before)
	}
}

func TestReceivePackRedirects(t *testing.T) {
	r := newTestRemote(t)
	ctx := context.Background()
	// Another host must never see the token.
	var mu sync.Mutex
	var leaked []string
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		mu.Lock()
		leaked = append(leaked, req.Header.Get("Authorization"))
		mu.Unlock()
		http.NotFound(w, req)
	}))
	defer other.Close()
	away := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, other.URL+req.URL.Path+"?"+req.URL.RawQuery, http.StatusFound)
	}))
	defer away.Close()
	gr, _ := ParseGitRemote(away.URL + "/git/")
	if _, err := gr.OpenReceivePack(ctx, nil, "acme/web", pushTok); err == nil || !strings.Contains(err.Error(), "another host") {
		t.Fatalf("cross-host redirect: %v", err)
	}
	mu.Lock()
	if len(leaked) != 0 {
		t.Fatalf("the other host got requests: %q", leaked)
	}
	mu.Unlock()

	// A same-host redirect (a renamed repo) is followed, and the push goes
	// to the new location.
	mux := http.NewServeMux()
	mux.HandleFunc("/old/", func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, "/git/"+strings.TrimPrefix(req.URL.Path, "/old/")+"?"+req.URL.RawQuery, http.StatusMovedPermanently)
	})
	mux.Handle("/git/", r.srv.Config.Handler)
	same := httptest.NewServer(mux)
	defer same.Close()
	gr, _ = ParseGitRemote(same.URL + "/old/")
	rp, err := gr.OpenReceivePack(ctx, nil, "acme/web", pushTok)
	if err != nil {
		t.Fatalf("same-host redirect: %v", err)
	}
	if rp.url != same.URL+"/git/acme/web.git" {
		t.Fatalf("push URL = %s", rp.url)
	}
	if err := rp.Update(ctx, RefUpdate{Ref: "refs/heads/agent/s9", New: rp.Ref("refs/heads/main")}, EmptyPack()); err != nil {
		t.Fatalf("Update after a redirect: %v", err)
	}
}

func TestParseBundle(t *testing.T) {
	const a, b = "1111111111111111111111111111111111111111", "2222222222222222222222222222222222222222"
	pack := EmptyPack()
	good := "# v2 git bundle\n-" + a + " base commit\n" + b + " refs/heads/agent/s1\n\n" + string(pack)
	bd, err := ParseBundle([]byte(good))
	if err != nil {
		t.Fatal(err)
	}
	if len(bd.Prerequisites) != 1 || bd.Prerequisites[0] != a || len(bd.Refs) != 1 ||
		bd.Refs[0] != (BundleRef{"refs/heads/agent/s1", b}) || !bytes.Equal(bd.Pack, pack) {
		t.Fatalf("bundle = %+v", bd)
	}
	// Several refs parse; the caller decides what it accepts.
	two := "# v3 git bundle\n@object-format=sha1\n" + b + " refs/heads/agent/s1\n" + a + " refs/heads/main\n\n" + string(pack)
	if bd, err := ParseBundle([]byte(two)); err != nil || len(bd.Refs) != 2 || bd.Refs[1].Name != "refs/heads/main" {
		t.Fatalf("two refs: %+v, %v", bd, err)
	}
	for name, in := range map[string]string{
		"empty":         "",
		"not a bundle":  "PACK....",
		"v1":            "# v1 git bundle\n" + b + " refs/heads/x\n\n" + string(pack),
		"filter":        "# v3 git bundle\n@filter=blob:none\n" + b + " refs/heads/x\n\n" + string(pack),
		"sha256":        "# v3 git bundle\n@object-format=sha256\n" + b + " refs/heads/x\n\n" + string(pack),
		"capability v2": "# v2 git bundle\n@object-format=sha1\n" + b + " refs/heads/x\n\n" + string(pack),
		"short oid":     "# v2 git bundle\n1234 refs/heads/x\n\n" + string(pack),
		"bad prereq":    "# v2 git bundle\n-zz\n" + b + " refs/heads/x\n\n" + string(pack),
		"no ref name":   "# v2 git bundle\n" + b + "\n\n" + string(pack),
		"space in ref":  "# v2 git bundle\n" + b + " refs/heads/x y\n\n" + string(pack),
		"truncated":     "# v2 git bundle\n" + b + " refs/heads/x\n",
		"no pack":       "# v2 git bundle\n" + b + " refs/heads/x\n\n",
		"not a pack":    "# v2 git bundle\n" + b + " refs/heads/x\n\nKCAP" + strings.Repeat("\x00", 40),
		"pack v9":       "# v2 git bundle\n" + b + " refs/heads/x\n\nPACK\x00\x00\x00\x09" + strings.Repeat("\x00", 40),
		"huge header":   "# v2 git bundle\n" + strings.Repeat("-"+a+"\n", maxBundleLines+2) + "\n" + string(pack),
	} {
		if _, err := ParseBundle([]byte(in)); err == nil {
			t.Errorf("%s: parsed", name)
		}
	}
}

func TestReadPushResponse(t *testing.T) {
	pkt := func(s string) string { var b bytes.Buffer; writePkt(&b, s); return b.String() }
	band := func(n byte, s string) string { return pkt(string([]byte{n}) + s) }
	report := pkt("unpack ok\n") + pkt("ok refs/heads/agent/s1\n") + "0000"

	lines, remote, err := readPushResponse(strings.NewReader(band(2, "remote: hello\n")+band(1, report)+"0000"), true)
	if err != nil || checkReport(lines, "refs/heads/agent/s1") != nil || remote != "remote: hello" {
		t.Fatalf("sideband: %q %q %v", lines, remote, err)
	}
	if lines, _, err := readPushResponse(strings.NewReader(report), false); err != nil || checkReport(lines, "refs/heads/agent/s1") != nil {
		t.Fatalf("plain: %q %v", lines, err)
	}
	ng := pkt("unpack ok\n") + pkt("ng refs/heads/agent/s1 protected branch hook declined\n") + "0000"
	lines, _, _ = readPushResponse(strings.NewReader(ng), false)
	if err := checkReport(lines, "refs/heads/agent/s1"); err == nil || !strings.Contains(err.Error(), "protected branch hook declined") {
		t.Fatalf("ng: %v", err)
	}
	bad := pkt("unpack index-pack abnormal exit\n") + pkt("ng refs/heads/agent/s1 unpacker error\n") + "0000"
	lines, _, _ = readPushResponse(strings.NewReader(bad), false)
	if err := checkReport(lines, "refs/heads/agent/s1"); err == nil || !strings.Contains(err.Error(), "could not store") {
		t.Fatalf("unpack: %v", err)
	}
	other := pkt("unpack ok\n") + pkt("ok refs/heads/main\n") + "0000"
	lines, _, _ = readPushResponse(strings.NewReader(other), false)
	if checkReport(lines, "refs/heads/agent/s1") == nil {
		t.Fatal("a report for another ref was accepted")
	}
	if _, _, err := readPushResponse(strings.NewReader(band(3, "fatal: out of disk\n")), true); err == nil || !strings.Contains(err.Error(), "out of disk") {
		t.Fatalf("band 3: %v", err)
	}
	for _, in := range []string{"", "zzzz", "0003", "00ffshort"} {
		if lines, _, err := readPushResponse(strings.NewReader(in), false); err == nil && checkReport(lines, "refs/heads/x") == nil {
			t.Errorf("%q accepted", in)
		}
	}
}

func sameLines(a, b string) bool {
	as, bs := strings.Split(a, "\n"), strings.Split(b, "\n")
	if len(as) != len(bs) {
		return false
	}
	seen := map[string]int{}
	for _, l := range as {
		seen[l]++
	}
	for _, l := range bs {
		if seen[l] == 0 {
			return false
		}
		seen[l]--
	}
	return true
}
