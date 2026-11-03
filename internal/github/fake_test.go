package github

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	jwt "github.com/golang-jwt/jwt/v4"
)

const (
	testAppID    = 4242
	testClientID = "Iv1.testclient"
	testSecret   = "test-client-secret"
)

var (
	testKeyOnce sync.Once
	testKey     *rsa.PrivateKey
	testKeyPEM  []byte
)

// rsaKey returns a 2048-bit key generated once per test binary.
func rsaKey(t *testing.T) (*rsa.PrivateKey, []byte) {
	t.Helper()
	testKeyOnce.Do(func() {
		k, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			panic(err)
		}
		testKey = k
		testKeyPEM = pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(k)})
	})
	return testKey, testKeyPEM
}

// fakeClock is a manual Clock. After channels fire when Advance moves
// past their deadline.
type fakeClock struct {
	mu      sync.Mutex
	now     time.Time
	waiters []fakeWaiter
}

type fakeWaiter struct {
	at time.Time
	ch chan time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{now: time.Date(2026, 11, 3, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) After(d time.Duration) <-chan time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := make(chan time.Time, 1)
	if d <= 0 {
		ch <- c.now
		return ch
	}
	c.waiters = append(c.waiters, fakeWaiter{at: c.now.Add(d), ch: ch})
	return ch
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	kept := c.waiters[:0]
	for _, w := range c.waiters {
		if !w.at.After(c.now) {
			w.ch <- c.now
		} else {
			kept = append(kept, w)
		}
	}
	c.waiters = kept
}

// waitForWaiters blocks until at least n After calls are pending.
func (c *fakeClock) waitForWaiters(t *testing.T, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		got := len(c.waiters)
		c.mu.Unlock()
		if got >= n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d clock waiters", n)
}

// mintRequest is one POST /app/installations/{id}/access_tokens.
type mintRequest struct {
	InstallationID int64
	Body           map[string]any
	Claims         jwt.RegisteredClaims
	Header         map[string]any
}

// fakeGitHub is an httptest GitHub API. It verifies App JWTs against
// the test key, mints tokens that expire an hour after the fake clock's
// now, and tracks which tokens are live.
type fakeGitHub struct {
	t     *testing.T
	srv   *httptest.Server
	mux   *http.ServeMux
	clock *fakeClock

	mu       sync.Mutex
	mints    []mintRequest
	revoked  []string
	live     map[string]bool
	minted   int
	mintFail int // when non-zero, mints answer with this status
}

func newFakeGitHub(t *testing.T) *fakeGitHub {
	t.Helper()
	f := &fakeGitHub{t: t, mux: http.NewServeMux(), clock: newFakeClock(), live: map[string]bool{}}
	f.mux.HandleFunc("POST /app/installations/{id}/access_tokens", f.handleMint)
	f.mux.HandleFunc("DELETE /installation/token", f.handleRevoke)
	f.srv = httptest.NewServer(f.mux)
	t.Cleanup(f.srv.Close)
	return f
}

// app builds an App pointed at the fake.
func (f *fakeGitHub) app(t *testing.T, mod ...func(*AppConfig)) *App {
	t.Helper()
	_, keyPEM := rsaKey(t)
	cfg := AppConfig{
		AppID:         testAppID,
		ClientID:      testClientID,
		ClientSecret:  testSecret,
		PrivateKeyPEM: keyPEM,
		RedirectURL:   "https://keith.example/auth/github/callback",
		APIBaseURL:    f.srv.URL + "/",
		WebBaseURL:    f.srv.URL,
		HTTPClient:    f.srv.Client(),
		Clock:         f.clock,
	}
	for _, m := range mod {
		m(&cfg)
	}
	a, err := NewApp(cfg)
	if err != nil {
		t.Fatalf("NewApp: %v", err)
	}
	return a
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func bearer(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// parseAppJWT verifies r's bearer JWT with the test public key.
func (f *fakeGitHub) parseAppJWT(r *http.Request) (*jwt.Token, jwt.RegisteredClaims, error) {
	key, _ := rsaKey(f.t)
	var claims jwt.RegisteredClaims
	p := jwt.NewParser(jwt.WithValidMethods([]string{"RS256"}), jwt.WithoutClaimsValidation())
	tok, err := p.ParseWithClaims(bearer(r), &claims, func(*jwt.Token) (any, error) { return &key.PublicKey, nil })
	return tok, claims, err
}

func (f *fakeGitHub) handleMint(w http.ResponseWriter, r *http.Request) {
	tok, claims, err := f.parseAppJWT(r)
	if err != nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "A JSON web token could not be decoded"})
		return
	}
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	body, _ := io.ReadAll(r.Body)
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"message": "Problems parsing JSON"})
		return
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	f.mints = append(f.mints, mintRequest{InstallationID: id, Body: m, Claims: claims, Header: tok.Header})
	if f.mintFail != 0 {
		writeJSON(w, f.mintFail, map[string]string{"message": "mint failed"})
		return
	}
	f.minted++
	value := fmt.Sprintf("ghs_test%d", f.minted)
	f.live[value] = true
	writeJSON(w, http.StatusCreated, map[string]any{
		"token":      value,
		"expires_at": f.clock.Now().Add(time.Hour).Format(time.RFC3339),
	})
}

func (f *fakeGitHub) handleRevoke(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	tok := bearer(r)
	if !f.live[tok] {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "Bad credentials"})
		return
	}
	delete(f.live, tok)
	f.revoked = append(f.revoked, tok)
	w.WriteHeader(http.StatusNoContent)
}

// authorized reports whether r carries a live minted token, and answers
// 401 if not.
func (f *fakeGitHub) authorized(w http.ResponseWriter, r *http.Request) bool {
	f.mu.Lock()
	ok := f.live[bearer(r)]
	f.mu.Unlock()
	if !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"message": "Bad credentials"})
	}
	return ok
}

func (f *fakeGitHub) mintCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.mints)
}

func (f *fakeGitHub) lastMint(t *testing.T) mintRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.mints) == 0 {
		t.Fatal("no token was minted")
	}
	return f.mints[len(f.mints)-1]
}

func (f *fakeGitHub) revokedTokens() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.revoked...)
}

func (f *fakeGitHub) setMintFail(status int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.mintFail = status
}

// linkHeader writes a Link header for page-based pagination.
func linkHeader(w http.ResponseWriter, r *http.Request, page, last int) {
	if page >= last {
		return
	}
	u := *r.URL
	q := u.Query()
	q.Set("page", strconv.Itoa(page+1))
	u.RawQuery = q.Encode()
	next := "http://" + r.Host + u.String()
	q.Set("page", strconv.Itoa(last))
	u.RawQuery = q.Encode()
	lastURL := "http://" + r.Host + u.String()
	w.Header().Set("Link", fmt.Sprintf(`<%s>; rel="next", <%s>; rel="last"`, next, lastURL))
}

func pageParam(r *http.Request) int {
	p, err := strconv.Atoi(r.URL.Query().Get("page"))
	if err != nil || p < 1 {
		return 1
	}
	return p
}
