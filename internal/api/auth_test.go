package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/store"
)

func TestSignedTokens(t *testing.T) {
	s := newSigner([]byte(testSecret))
	now := time.Now()
	tok, err := s.signSession("user-1", now.Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	c, err := s.verifySession(tok, now)
	if err != nil || c.UserID != "user-1" {
		t.Fatalf("verify: %+v %v", c, err)
	}

	// Expired.
	if _, err := s.verifySession(tok, now.Add(2*time.Hour)); !errors.Is(err, errTokenExpired) {
		t.Errorf("expired: %v", err)
	}
	// Tampered body or signature.
	body, sig, _ := strings.Cut(tok, ".")
	forged, _ := s.sign(purposeSession, sessionClaims{claims{now.Add(time.Hour).Unix()}, "user-2"})
	fbody, _, _ := strings.Cut(forged, ".")
	for _, bad := range []string{fbody + "." + sig, body + "." + sig[:len(sig)-2] + "AA", body, "", ".", body + ".", "a.b.c"} {
		if _, err := s.verifySession(bad, now); err == nil {
			t.Errorf("tampered token %q accepted", bad)
		}
	}
	// Another secret.
	if _, err := newSigner([]byte("another-secret-another-secret-123")).verifySession(tok, now); err == nil {
		t.Error("token verified under another secret")
	}
	// Purposes do not mix: a setup state is not a session.
	setup, _ := s.signSetupState("org-1", "user-1", now.Add(time.Hour))
	if _, err := s.verifySession(setup, now); err == nil {
		t.Error("setup state accepted as a session")
	}
	if st, err := s.verifySetupState(setup, now); err != nil || st.OrgID != "org-1" || st.UserID != "user-1" || st.Nonce == "" {
		t.Errorf("setup state %+v %v", st, err)
	}
	oauth, _ := s.signOAuthState("st", "/x", now.Add(time.Minute))
	if _, err := s.verifySetupState(oauth, now); err == nil {
		t.Error("oauth state accepted as setup state")
	}
}

func TestValidNext(t *testing.T) {
	for next, want := range map[string]bool{
		"/sessions": true, "/sessions/abc?tab=diff": true, "/": true,
		"": false, "sessions": false, "//evil.test": false, "/\\evil.test": false,
		"https://evil.test": false, "/a b": false, "/a\nb": false,
	} {
		if got := validNext(next); got != want {
			t.Errorf("validNext(%q) = %v, want %v", next, got, want)
		}
	}
}

func TestCSRF(t *testing.T) {
	h := newHarness(t, nil)
	for _, origin := range []string{"", "https://evil.test", testWebOrigin + ".evil.test", "null"} {
		for _, m := range []string{"POST", "PUT", "DELETE", "PATCH"} {
			w := h.do(m, "/api/orgs/00000000-0000-4000-8000-000000000000/sessions", `{}`, withOrigin(origin))
			wantError(t, w, http.StatusForbidden, codeBadOrigin)
		}
		wantError(t, h.do("POST", "/auth/logout", nil, withOrigin(origin)), http.StatusForbidden, codeBadOrigin)
		// The check runs before auth: a signed-in request is refused the
		// same way, and nothing reaches the database (st is nil).
		wantError(t, h.do("POST", "/api/sessions/x/stop", nil, withOrigin(origin), h.as("u")), http.StatusForbidden, codeBadOrigin)
	}
	// Safe methods need no Origin; without a cookie they are 401.
	wantError(t, h.do("GET", "/api/me", nil), http.StatusUnauthorized, codeUnauthenticated)
	// The webhook is exempt (it fails on its signature instead).
	if w := h.do("POST", "/hooks/github", `{}`, withOrigin("")); w.Code == http.StatusForbidden {
		t.Errorf("webhook got 403: %s", w.Body)
	}
}

func TestCORS(t *testing.T) {
	h := newHarness(t, nil)
	w := h.do("GET", "/api/me", nil, withOrigin(testWebOrigin))
	if w.Header().Get("Access-Control-Allow-Origin") != testWebOrigin ||
		w.Header().Get("Access-Control-Allow-Credentials") != "true" ||
		!strings.Contains(strings.Join(w.Header().Values("Vary"), ","), "Origin") {
		t.Fatalf("CORS headers %v", w.Header())
	}
	w = h.do("GET", "/api/me", nil, withOrigin("https://evil.test"))
	if w.Header().Get("Access-Control-Allow-Origin") != "" || w.Header().Get("Access-Control-Allow-Credentials") != "" {
		t.Fatalf("other origin got CORS headers %v", w.Header())
	}

	// Preflight.
	for _, path := range []string{"/api/sessions/abc/stop", "/api/orgs/x/secrets/NAME", "/auth/logout"} {
		w = h.do("OPTIONS", path, nil, withOrigin(testWebOrigin), withHeader("Access-Control-Request-Method", "POST"))
		if w.Code != http.StatusNoContent || w.Header().Get("Access-Control-Allow-Methods") != "GET, POST, PUT, DELETE" ||
			w.Header().Get("Access-Control-Allow-Headers") != "Content-Type" || w.Header().Get("Access-Control-Max-Age") != "600" ||
			w.Header().Get("Access-Control-Allow-Origin") != testWebOrigin {
			t.Fatalf("preflight %s: %d %v", path, w.Code, w.Header())
		}
		w = h.do("OPTIONS", path, nil, withOrigin("https://evil.test"), withHeader("Access-Control-Request-Method", "POST"))
		if w.Header().Get("Access-Control-Allow-Origin") != "" || w.Header().Get("Access-Control-Allow-Methods") != "" {
			t.Fatalf("preflight from other origin %v", w.Header())
		}
	}
}

func TestCookieAuth(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	valid, _ := h.srv.signer.signSession("00000000-0000-4000-8000-000000000001", now.Add(time.Hour))
	expired, _ := h.srv.signer.signSession("00000000-0000-4000-8000-000000000001", now.Add(-time.Second))
	body, sig, _ := strings.Cut(valid, ".")
	other, _ := h.srv.signer.signSession("00000000-0000-4000-8000-000000000002", now.Add(time.Hour))
	otherBody, _, _ := strings.Cut(other, ".")
	setup, _ := h.srv.signer.signSetupState("o", "00000000-0000-4000-8000-000000000001", now.Add(time.Hour))
	for name, v := range map[string]string{
		"expired": expired, "swapped body": otherBody + "." + sig, "bad sig": body + ".AAAA",
		"garbage": "not-a-token", "setup state": setup, "empty": "",
	} {
		w := h.do("GET", "/api/me", nil, withCookie(sessionCookie, v))
		if w.Code != http.StatusUnauthorized {
			t.Errorf("%s: status %d", name, w.Code)
		}
	}
	// Unknown /api routes are 401 without a cookie too.
	wantError(t, h.do("GET", "/api/nope", nil), http.StatusUnauthorized, codeUnauthenticated)
}

func TestBodyChecks(t *testing.T) {
	h := newHarness(t, nil)
	path := "/api/orgs/00000000-0000-4000-8000-000000000000/sessions"
	wantError(t, h.do("POST", path, `{"prompt":"x"}`, withHeader("Content-Type", "text/plain")),
		http.StatusUnsupportedMediaType, codeUnsupportedMedia)
	wantError(t, h.do("POST", path, `{"prompt":"x"}`, withHeader("Content-Type", "")),
		http.StatusUnsupportedMediaType, codeUnsupportedMedia)
	big := `{"prompt":"` + strings.Repeat("a", maxBody) + `"}`
	wantError(t, h.do("POST", path, big), http.StatusRequestEntityTooLarge, codeTooLarge)
	// application/json with a charset is fine (it then fails auth).
	wantError(t, h.do("POST", path, `{}`, withHeader("Content-Type", "application/json; charset=utf-8")),
		http.StatusUnauthorized, codeUnauthenticated)
}

func TestDecodeJSON(t *testing.T) {
	type body struct {
		Prompt *string  `json:"prompt"`
		IDs    []string `json:"repo_ids"`
	}
	for in, want := range map[string]string{
		`{"prompt":1}`:         codeValidationFailed,
		`{"repo_ids":"x"}`:     codeValidationFailed,
		`{"prompt":`:           codeBadRequest,
		`[1]`:                  codeBadRequest,
		`{"prompt":"a"} {}`:    codeBadRequest,
		`{"prompt":"a","x":1}`: "",
	} {
		r := httptest.NewRequest("POST", "/", strings.NewReader(in))
		var b body
		err := decodeJSON(r, &b)
		var ae *apiError
		switch {
		case want == "" && err != nil:
			t.Errorf("%s: %v", in, err)
		case want != "" && (!errors.As(err, &ae) || ae.Code != want):
			t.Errorf("%s: err %v, want %s", in, err, want)
		}
	}
	r := httptest.NewRequest("POST", "/", strings.NewReader(`{"prompt":1}`))
	var b body
	var ae *apiError
	if errors.As(decodeJSON(r, &b), &ae); ae == nil || ae.Fields["prompt"] == "" {
		t.Errorf("type error fields: %+v", ae)
	}
}

func TestLogout(t *testing.T) {
	h := newHarness(t, nil)
	w := h.do("POST", "/auth/logout", nil)
	if w.Code != http.StatusNoContent {
		t.Fatalf("status %d", w.Code)
	}
	sc := w.Header().Get("Set-Cookie")
	for _, want := range []string{"keith_session=", "Path=/", "Max-Age=0", "HttpOnly", "Secure", "SameSite=Lax"} {
		if !strings.Contains(sc, want) {
			t.Errorf("Set-Cookie %q lacks %q", sc, want)
		}
	}
	if strings.Contains(sc, "Domain") {
		t.Errorf("Set-Cookie has a Domain: %q", sc)
	}
	// With a (bogus) cookie it is still 204.
	if w := h.do("POST", "/auth/logout", nil, withCookie(sessionCookie, "junk")); w.Code != http.StatusNoContent {
		t.Fatalf("status %d", w.Code)
	}
}

func TestLogin(t *testing.T) {
	h := newHarness(t, nil)
	for next, want := range map[string]string{"/sessions/abc": "/sessions/abc", "//evil.test": defaultNext, "": defaultNext} {
		w := h.do("GET", "/auth/github/login?next="+url.QueryEscape(next), nil)
		if w.Code != http.StatusFound {
			t.Fatalf("status %d", w.Code)
		}
		loc, _ := url.Parse(w.Header().Get("Location"))
		state := loc.Query().Get("state")
		if len(state) < 43 { // 32 bytes base64url
			t.Fatalf("state %q too short", state)
		}
		sc := w.Header().Get("Set-Cookie")
		for _, a := range []string{"keith_oauth_state=", "Path=/auth/github", "Max-Age=600", "HttpOnly", "Secure", "SameSite=Lax"} {
			if !strings.Contains(sc, a) {
				t.Errorf("Set-Cookie %q lacks %q", sc, a)
			}
		}
		val := strings.TrimPrefix(strings.SplitN(sc, ";", 2)[0], "keith_oauth_state=")
		c, err := h.srv.signer.verifyOAuthState(val, time.Now())
		if err != nil || c.State != state || c.Next != want {
			t.Errorf("next %q: cookie %+v %v, want next %q", next, c, err, want)
		}
	}
}

// loginState runs /auth/github/login and returns the state and cookie.
func (h *harness) loginState(next string) (state, cookie string) {
	w := h.do("GET", "/auth/github/login?next="+url.QueryEscape(next), nil)
	loc, _ := url.Parse(w.Header().Get("Location"))
	sc := w.Header().Get("Set-Cookie")
	return loc.Query().Get("state"), strings.TrimPrefix(strings.SplitN(sc, ";", 2)[0], "keith_oauth_state=")
}

func wantRedirect(t *testing.T, w *httptest.ResponseRecorder, want string) {
	t.Helper()
	if w.Code != http.StatusFound || w.Header().Get("Location") != want {
		t.Fatalf("got %d %q, want 302 %q", w.Code, w.Header().Get("Location"), want)
	}
}

func TestCallbackFailures(t *testing.T) {
	h := newHarness(t, nil)
	state, cookie := h.loginState("/x")

	// No cookie, wrong state, tampered cookie.
	wantRedirect(t, h.do("GET", "/auth/github/callback?code=c&state="+state, nil), testWebOrigin+"/login?error=oauth_state")
	wantRedirect(t, h.do("GET", "/auth/github/callback?code=c&state=other", nil, withCookie(oauthStateCookie, cookie)),
		testWebOrigin+"/login?error=oauth_state")
	wantRedirect(t, h.do("GET", "/auth/github/callback?code=c&state="+state, nil, withCookie(oauthStateCookie, cookie+"x")),
		testWebOrigin+"/login?error=oauth_state")
	// The callback clears the state cookie.
	w := h.do("GET", "/auth/github/callback?state=x", nil)
	if sc := w.Header().Get("Set-Cookie"); !strings.Contains(sc, "keith_oauth_state=") || !strings.Contains(sc, "Max-Age=0") {
		t.Errorf("state cookie not cleared: %q", sc)
	}
	// Denied on GitHub.
	wantRedirect(t, h.do("GET", "/auth/github/callback?error=access_denied&state="+state, nil, withCookie(oauthStateCookie, cookie)),
		testWebOrigin+"/login?error=oauth_denied")
	// Exchange or user lookup fails.
	h.gh.exchErr = errors.New("bad code")
	wantRedirect(t, h.do("GET", "/auth/github/callback?code=c&state="+state, nil, withCookie(oauthStateCookie, cookie)),
		testWebOrigin+"/login?error=oauth_failed")
	h.gh.exchErr = nil
	h.gh.userErr = errors.New("boom")
	wantRedirect(t, h.do("GET", "/auth/github/callback?code=c&state="+state, nil, withCookie(oauthStateCookie, cookie)),
		testWebOrigin+"/login?error=oauth_failed")
}

func TestHealthRequestIDAndRecover(t *testing.T) {
	h := newHarness(t, nil)
	w := h.do("GET", "/healthz", nil)
	if w.Code != 200 || strings.TrimSpace(w.Body.String()) != `{"ok":true}` {
		t.Fatalf("healthz %d %s", w.Code, w.Body)
	}
	h.ready = errors.New("redis down")
	w = h.do("GET", "/healthz", nil)
	if w.Code != 503 || strings.TrimSpace(w.Body.String()) != `{"ok":false}` {
		t.Fatalf("healthz down %d %s", w.Code, w.Body)
	}

	if id := h.do("GET", "/healthz", nil, withHeader("X-Request-ID", "abc-123")).Header().Get("X-Request-ID"); id != "abc-123" {
		t.Errorf("request id %q", id)
	}
	if id := h.do("GET", "/healthz", nil, withHeader("X-Request-ID", "bad id\n")).Header().Get("X-Request-ID"); id == "" || id == "bad id\n" {
		t.Errorf("bad request id kept: %q", id)
	}

	panicky := h.srv.withRecover(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic("boom") }))
	rec := httptest.NewRecorder()
	panicky.ServeHTTP(rec, httptest.NewRequest("GET", "/x", nil))
	wantError(t, rec, http.StatusInternalServerError, codeInternal)
	if strings.Contains(rec.Body.String(), "boom") {
		t.Error("panic value leaked")
	}
}

func TestProviders(t *testing.T) {
	cases := []struct {
		name    string
		cfg     func(*config.Config)
		want    string
		wantDef string
	}{
		{"anthropic only", func(c *config.Config) {}, `[{"id":"anthropic","default_model":"claude-opus-5-5"}]`, "anthropic"},
		{"both", func(c *config.Config) { c.GeminiAPIKey = "g"; c.DefaultGeminiModel = "gemini-3-pro" },
			`[{"id":"anthropic","default_model":"claude-opus-5-5"},{"id":"gemini","default_model":"gemini-3-pro"}]`, "anthropic"},
		{"gemini without default model", func(c *config.Config) { c.AnthropicAPIKey = ""; c.GeminiAPIKey = "g" },
			`[{"id":"gemini","default_model":""}]`, "gemini"},
		{"none", func(c *config.Config) { c.AnthropicAPIKey = "" }, `[]`, "anthropic"},
	}
	for _, tc := range cases {
		h := newHarness(t, nil)
		tc.cfg(h.cfg)
		rec := httptest.NewRecorder()
		if err := h.srv.handleProviders(rec, httptest.NewRequest("GET", "/api/providers", nil)); err != nil {
			t.Fatal(err)
		}
		var got struct {
			DefaultProvider string          `json:"default_provider"`
			Providers       json.RawMessage `json:"providers"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if string(got.Providers) != tc.want || got.DefaultProvider != tc.wantDef {
			t.Errorf("%s: %s", tc.name, rec.Body)
		}
	}
}

func TestNewChecksDeps(t *testing.T) {
	h := newHarness(t, nil)
	base := Deps{Config: testConfig(), Store: h.st, Bus: h.bus, Queue: h.q, Control: h.ctl, GitHub: h.gh,
		Dedupe: h.dedupe, Keyring: h.srv.keyring, Logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	if _, err := New(base); err != nil {
		t.Fatal(err)
	}
	short := base
	short.Config = testConfig()
	short.Config.SessionSecret = "short"
	noHook := base
	noHook.Config = testConfig()
	noHook.Config.GitHubWebhookSecret = ""
	noGH := base
	noGH.GitHub = nil
	for name, d := range map[string]Deps{"short secret": short, "no webhook secret": noHook, "no github": noGH} {
		if _, err := New(d); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	// Plain-http PUBLIC_URL drops Secure for local development.
	local := base
	local.Config = testConfig()
	local.Config.PublicURL = "http://localhost:8080"
	s, _ := New(local)
	w := httptest.NewRecorder()
	s.setCookie(w, "c", "v", "/", 10)
	if strings.Contains(w.Header().Get("Set-Cookie"), "Secure") {
		t.Error("Secure set for http PUBLIC_URL")
	}
}

// --- Database-backed (KEITH_IT=1) ---

func TestCallbackSignsInIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	w.gh.user = github.User{ID: 987654321, Login: "newbie", Name: "New Bie", AvatarURL: "https://a.test/n"}
	state, cookie := w.loginState("/sessions/xyz")
	resp := w.do("GET", "/auth/github/callback?code=the-code&state="+state, nil, withCookie(oauthStateCookie, cookie))
	wantRedirect(t, resp, testWebOrigin+"/sessions/xyz")
	var session string
	for _, c := range resp.Result().Cookies() {
		if c.Name == sessionCookie {
			session = c.Value
			if c.MaxAge != 2592000 || !c.HttpOnly || !c.Secure || c.SameSite != http.SameSiteLaxMode || c.Path != "/" || c.Domain != "" {
				t.Errorf("session cookie %+v", c)
			}
		}
	}
	if session == "" {
		t.Fatal("no session cookie")
	}
	if calls := w.gh.Calls(); !strings.Contains(strings.Join(calls, ","), "Exchange:the-code") {
		t.Errorf("calls %v", calls)
	}
	u, err := w.st.GetUserByGitHubID(ctx, 987654321)
	if err != nil || u.GitHubLogin != "newbie" {
		t.Fatalf("user %+v %v", u, err)
	}
	// The personal org exists, and /api/me shows it.
	me := decode[meResponse](t, w.do("GET", "/api/me", nil, withCookie(sessionCookie, session)), 200)
	if me.User.ID != u.ID || me.User.Login != "newbie" || me.User.Name != "New Bie" || len(me.Orgs) != 1 ||
		me.Orgs[0].Name != "newbie" || me.Orgs[0].Role != store.RoleOwner {
		t.Fatalf("me %+v", me)
	}
	// Signing in again creates no second org.
	state, cookie = w.loginState("/")
	w.do("GET", "/auth/github/callback?code=c2&state="+state, nil, withCookie(oauthStateCookie, cookie))
	if orgs, _ := w.st.ListOrgsForUser(ctx, u.ID); len(orgs) != 1 {
		t.Fatalf("orgs after second sign-in: %d", len(orgs))
	}
}

func TestMeIT(t *testing.T) {
	w := newWorld(t)
	me := decode[meResponse](t, w.do("GET", "/api/me", nil, w.as(w.outsider.ID)), 200)
	if len(me.Orgs) != 1 || me.Orgs[0].ID != w.otherOrg.ID {
		t.Fatalf("me %+v", me)
	}
	// Sorted by name.
	if err := w.st.AddOrgMember(context.Background(), w.org.ID, w.outsider.ID, store.RoleMember); err != nil {
		t.Fatal(err)
	}
	me = decode[meResponse](t, w.do("GET", "/api/me", nil, w.as(w.outsider.ID)), 200)
	if len(me.Orgs) != 2 || me.Orgs[0].Name != "acme" || me.Orgs[1].Name != "zeta" || me.Orgs[0].Role != store.RoleMember {
		t.Fatalf("me %+v", me)
	}
	// A cookie for a deleted user is 401.
	wantError(t, w.do("GET", "/api/me", nil, w.as("00000000-0000-4000-8000-00000000dead")), 401, codeUnauthenticated)
	// Unknown routes with a cookie are 404.
	wantError(t, w.do("GET", "/api/nope", nil, w.as(w.owner.ID)), 404, codeNotFound)
	// Providers needs a cookie and then answers.
	if r := w.do("GET", "/api/providers", nil, w.as(w.member.ID)); r.Code != 200 {
		t.Fatalf("providers %d", r.Code)
	}
}

func TestInstallationsAndInstallURLIT(t *testing.T) {
	w := newWorld(t)
	base := "/api/orgs/" + w.org.ID + "/github/"

	got := decode[[]map[string]any](t, w.do("GET", base+"installations", nil, w.as(w.member.ID)), 200)
	if len(got) != 1 || got[0]["account_login"] != "acme" || got[0]["status"] != "active" ||
		int64(got[0]["installation_id"].(float64)) != w.inst.InstallationID {
		t.Fatalf("installations %+v", got)
	}
	if _, err := time.Parse(time.RFC3339, got[0]["created_at"].(string)); err != nil || !strings.HasSuffix(got[0]["created_at"].(string), "Z") {
		t.Errorf("created_at %v", got[0]["created_at"])
	}
	wantError(t, w.do("GET", base+"installations", nil, w.as(w.outsider.ID)), 403, codeNotAMember)
	wantError(t, w.do("GET", "/api/orgs/00000000-0000-4000-8000-000000000000/github/installations", nil, w.as(w.owner.ID)), 403, codeNotAMember)
	wantError(t, w.do("GET", "/api/orgs/not-a-uuid/github/installations", nil, w.as(w.owner.ID)), 400, codeBadRequest)

	// install-url: owner or admin.
	wantError(t, w.do("GET", base+"install-url", nil, w.as(w.member.ID)), 403, codeForbiddenRole)
	wantError(t, w.do("GET", base+"install-url", nil, w.as(w.outsider.ID)), 403, codeNotAMember)
	for _, u := range []store.User{w.owner, w.admin} {
		resp := decode[map[string]string](t, w.do("GET", base+"install-url", nil, w.as(u.ID)), 200)
		loc, err := url.Parse(resp["url"])
		if err != nil || !strings.HasPrefix(resp["url"], "https://github.test/apps/keith-agent/installations/new?") {
			t.Fatalf("url %q", resp["url"])
		}
		st, err := w.srv.signer.verifySetupState(loc.Query().Get("state"), time.Now())
		if err != nil || st.OrgID != w.org.ID || st.UserID != u.ID {
			t.Fatalf("state %+v %v", st, err)
		}
		if exp := time.Unix(st.Exp, 0); exp.Before(time.Now().Add(14*time.Minute)) || exp.After(time.Now().Add(16*time.Minute)) {
			t.Errorf("state expiry %v", exp)
		}
	}
}

func TestSetupIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	newInst := randID()
	state := func(orgID, userID string, ttl time.Duration) string {
		s, err := w.srv.signer.signSetupState(orgID, userID, time.Now().Add(ttl))
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	setupURL := func(state string, extra string) string {
		return "/github/setup?installation_id=" + itoa64(newInst) + "&setup_action=install&code=c0de&state=" + url.QueryEscape(state) + extra
	}
	connectErr := func(code string) string { return testWebOrigin + "/connect?error=" + code }
	linked := func() bool {
		_, err := w.st.GetInstallation(ctx, newInst)
		return err == nil
	}
	w.gh.user = github.User{ID: w.admin.GitHubUserID, Login: "admin"}
	w.gh.inst = github.Installation{AccountID: randID(), AccountLogin: "acme-two", AccountType: github.AccountTypeOrganization}
	w.gh.repos = []github.Repo{{ID: randID(), FullName: "acme-two/svc", DefaultBranch: "trunk", Private: true}}
	good := state(w.org.ID, w.admin.ID, time.Hour)

	// No session: to login, coming back to /connect.
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil), testWebOrigin+"/login?next=%2Fconnect")
	// Bad, expired, or someone else's state.
	wantRedirect(t, w.do("GET", setupURL("junk", ""), nil, w.as(w.admin.ID)), connectErr("setup_state"))
	wantRedirect(t, w.do("GET", setupURL(state(w.org.ID, w.admin.ID, -time.Minute), ""), nil, w.as(w.admin.ID)), connectErr("setup_state"))
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil, w.as(w.owner.ID)), connectErr("setup_state"))
	// A member who is no longer admin.
	wantRedirect(t, w.do("GET", setupURL(state(w.org.ID, w.member.ID, time.Hour), ""), nil, w.as(w.member.ID)), connectErr("setup_state"))
	// setup_action=request: pending approval.
	wantRedirect(t, w.do("GET", "/github/setup?setup_action=request&state="+url.QueryEscape(good), nil, w.as(w.admin.ID)),
		testWebOrigin+"/connect?pending=1")

	// Spoofs: the token belongs to another GitHub user, or the user
	// cannot see the installation. Nothing is linked.
	w.gh.user.ID = w.admin.GitHubUserID + 1
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil, w.as(w.admin.ID)), connectErr("installation_not_verified"))
	w.gh.user.ID = w.admin.GitHubUserID
	w.gh.canAccess = false
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil, w.as(w.admin.ID)), connectErr("installation_not_verified"))
	if linked() {
		t.Fatal("spoofed installation linked")
	}
	if strings.Contains(strings.Join(w.gh.Calls(), ","), "GetInstallation") {
		t.Fatal("GetInstallation called before verification")
	}
	w.gh.canAccess = true
	// Missing code, exchange failure.
	wantRedirect(t, w.do("GET", strings.Replace(setupURL(good, ""), "&code=c0de", "", 1), nil, w.as(w.admin.ID)), connectErr("installation_not_verified"))
	w.gh.exchErr = errors.New("bad_verification_code")
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil, w.as(w.admin.ID)), connectErr("setup_failed"))
	w.gh.exchErr = nil
	if linked() {
		t.Fatal("linked after failures")
	}

	// Success.
	wantRedirect(t, w.do("GET", setupURL(good, ""), nil, w.as(w.admin.ID)), testWebOrigin+"/repos?installed=1")
	inst, err := w.st.GetInstallation(ctx, newInst)
	if err != nil || inst.OrgID != w.org.ID || inst.AccountLogin != "acme-two" || inst.Status != store.InstallationActive {
		t.Fatalf("installation %+v %v", inst, err)
	}
	repos, _ := w.st.ListRepos(ctx, w.org.ID, false)
	found := false
	for _, r := range repos {
		if r.FullName == "acme-two/svc" && r.DefaultBranch == "trunk" && r.InstallationID == newInst {
			found = true
		}
	}
	if !found {
		t.Fatalf("synced repos %+v", repos)
	}

	// The same installation for another org: installation_other_org.
	w.gh.user = github.User{ID: w.outsider.GitHubUserID, Login: "outsider"}
	wantRedirect(t, w.do("GET", setupURL(state(w.otherOrg.ID, w.outsider.ID, time.Hour), ""), nil, w.as(w.outsider.ID)),
		connectErr("installation_other_org"))

	// The install redirect may land on the OAuth callback instead.
	w.gh.user = github.User{ID: w.admin.GitHubUserID, Login: "admin"}
	resp := w.do("GET", "/auth/github/callback?installation_id="+itoa64(newInst)+"&setup_action=update&code=c&state="+url.QueryEscape(good), nil, w.as(w.admin.ID))
	wantRedirect(t, resp, testWebOrigin+"/repos?installed=1")
}

// Seeing an installation is not owning it: GitHub lists an installation
// in /user/installations for anyone with read access to one of its
// repos. Setup links it only for the account itself (a user
// installation) or an active admin of the organization.
func TestSetupRequiresOwnershipIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	w.gh.user = github.User{ID: w.admin.GitHubUserID, Login: "admin"}
	w.gh.canAccess = true // the admin can see every installation below
	w.gh.repos = []github.Repo{{ID: randID(), FullName: "victim/secret", DefaultBranch: "main", Private: true}}
	good, err := w.srv.signer.signSetupState(w.org.ID, w.admin.ID, time.Now().Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	setup := func(instID int64) *httptest.ResponseRecorder {
		return w.do("GET", "/github/setup?installation_id="+itoa64(instID)+"&setup_action=install&code=c0de&state="+url.QueryEscape(good), nil, w.as(w.admin.ID))
	}
	connectErr := func(code string) string { return testWebOrigin + "/connect?error=" + code }
	linked := func(instID int64) bool {
		_, err := w.st.GetInstallation(ctx, instID)
		return err == nil
	}

	for _, tc := range []struct {
		name     string
		inst     github.Installation
		orgAdmin bool
		adminErr error
		want     string
	}{
		{"org installation, member but not admin", github.Installation{AccountID: randID(), AccountLogin: "victim", AccountType: github.AccountTypeOrganization}, false, nil, "installation_not_verified"},
		{"org membership lookup fails", github.Installation{AccountID: randID(), AccountLogin: "victim", AccountType: github.AccountTypeOrganization}, true, errors.New("403"), "setup_failed"},
		{"another user's installation", github.Installation{AccountID: w.admin.GitHubUserID + 1, AccountLogin: "victim", AccountType: github.AccountTypeUser}, true, nil, "installation_not_verified"},
		{"enterprise installation", github.Installation{AccountID: randID(), AccountLogin: "victim", AccountType: "Enterprise"}, true, nil, "installation_not_verified"},
	} {
		w.gh.inst, w.gh.orgAdmin, w.gh.adminErr = tc.inst, tc.orgAdmin, tc.adminErr
		instID := randID()
		wantRedirect(t, setup(instID), connectErr(tc.want))
		if linked(instID) {
			t.Fatalf("%s: linked", tc.name)
		}
	}
	if calls := strings.Join(w.gh.Calls(), ","); !strings.Contains(calls, "UserIsOrgAdmin:victim") {
		t.Errorf("membership not checked against the installation's account: %v", calls)
	}
	w.gh.adminErr = nil

	// The user's own account, and an org the user administers: linked.
	w.gh.inst = github.Installation{AccountID: w.admin.GitHubUserID, AccountLogin: "admin", AccountType: github.AccountTypeUser}
	w.gh.orgAdmin = false // not consulted for a user installation
	w.gh.repos = []github.Repo{{ID: randID(), FullName: "admin/dotfiles", DefaultBranch: "main"}}
	own := randID()
	wantRedirect(t, setup(own), testWebOrigin+"/repos?installed=1")
	if !linked(own) {
		t.Fatal("own user installation not linked")
	}
	w.gh.inst = github.Installation{AccountID: randID(), AccountLogin: "acme-three", AccountType: github.AccountTypeOrganization}
	w.gh.orgAdmin = true
	w.gh.repos = []github.Repo{{ID: randID(), FullName: "acme-three/app", DefaultBranch: "main", Private: true}}
	admin := randID()
	wantRedirect(t, setup(admin), testWebOrigin+"/repos?installed=1")
	if inst, err := w.st.GetInstallation(ctx, admin); err != nil || inst.OrgID != w.org.ID {
		t.Fatalf("administered org installation: %+v %v", inst, err)
	}
}

func itoa64(n int64) string { return strconv.FormatInt(n, 10) }
