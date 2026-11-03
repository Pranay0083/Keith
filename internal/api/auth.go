package api

import (
	"crypto/subtle"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"unicode"

	"github.com/Pranay0083/keith/internal/store"
)

// Sessions are stateless: keith_session is a signed (user ID, issue
// time, expiry) triple. Logout revokes it, and every other cookie issued
// to the user until then, by moving users.sessions_valid_after forward;
// sessionUser and the WebSocket recheck refuse a cookie issued at or
// before that time.

// setCookie sets an HttpOnly, SameSite=Lax, host-only cookie. maxAge < 0
// deletes it (Max-Age=0).
func (s *Server) setCookie(w http.ResponseWriter, name, value, path string, maxAge int) {
	http.SetCookie(w, &http.Cookie{
		Name:     name,
		Value:    value,
		Path:     path,
		MaxAge:   maxAge,
		HttpOnly: true,
		Secure:   s.secure,
		SameSite: http.SameSiteLaxMode,
	})
}

// validNext reports whether next is a safe same-origin path: it starts
// with one "/" (not "//" or "/\", which browsers treat as another host)
// and holds no control characters or spaces.
func validNext(next string) bool {
	if !strings.HasPrefix(next, "/") || strings.HasPrefix(next, "//") || strings.HasPrefix(next, "/\\") || len(next) > 2048 {
		return false
	}
	for _, r := range next {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

const defaultNext = "/sessions"

// handleLogin starts GitHub sign-in (GET /auth/github/login).
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	next := r.URL.Query().Get("next")
	if !validNext(next) {
		next = defaultNext
	}
	state := randomString(32)
	cookie, err := s.signer.signOAuthState(state, next, s.now().Add(oauthStateTTL))
	if err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}
	authURL, err := s.gh.AuthCodeURL(state)
	if err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}
	s.setCookie(w, oauthStateCookie, cookie, "/auth/github", int(oauthStateTTL.Seconds()))
	http.Redirect(w, r, authURL, http.StatusFound)
}

// loginFailed redirects to the web app's login page with an error code.
func (s *Server) loginFailed(w http.ResponseWriter, r *http.Request, code string, err error) {
	if err != nil {
		s.log(r).Warn("sign-in failed", "code", code, "err", err)
	}
	http.Redirect(w, r, s.webURLFor("/login?error="+url.QueryEscape(code)), http.StatusFound)
}

// handleCallback finishes GitHub sign-in (GET /auth/github/callback).
func (s *Server) handleCallback(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	// With "Request user authorization during installation" on, GitHub
	// may send the post-install redirect to the App's callback URL
	// instead of its setup URL. Those carry setup_action and the
	// install's own state.
	if q.Has("setup_action") || q.Has("installation_id") {
		s.handleSetup(w, r)
		return
	}

	// 1. The state must match the signed cookie; clear it either way.
	cookie, cerr := r.Cookie(oauthStateCookie)
	s.setCookie(w, oauthStateCookie, "", "/auth/github", -1)
	if cerr != nil {
		s.loginFailed(w, r, "oauth_state", nil)
		return
	}
	st, err := s.signer.verifyOAuthState(cookie.Value, s.now())
	if err != nil || subtle.ConstantTimeCompare([]byte(st.State), []byte(q.Get("state"))) != 1 {
		s.loginFailed(w, r, "oauth_state", nil)
		return
	}
	if e := q.Get("error"); e != "" {
		code := "oauth_failed"
		if e == "access_denied" {
			code = "oauth_denied"
		}
		s.loginFailed(w, r, code, nil)
		return
	}

	// 2. Code -> user token -> GitHub user.
	ctx := r.Context()
	tok, err := s.gh.Exchange(ctx, q.Get("code"))
	if err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}
	ghUser, err := s.gh.GetUser(ctx, tok.AccessToken)
	if err != nil || ghUser.ID <= 0 || ghUser.Login == "" {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}

	// 3. Upsert the user; 4. give them a personal org if they have none.
	user, err := s.store.UpsertUser(ctx, store.UpsertUserParams{
		GitHubUserID: ghUser.ID, GitHubLogin: ghUser.Login, Email: ghUser.Email,
		Name: ghUser.Name, AvatarURL: ghUser.AvatarURL,
	})
	if err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}
	if _, err := s.store.EnsurePersonalOrg(ctx, user.ID, user.GitHubLogin); err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}

	// 5. Session cookie, then back to the web app.
	now := s.now()
	val, err := s.signer.signSession(user.ID, now, now.Add(sessionTTL))
	if err != nil {
		s.loginFailed(w, r, "oauth_failed", err)
		return
	}
	s.setCookie(w, sessionCookie, val, "/", int(sessionTTL.Seconds()))
	next := st.Next
	if !validNext(next) {
		next = defaultNext
	}
	http.Redirect(w, r, s.webURLFor(next), http.StatusFound)
}

// handleLogout is POST /auth/logout. A valid cookie revokes every
// session cookie issued to its user so far, so a copy of it stops
// working too; then the cookie is cleared. It answers 204 with or
// without a valid cookie. If the revocation cannot be saved it answers
// 500 and keeps the cookie, so the client can retry.
func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) error {
	if c, err := r.Cookie(sessionCookie); err == nil {
		now := s.now()
		if cl, err := s.signer.verifySession(c.Value, now); err == nil {
			err := s.store.RevokeUserSessions(r.Context(), cl.UserID, now)
			if err != nil && !errors.Is(err, store.ErrNotFound) {
				return err
			}
		}
	}
	s.setCookie(w, sessionCookie, "", "/", -1)
	w.WriteHeader(http.StatusNoContent)
	return nil
}
