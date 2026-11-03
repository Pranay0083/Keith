package api

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

// Signed tokens: the session cookie, the OAuth state cookie and the
// install setup state. Each is
//
//	base64url(JSON claims) "." base64url(HMAC-SHA256(key, purpose \x00 part1))
//
// with key derived from SESSION_SECRET. The purpose string separates the
// three kinds, so one can never be replayed as another. Verification is
// constant time and checks the expiry claim.

// Token purposes.
const (
	purposeSession    = "keith/session/v1"
	purposeOAuthState = "keith/oauth-state/v1"
	purposeSetupState = "keith/setup-state/v1"
)

var (
	errTokenInvalid = errors.New("api: invalid token")
	errTokenExpired = errors.New("api: token expired")
)

// signer signs and verifies tokens with one secret.
type signer struct {
	key []byte
}

func newSigner(secret []byte) *signer {
	// Derive the MAC key so SESSION_SECRET itself is used for nothing
	// else.
	m := hmac.New(sha256.New, secret)
	m.Write([]byte("keith/api/token-key/v1"))
	return &signer{key: m.Sum(nil)}
}

func (s *signer) mac(purpose, body string) []byte {
	m := hmac.New(sha256.New, s.key)
	m.Write([]byte(purpose))
	m.Write([]byte{0})
	m.Write([]byte(body))
	return m.Sum(nil)
}

// claims are the fields every token carries.
type claims struct {
	// Exp is the expiry, in Unix seconds.
	Exp int64 `json:"exp"`
}

// sign encodes v (a struct embedding claims) and signs it.
func (s *signer) sign(purpose string, v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	body := base64.RawURLEncoding.EncodeToString(b)
	return body + "." + base64.RawURLEncoding.EncodeToString(s.mac(purpose, body)), nil
}

// verify checks token's signature and expiry and decodes it into v,
// which must embed claims (exp is read through expOf).
func (s *signer) verify(purpose, token string, now time.Time, v any, expOf func() int64) error {
	if len(token) > 4096 {
		return errTokenInvalid
	}
	body, sig, ok := strings.Cut(token, ".")
	if !ok || body == "" || sig == "" {
		return errTokenInvalid
	}
	got, err := base64.RawURLEncoding.DecodeString(sig)
	if err != nil || !hmac.Equal(got, s.mac(purpose, body)) {
		return errTokenInvalid
	}
	b, err := base64.RawURLEncoding.DecodeString(body)
	if err != nil {
		return errTokenInvalid
	}
	if err := json.Unmarshal(b, v); err != nil {
		return errTokenInvalid
	}
	if now.Unix() >= expOf() {
		return errTokenExpired
	}
	return nil
}

// sessionClaims is the session cookie: who is signed in, since when and
// until when.
type sessionClaims struct {
	claims
	UserID string `json:"uid"`
	// IssuedAtMs is when the cookie was issued, in Unix milliseconds, so
	// logout can revoke it (users.sessions_valid_after). Cookies from
	// before it existed have 0.
	IssuedAtMs int64 `json:"iat_ms,omitempty"`
}

// IssuedAt returns the cookie's issue time.
func (c sessionClaims) IssuedAt() time.Time { return time.UnixMilli(c.IssuedAtMs) }

// oauthStateClaims is the keith_oauth_state cookie.
type oauthStateClaims struct {
	claims
	State string `json:"state"`
	Next  string `json:"next"`
}

// setupStateClaims is the state passed through a GitHub App install.
type setupStateClaims struct {
	claims
	OrgID  string `json:"org"`
	UserID string `json:"uid"`
	Nonce  string `json:"n"`
}

func (s *signer) signSession(userID string, iat, exp time.Time) (string, error) {
	return s.sign(purposeSession, sessionClaims{claims: claims{exp.Unix()}, UserID: userID, IssuedAtMs: iat.UnixMilli()})
}

func (s *signer) verifySession(token string, now time.Time) (sessionClaims, error) {
	var c sessionClaims
	err := s.verify(purposeSession, token, now, &c, func() int64 { return c.Exp })
	if err == nil && c.UserID == "" {
		err = errTokenInvalid
	}
	return c, err
}

func (s *signer) signOAuthState(state, next string, exp time.Time) (string, error) {
	return s.sign(purposeOAuthState, oauthStateClaims{claims{exp.Unix()}, state, next})
}

func (s *signer) verifyOAuthState(token string, now time.Time) (oauthStateClaims, error) {
	var c oauthStateClaims
	err := s.verify(purposeOAuthState, token, now, &c, func() int64 { return c.Exp })
	if err == nil && c.State == "" {
		err = errTokenInvalid
	}
	return c, err
}

func (s *signer) signSetupState(orgID, userID string, exp time.Time) (string, error) {
	return s.sign(purposeSetupState, setupStateClaims{claims{exp.Unix()}, orgID, userID, randomString(16)})
}

func (s *signer) verifySetupState(token string, now time.Time) (setupStateClaims, error) {
	var c setupStateClaims
	err := s.verify(purposeSetupState, token, now, &c, func() int64 { return c.Exp })
	if err == nil && (c.OrgID == "" || c.UserID == "") {
		err = errTokenInvalid
	}
	return c, err
}

// randomString returns n random bytes, base64url-encoded.
func randomString(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("api: crypto/rand failed: " + err.Error())
	}
	return base64.RawURLEncoding.EncodeToString(b)
}
