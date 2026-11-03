package github

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestNewAppValidates(t *testing.T) {
	_, keyPEM := rsaKey(t)
	cases := map[string]AppConfig{
		"no app id":    {PrivateKeyPEM: keyPEM},
		"no key":       {AppID: 1},
		"garbage key":  {AppID: 1, PrivateKeyPEM: []byte("not a key")},
		"bad api url":  {AppID: 1, PrivateKeyPEM: keyPEM, APIBaseURL: "ftp://x"},
		"api userinfo": {AppID: 1, PrivateKeyPEM: keyPEM, APIBaseURL: "https://u:p@api.github.com/"},
		"relative web": {AppID: 1, PrivateKeyPEM: keyPEM, WebBaseURL: "github.com"},
	}
	for name, cfg := range cases {
		if _, err := NewApp(cfg); err == nil {
			t.Errorf("%s: NewApp succeeded, want error", name)
		}
	}
}

func TestNewAppAcceptsPKCS8(t *testing.T) {
	k, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(k)
	if err != nil {
		t.Fatal(err)
	}
	pkcs8 := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})
	if _, err := NewApp(AppConfig{AppID: 1, PrivateKeyPEM: pkcs8}); err != nil {
		t.Fatalf("NewApp with PKCS#8 key: %v", err)
	}
}

func TestAppJWTClaims(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	if _, err := a.InstallationToken(context.Background(), 99, []string{"frontend"}, SandboxPermissions); err != nil {
		t.Fatal(err)
	}
	m := f.lastMint(t)
	now := f.clock.Now()
	if got, want := m.Claims.Issuer, testClientID; got != want {
		t.Errorf("iss = %q, want %q", got, want)
	}
	if got, want := m.Claims.IssuedAt.Time, now.Add(-60*time.Second); !got.Equal(want) {
		t.Errorf("iat = %v, want now-60s = %v", got, want)
	}
	if got, want := m.Claims.ExpiresAt.Time, now.Add(540*time.Second); !got.Equal(want) {
		t.Errorf("exp = %v, want now+540s = %v", got, want)
	}
	if alg := m.Header["alg"]; alg != "RS256" {
		t.Errorf("alg = %v, want RS256", alg)
	}
	if m.InstallationID != 99 {
		t.Errorf("installation = %d, want 99", m.InstallationID)
	}
}

func TestAppJWTIssuerFallsBackToAppID(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t, func(c *AppConfig) { c.ClientID = "" })
	if _, err := a.InstallationToken(context.Background(), 1, []string{"r"}, SandboxPermissions); err != nil {
		t.Fatal(err)
	}
	if got := f.lastMint(t).Claims.Issuer; got != fmt.Sprint(testAppID) {
		t.Errorf("iss = %q, want app id %d", got, testAppID)
	}
}

func TestInstallationTokenRequestBody(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	ctx := context.Background()

	tok, err := a.InstallationToken(ctx, 7, []string{"acme/frontend", "acme/backend"}, SandboxPermissions)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"repositories": []any{"frontend", "backend"},
		"permissions":  map[string]any{"contents": "write"},
	}
	if got := f.lastMint(t).Body; !reflect.DeepEqual(got, want) {
		t.Errorf("sandbox token body = %v, want %v", got, want)
	}
	if tok.Value != "ghs_test1" || !tok.ExpiresAt.Equal(f.clock.Now().Add(time.Hour)) {
		t.Errorf("token = %q expiring %v", tok.Value, tok.ExpiresAt)
	}

	if _, err := a.InstallationToken(ctx, 7, []string{"api"}, WorkerPermissions); err != nil {
		t.Fatal(err)
	}
	want = map[string]any{
		"repositories": []any{"api"},
		"permissions": map[string]any{
			"contents": "read", "pull_requests": "write", "issues": "read", "metadata": "read",
		},
	}
	if got := f.lastMint(t).Body; !reflect.DeepEqual(got, want) {
		t.Errorf("worker token body = %v, want %v", got, want)
	}
}

func TestInstallationTokenRefusesUnscoped(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	ctx := context.Background()
	cases := []struct {
		name  string
		id    int64
		repos []string
		perms Permissions
	}{
		{"nil repos", 1, nil, SandboxPermissions},
		{"empty repos", 1, []string{}, SandboxPermissions},
		{"zero perms", 1, []string{"r"}, Permissions{}},
		{"bad access", 1, []string{"r"}, Permissions{Contents: "admin"}},
		{"metadata write", 1, []string{"r"}, Permissions{Metadata: Write}},
		{"mixed owners", 1, []string{"acme/a", "evil/b"}, SandboxPermissions},
		{"bad repo", 1, []string{"acme/../x"}, SandboxPermissions},
		{"empty repo name", 1, []string{""}, SandboxPermissions},
		{"zero installation", 0, []string{"r"}, SandboxPermissions},
	}
	for _, c := range cases {
		if _, err := a.InstallationToken(ctx, c.id, c.repos, c.perms); err == nil {
			t.Errorf("%s: InstallationToken succeeded, want error", c.name)
		}
	}
	if _, err := a.InstallationToken(ctx, 1, nil, SandboxPermissions); !errors.Is(err, ErrNoRepos) {
		t.Errorf("nil repos: err = %v, want ErrNoRepos", err)
	}
	if n := f.mintCount(); n != 0 {
		t.Errorf("%d tokens minted for invalid requests, want 0", n)
	}
}

func TestInstallationTokenAPIError(t *testing.T) {
	f := newFakeGitHub(t)
	f.setMintFail(http.StatusNotFound)
	_, err := f.app(t).InstallationToken(context.Background(), 5, []string{"r"}, SandboxPermissions)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestRevokeToken(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	ctx := context.Background()
	tok, err := a.InstallationToken(ctx, 1, []string{"r"}, SandboxPermissions)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.RevokeToken(ctx, tok.Value); err != nil {
		t.Fatalf("RevokeToken: %v", err)
	}
	if got := f.revokedTokens(); !reflect.DeepEqual(got, []string{tok.Value}) {
		t.Errorf("revoked = %v, want [%s]", got, tok.Value)
	}
	// A second revoke sees 401 and still succeeds.
	if err := a.RevokeToken(ctx, tok.Value); err != nil {
		t.Errorf("second RevokeToken: %v", err)
	}
	if err := a.RevokeToken(ctx, ""); err == nil {
		t.Error("RevokeToken(\"\") succeeded")
	}
}

func TestTokenRedacted(t *testing.T) {
	tok := Token{Value: "ghs_supersecret", ExpiresAt: time.Unix(0, 0)}
	var logged strings.Builder
	slog.New(slog.NewJSONHandler(&logged, nil)).Info("x", "tok", tok)
	for _, s := range []string{fmt.Sprint(tok), fmt.Sprintf("%v %+v %#v %s", tok, tok, tok, tok), logged.String()} {
		if strings.Contains(s, "supersecret") {
			t.Errorf("token value leaked: %s", s)
		}
	}
}
