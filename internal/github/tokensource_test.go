package github

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"sync"
	"testing"
	"time"
)

func newTestSource(t *testing.T, f *fakeGitHub) *TokenSource {
	t.Helper()
	s, err := NewTokenSource(f.app(t), 77, []string{"acme/api", "acme/web"}, SandboxPermissions)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestNewTokenSourceRefusesUnscoped(t *testing.T) {
	f := newFakeGitHub(t)
	a := f.app(t)
	if _, err := NewTokenSource(a, 1, nil, SandboxPermissions); !errors.Is(err, ErrNoRepos) {
		t.Errorf("nil repos: err = %v, want ErrNoRepos", err)
	}
	if _, err := NewTokenSource(a, 1, []string{"r"}, Permissions{}); err == nil {
		t.Error("zero perms: no error")
	}
	if _, err := NewTokenSource(a, 0, []string{"r"}, SandboxPermissions); err == nil {
		t.Error("zero installation: no error")
	}
}

func TestTokenSourceRefreshTiming(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	var refreshed []string
	s.OnRefresh(func(tok Token) { refreshed = append(refreshed, tok.Value) })
	ctx := context.Background()

	get := func() string {
		t.Helper()
		tok, err := s.Token(ctx)
		if err != nil {
			t.Fatalf("Token: %v", err)
		}
		return tok.Value
	}

	if got := get(); got != "ghs_test1" {
		t.Fatalf("first token = %q", got)
	}
	// 10m01s left: still cached.
	f.clock.Advance(49*time.Minute + 59*time.Second)
	if got := get(); got != "ghs_test1" {
		t.Errorf("with 10m01s left, token = %q, want cached ghs_test1", got)
	}
	// Exactly 10m left: refreshed.
	f.clock.Advance(time.Second)
	if got := get(); got != "ghs_test2" {
		t.Errorf("with 10m left, token = %q, want refreshed ghs_test2", got)
	}
	if got := get(); got != "ghs_test2" {
		t.Errorf("right after refresh, token = %q", got)
	}
	if n := f.mintCount(); n != 2 {
		t.Errorf("minted %d tokens, want 2", n)
	}
	if want := []string{"ghs_test1", "ghs_test2"}; !reflect.DeepEqual(refreshed, want) {
		t.Errorf("OnRefresh saw %v, want %v", refreshed, want)
	}
	m := f.lastMint(t)
	if m.InstallationID != 77 || !reflect.DeepEqual(m.Body["repositories"], []any{"api", "web"}) {
		t.Errorf("refresh mint = installation %d body %v", m.InstallationID, m.Body)
	}
}

func TestTokenSourceConcurrentCallersShareOneMint(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	ctx := context.Background()

	burst := func() {
		var wg sync.WaitGroup
		for range 32 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := s.Token(ctx); err != nil {
					t.Errorf("Token: %v", err)
				}
			}()
		}
		wg.Wait()
	}
	burst()
	if n := f.mintCount(); n != 1 {
		t.Fatalf("32 concurrent first calls minted %d tokens, want 1", n)
	}
	f.clock.Advance(55 * time.Minute)
	burst()
	if n := f.mintCount(); n != 2 {
		t.Fatalf("32 concurrent refresh calls minted %d tokens in total, want 2", n)
	}
}

func TestTokenSourceRunRefreshesInBackground(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	got := make(chan string, 4)
	s.OnRefresh(func(tok Token) { got <- tok.Value })

	ctx, cancel := context.WithCancel(context.Background())
	runErr := make(chan error, 1)
	go func() { runErr <- s.Run(ctx) }()

	if v := recv(t, got); v != "ghs_test1" {
		t.Fatalf("first token = %q", v)
	}
	f.clock.waitForWaiters(t, 1)
	// Run sleeps until 10 minutes before expiry.
	f.clock.Advance(49 * time.Minute)
	select {
	case v := <-got:
		t.Fatalf("refreshed %q 11 minutes before expiry", v)
	case <-time.After(20 * time.Millisecond):
	}
	f.clock.Advance(time.Minute)
	if v := recv(t, got); v != "ghs_test2" {
		t.Fatalf("background refresh = %q, want ghs_test2", v)
	}
	f.clock.waitForWaiters(t, 1)

	cancel()
	if err := <-runErr; !errors.Is(err, context.Canceled) {
		t.Errorf("Run = %v, want context.Canceled", err)
	}
}

func TestTokenSourceRunRetriesFailedMint(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	errs := make(chan error, 4)
	s.OnError(func(err error) { errs <- err })
	got := make(chan string, 4)
	s.OnRefresh(func(tok Token) { got <- tok.Value })

	f.setMintFail(http.StatusInternalServerError)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = s.Run(ctx) }()

	if err := recv(t, errs); err == nil {
		t.Fatal("OnError got nil")
	}
	f.clock.waitForWaiters(t, 1)
	f.setMintFail(0)
	f.clock.Advance(refreshRetry)
	if v := recv(t, got); v != "ghs_test1" {
		t.Fatalf("token after retry = %q", v)
	}
}

func TestTokenSourceKeepsUsableTokenWhenRefreshFails(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	var errs []error
	s.OnError(func(err error) { errs = append(errs, err) })
	ctx := context.Background()
	if _, err := s.Token(ctx); err != nil {
		t.Fatal(err)
	}

	f.setMintFail(http.StatusBadGateway)
	f.clock.Advance(52 * time.Minute) // 8 minutes left
	tok, err := s.Token(ctx)
	if err != nil || tok.Value != "ghs_test1" {
		t.Fatalf("Token with 8m left and a failing refresh = %q, %v; want ghs_test1, nil", tok.Value, err)
	}
	if len(errs) != 1 {
		t.Errorf("OnError called %d times, want 1", len(errs))
	}

	f.clock.Advance(7*time.Minute + 30*time.Second) // 30 seconds left
	if _, err := s.Token(ctx); err == nil {
		t.Error("Token with 30s left and a failing refresh succeeded")
	}
}

func TestTokenSourceRevoke(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	ctx := context.Background()
	tok, err := s.Token(ctx)
	if err != nil {
		t.Fatal(err)
	}

	runErr := make(chan error, 1)
	go func() { runErr <- s.Run(ctx) }()
	f.clock.waitForWaiters(t, 1)

	if err := s.Revoke(ctx); err != nil {
		t.Fatalf("Revoke: %v", err)
	}
	if got := f.revokedTokens(); !reflect.DeepEqual(got, []string{tok.Value}) {
		t.Errorf("revoked %v, want [%s]", got, tok.Value)
	}
	if err := <-runErr; err != nil {
		t.Errorf("Run after Revoke = %v, want nil", err)
	}
	if _, err := s.Token(ctx); !errors.Is(err, ErrTokenSourceClosed) {
		t.Errorf("Token after Revoke = %v, want ErrTokenSourceClosed", err)
	}
	if err := s.Revoke(ctx); err != nil {
		t.Errorf("second Revoke: %v", err)
	}
	if n := f.mintCount(); n != 1 {
		t.Errorf("minted %d tokens, want 1", n)
	}
}

func TestTokenSourceRevokeBeforeFirstToken(t *testing.T) {
	f := newFakeGitHub(t)
	s := newTestSource(t, f)
	if err := s.Revoke(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.revokedTokens()) != 0 {
		t.Error("revoked a token that was never minted")
	}
}

func recv[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(5 * time.Second):
		t.Fatal("timed out")
		var zero T
		return zero
	}
}
