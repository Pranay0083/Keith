package store_test

import (
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

func TestRevokeUserSessions(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	u := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 1, GitHubLogin: "u"}))
	if !u.SessionsValidAfter.IsZero() {
		t.Fatalf("new user has a cutoff: %v", u.SessionsValidAfter)
	}
	at := time.Date(2026, 11, 3, 10, 0, 0, 123456000, time.UTC)
	ok(t, s.RevokeUserSessions(ctx, u.ID, at))
	if got := must(s.GetUser(ctx, u.ID)).SessionsValidAfter; !got.Equal(at) {
		t.Fatalf("cutoff = %v, want %v", got, at)
	}
	// It only moves forward.
	ok(t, s.RevokeUserSessions(ctx, u.ID, at.Add(-time.Hour)))
	if got := must(s.GetUser(ctx, u.ID)).SessionsValidAfter; !got.Equal(at) {
		t.Fatalf("cutoff moved back to %v", got)
	}
	ok(t, s.RevokeUserSessions(ctx, u.ID, at.Add(time.Second)))
	if got := must(s.GetUsers(ctx, []string{u.ID}))[u.ID].SessionsValidAfter; !got.Equal(at.Add(time.Second)) {
		t.Fatalf("cutoff = %v", got)
	}
	// Signing in again keeps the cutoff.
	if got := must(s.UpsertUser(ctx, store.UpsertUserParams{GitHubUserID: 1, GitHubLogin: "u2"})).SessionsValidAfter; !got.Equal(at.Add(time.Second)) {
		t.Fatalf("cutoff after sign-in = %v", got)
	}
	isErr(t, s.RevokeUserSessions(ctx, missingID, at), store.ErrNotFound)
	isErr(t, s.RevokeUserSessions(ctx, "nope", at), store.ErrNotFound)
}
