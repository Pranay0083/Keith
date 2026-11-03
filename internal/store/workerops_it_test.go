package store_test

import (
	"slices"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

func TestClearSessionSandboxIsCompareAndSet(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	id := f.Session.ID

	if err := s.SetSessionSandbox(ctx, id, "keith-sb-new", "docker"); err != nil {
		t.Fatal(err)
	}
	if ok := must(s.ClearSessionSandbox(ctx, id, "keith-sb-old")); ok {
		t.Fatal("cleared a sandbox that is not the session's")
	}
	if got := must(s.GetSession(ctx, id)); got.SandboxID != "keith-sb-new" || got.SandboxProvider != "docker" {
		t.Fatalf("sandbox = %q/%q", got.SandboxID, got.SandboxProvider)
	}
	if ok := must(s.ClearSessionSandbox(ctx, id, "keith-sb-new")); !ok {
		t.Fatal("did not clear the session's sandbox")
	}
	if got := must(s.GetSession(ctx, id)); got.SandboxID != "" || got.SandboxProvider != "" {
		t.Fatalf("after clear: sandbox = %q/%q", got.SandboxID, got.SandboxProvider)
	}
	if ok := must(s.ClearSessionSandbox(ctx, id, "keith-sb-new")); ok {
		t.Fatal("cleared twice")
	}
}

func TestListStaleLiveSessions(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	id := f.Session.ID

	stale := func(before time.Time) []string {
		t.Helper()
		var out []string
		for _, x := range must(s.ListStaleLiveSessions(ctx, before, 0)) {
			out = append(out, x.ID)
		}
		return out
	}
	future := time.Now().Add(time.Hour)
	if got := stale(future); !slices.Contains(got, id) {
		t.Fatalf("queued session not listed: %v", got)
	}
	if got := stale(time.Now().Add(-time.Hour)); slices.Contains(got, id) {
		t.Fatalf("recently updated session listed: %v", got)
	}
	for _, st := range []events.SessionStatus{events.StatusWaitingForUser, events.StatusCompleted} {
		if err := s.SetSessionStatus(ctx, id, st, ""); err != nil {
			t.Fatal(err)
		}
		if got := stale(future); slices.Contains(got, id) {
			t.Fatalf("%s session listed", st)
		}
	}
	if err := s.SetSessionStatus(ctx, id, events.StatusRunning, ""); err != nil {
		t.Fatal(err)
	}
	if got := stale(future); !slices.Contains(got, id) {
		t.Fatalf("running session not listed: %v", got)
	}
}
