package store_test

import (
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

func TestReaperQueries(t *testing.T) {
	s := storetest.New(t)
	ctx := ctxT(t)
	f := storetest.Seed(t, s)
	id := f.Session.ID

	idle := func(since time.Time) []string {
		t.Helper()
		var out []string
		for _, x := range must(s.ListIdleSandboxSessions(ctx, since, 0)) {
			out = append(out, x.ID)
		}
		return out
	}
	future := time.Now().Add(time.Hour)
	if got := idle(future); len(got) != 0 {
		t.Fatalf("no sandbox yet: idle = %v", got)
	}
	if err := s.SetSessionSandbox(ctx, id, "keith-sb-1", "docker"); err != nil {
		t.Fatal(err)
	}
	if got := idle(future); len(got) != 0 {
		t.Fatalf("queued session is not idle: %v", got)
	}
	if err := s.SetSessionStatus(ctx, id, events.StatusWaitingForUser, ""); err != nil {
		t.Fatal(err)
	}
	if got := idle(future); len(got) != 1 || got[0] != id {
		t.Fatalf("waiting session: idle = %v", got)
	}
	if got := idle(time.Now().Add(-time.Hour)); len(got) != 0 {
		t.Fatalf("recently updated session is idle: %v", got)
	}
	if err := s.SetSessionStatus(ctx, id, events.StatusCompleted, ""); err != nil {
		t.Fatal(err)
	}
	if got := idle(future); len(got) != 1 {
		t.Fatalf("completed session: idle = %v", got)
	}

	used := must(s.SandboxIDsInUse(ctx, []string{"keith-sb-1", "keith-sb-orphan"}))
	if !used["keith-sb-1"] || used["keith-sb-orphan"] || len(used) != 1 {
		t.Fatalf("in use = %v", used)
	}
	if err := s.SetSessionSandbox(ctx, id, "", ""); err != nil {
		t.Fatal(err)
	}
	if got := idle(future); len(got) != 0 {
		t.Fatalf("after clearing the sandbox: idle = %v", got)
	}
}
