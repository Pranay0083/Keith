package store_test

import (
	"testing"

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
