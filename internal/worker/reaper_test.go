package worker

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/llm/llmtest"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

// stuckDeleteProvider fails Delete for one sandbox.
type stuckDeleteProvider struct {
	*memsandbox.Provider
	stuck string
}

type stuckSandbox struct{ sandbox.Sandbox }

func (stuckSandbox) Delete(context.Context) error {
	return errors.New("sandbox is in state error; cannot delete")
}

func (p *stuckDeleteProvider) Get(ctx context.Context, id string) (sandbox.Sandbox, error) {
	sb, err := p.Provider.Get(ctx, id)
	if err == nil && id == p.stuck {
		return stuckSandbox{sb}, nil
	}
	return sb, err
}

// One sandbox that cannot be deleted, at the head of the idle list
// (oldest updated_at), must not stop the reaper from reaping the others,
// nor from running the orphan pass.
func TestReaperSkipsASandboxItCannotDelete(t *testing.T) {
	e := newEnv(t)
	e.opts.IdleTimeout = time.Minute
	e.opts.OrphanLabels = map[string]string{"keith.test": "w"}
	e.opts.SandboxLabels = map[string]string{"keith.test": "w"}
	s1 := e.newSession()
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s1.ID); err != nil {
		t.Fatal(err)
	}
	s2 := e.newSession()
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s2.ID); err != nil {
		t.Fatal(err)
	}
	e.st.Backdate(s1.ID, 3*time.Minute)
	e.st.Backdate(s2.ID, 2*time.Minute)
	ctx := context.Background()
	orphan, _ := e.sbx.Create(ctx, sandbox.Spec{Labels: map[string]string{"keith.test": "w"}})

	w := e.worker()
	stuck := e.st.Session(s1.ID).SandboxID
	lp := &listingProvider{Provider: e.sbx, ids: []string{orphan.ID()}}
	w.d.Sandboxes = &stuckListingProvider{stuckDeleteProvider{Provider: e.sbx, stuck: stuck}, lp}
	now := time.Now()
	w.d.Now = func() time.Time { return now }
	if _, err := w.Reap(ctx); err != nil {
		t.Fatalf("first pass: %v", err)
	}
	now = now.Add(w.o.OrphanGrace + time.Second)
	n, err := w.Reap(ctx)
	if err != nil {
		t.Fatalf("second pass: %v", err)
	}
	if e.st.Session(s2.ID).SandboxID != "" {
		t.Error("s2's idle sandbox was not reaped because s1's delete fails first")
	}
	if e.st.Session(s1.ID).SandboxID != stuck {
		t.Error("s1's sandbox was forgotten although its delete failed")
	}
	if ms, _ := e.sbx.Sandbox(orphan.ID()); !ms.Deleted() {
		t.Errorf("orphan not deleted (reaped %d)", n)
	}
}

// stuckListingProvider is a stuckDeleteProvider that can also list.
type stuckListingProvider struct {
	stuckDeleteProvider
	lister *listingProvider
}

func (p *stuckListingProvider) List(ctx context.Context, labels map[string]string) ([]string, error) {
	return p.lister.List(ctx, labels)
}

// movingProvider records a new sandbox for the session while the
// reaper deletes the old one, as a worker that took the session over
// after the reaper's lease lapsed would.
type movingProvider struct {
	*memsandbox.Provider
	onDelete func()
}

type movingSandbox struct {
	sandbox.Sandbox
	onDelete func()
}

func (m movingSandbox) Delete(ctx context.Context) error {
	m.onDelete()
	return m.Sandbox.Delete(ctx)
}

func (p *movingProvider) Get(ctx context.Context, id string) (sandbox.Sandbox, error) {
	sb, err := p.Provider.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	return movingSandbox{sb, p.onDelete}, nil
}

// The reaper clears sessions.sandbox_id only if it still names the
// sandbox it deleted.
func TestReaperKeepsASandboxRecordedMeanwhile(t *testing.T) {
	e := newEnv(t)
	e.opts.IdleTimeout = time.Minute
	s := e.newSession()
	e.llm.Push(llmtest.Text("done"))
	if err := e.run(s.ID); err != nil {
		t.Fatal(err)
	}
	e.st.Backdate(s.ID, 2*time.Minute)
	w := e.worker()
	ctx := context.Background()
	w.d.Sandboxes = &movingProvider{Provider: e.sbx, onDelete: func() {
		_ = e.st.SetSessionSandbox(ctx, s.ID, "sb-new", memsandbox.Name)
	}}
	if _, err := w.Reap(ctx); err != nil {
		t.Fatal(err)
	}
	if got := e.st.Session(s.ID).SandboxID; got != "sb-new" {
		t.Errorf("sandbox_id = %q, want the newly recorded sb-new", got)
	}
}
