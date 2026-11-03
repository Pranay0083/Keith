package events_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/memevents"
)

const sid = "s1"

// flakyBus wraps a memevents.Bus with the failures a real Bus has:
// dropped publishes, broken subscriptions and outages.
type flakyBus struct {
	inner *memevents.Bus

	mu         sync.Mutex
	drop       func(events.Event) bool
	down       bool
	cancels    []func()
	subscribes int
}

func newFlakyBus() *flakyBus { return &flakyBus{inner: memevents.NewBus()} }

func (b *flakyBus) Publish(ctx context.Context, e events.Event) error {
	b.mu.Lock()
	drop := b.down || (b.drop != nil && b.drop(e))
	b.mu.Unlock()
	if drop {
		return nil
	}
	return b.inner.Publish(ctx, e)
}

func (b *flakyBus) Subscribe(ctx context.Context, sessionID string) (<-chan events.Event, func(), error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subscribes++
	if b.down {
		return nil, nil, errors.New("bus down")
	}
	ch, cancel, err := b.inner.Subscribe(ctx, sessionID)
	if err == nil {
		b.cancels = append(b.cancels, cancel)
	}
	return ch, cancel, err
}

// breakAll closes every live subscription, like a dropped connection.
func (b *flakyBus) breakAll() {
	b.mu.Lock()
	cancels := b.cancels
	b.cancels = nil
	b.mu.Unlock()
	for _, c := range cancels {
		c()
	}
}

func (b *flakyBus) setDown(down bool) {
	b.mu.Lock()
	b.down = down
	b.mu.Unlock()
}

func (b *flakyBus) setDrop(f func(events.Event) bool) {
	b.mu.Lock()
	b.drop = f
	b.mu.Unlock()
}

func (b *flakyBus) subscribeCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.subscribes
}

// hookStore runs a hook before each After call.
type hookStore struct {
	*memevents.Store
	before func(call int)
	err    error
	calls  int
}

func (s *hookStore) After(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]events.Event, error) {
	s.calls++
	if s.before != nil {
		s.before(s.calls)
	}
	if s.err != nil {
		return nil, s.err
	}
	return s.Store.After(ctx, sessionID, afterSeq, limit)
}

func emit(t *testing.T, em *events.Emitter, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if _, err := em.Emit(context.Background(), sid, events.TypeToolCallOutput,
			events.ToolCallOutputPayload{CallID: "c", Chunk: fmt.Sprint(i)}); err != nil {
			t.Fatal(err)
		}
	}
}

func emitDelta(t *testing.T, em *events.Emitter, text string) {
	t.Helper()
	if _, err := em.Emit(context.Background(), sid, events.TypeMessageDelta,
		events.MessageDeltaPayload{MessageID: "m", Text: text}); err != nil {
		t.Fatal(err)
	}
}

// recv reads n events or fails after a timeout.
func recv(t *testing.T, ch <-chan events.Event, n int) []events.Event {
	t.Helper()
	var out []events.Event
	timeout := time.After(5 * time.Second)
	for len(out) < n {
		select {
		case ev, ok := <-ch:
			if !ok {
				t.Fatalf("channel closed after %d of %d events: %v", len(out), n, seqs(out))
			}
			out = append(out, ev)
		case <-timeout:
			t.Fatalf("timed out after %d of %d events: %v", len(out), n, seqs(out))
		}
	}
	return out
}

// quiet fails if anything arrives on ch within d.
func quiet(t *testing.T, ch <-chan events.Event, d time.Duration) {
	t.Helper()
	select {
	case ev, ok := <-ch:
		if ok {
			t.Fatalf("unexpected event seq=%d type=%s", ev.Seq, ev.Type)
		}
		t.Fatal("channel closed unexpectedly")
	case <-time.After(d):
	}
}

func seqs(evs []events.Event) []int64 {
	out := make([]int64, len(evs))
	for i, ev := range evs {
		out[i] = ev.Seq
	}
	return out
}

func wantSeqs(t *testing.T, evs []events.Event, from, to int64) {
	t.Helper()
	got := seqs(evs)
	if int64(len(got)) != to-from+1 {
		t.Fatalf("seqs = %v, want %d..%d", got, from, to)
	}
	for i, s := range got {
		if s != from+int64(i) {
			t.Fatalf("seqs = %v, want %d..%d", got, from, to)
		}
	}
}

func TestReplayBacklogThenLive(t *testing.T) {
	store, bus := memevents.NewStore(), memevents.NewBus()
	em := &events.Emitter{Store: store, Bus: bus}
	emit(t, em, 5)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := events.Replay(ctx, store, bus, sid, 2)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch, 3), 3, 5)

	emit(t, em, 2)
	emitDelta(t, em, "typing")
	got := recv(t, ch, 3)
	wantSeqs(t, got[:2], 6, 7)
	if got[2].Seq != 0 || got[2].Type != events.TypeMessageDelta {
		t.Fatalf("delta = %+v", got[2])
	}

	// A stale publish of an already-sent seq is dropped.
	old := store.All(sid)[3]
	if err := bus.Publish(ctx, old); err != nil {
		t.Fatal(err)
	}
	quiet(t, ch, 50*time.Millisecond)
}

func TestReplayPagesLargeBacklog(t *testing.T) {
	store, bus := memevents.NewStore(), memevents.NewBus()
	em := &events.Emitter{Store: store, Bus: bus}
	n := events.ReplayPageSize*2 + 7
	emit(t, em, n)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := events.Replay(ctx, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch, n), 1, int64(n))
	quiet(t, ch, 20*time.Millisecond)
}

// Events written between Subscribe and the Store read arrive exactly once,
// whether the read sees them or only the Bus does.
func TestReplayRaceBetweenSubscribeAndRead(t *testing.T) {
	for _, tc := range []struct {
		name      string
		afterRead bool
	}{{"stored before read", false}, {"stored after read", true}} {
		t.Run(tc.name, func(t *testing.T) {
			mem, bus := memevents.NewStore(), memevents.NewBus()
			em := &events.Emitter{Store: mem, Bus: bus}
			emit(t, em, 2)

			store := &hookStore{Store: mem}
			store.before = func(call int) {
				if call == 1 && !tc.afterRead {
					emit(t, em, 2) // lands in the read and on the Bus
				}
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ch, err := events.Replay(ctx, store, bus, sid, 0)
			if err != nil {
				t.Fatal(err)
			}
			if tc.afterRead {
				emit(t, em, 2) // only on the Bus
			}
			wantSeqs(t, recv(t, ch, 4), 1, 4)
			quiet(t, ch, 50*time.Millisecond)
		})
	}
}

func TestReplayFillsBusGaps(t *testing.T) {
	store, bus := memevents.NewStore(), newFlakyBus()
	em := &events.Emitter{Store: store, Bus: bus}
	emit(t, em, 2)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := events.Replay(ctx, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch, 2), 1, 2)

	// The Bus loses seqs 3 and 4; seq 5 reveals the gap.
	bus.setDrop(func(e events.Event) bool { return e.Seq == 3 || e.Seq == 4 })
	emit(t, em, 3)
	wantSeqs(t, recv(t, ch, 3), 3, 5)
	quiet(t, ch, 50*time.Millisecond)
}

// The client-side case: a client reads some events, disconnects, misses
// more, and reconnects with the last seq it saw.
func TestReplayReconnectWithLastSeq(t *testing.T) {
	store, bus := memevents.NewStore(), memevents.NewBus()
	em := &events.Emitter{Store: store, Bus: bus}
	emit(t, em, 3)

	ctx1, cancel1 := context.WithCancel(context.Background())
	ch1, err := events.Replay(ctx1, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	first := recv(t, ch1, 3)
	cancel1()
	for range ch1 { // closes on cancel
	}
	if n := bus.Subscribers(sid); n != 0 {
		t.Fatalf("%d subscribers left after cancel", n)
	}

	emit(t, em, 4) // missed while disconnected
	emitDelta(t, em, "lost")

	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	ch2, err := events.Replay(ctx2, store, bus, sid, first[len(first)-1].Seq)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch2, 4), 4, 7)
	emit(t, em, 1)
	wantSeqs(t, recv(t, ch2, 1), 8, 8)
	quiet(t, ch2, 50*time.Millisecond)
}

// The server-side case: the Bus connection breaks under a live replay and
// stays down for a while; Replay resubscribes and loses nothing.
func TestReplayResubscribesAfterBusFailure(t *testing.T) {
	store, bus := memevents.NewStore(), newFlakyBus()
	em := &events.Emitter{Store: store, Bus: bus}
	emit(t, em, 2)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := events.Replay(ctx, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch, 2), 1, 2)

	bus.setDown(true)
	bus.breakAll()
	// Stored while the Bus is down: Replay polls the Store meanwhile.
	for i := 0; i < 3; i++ {
		// The flaky bus swallows publishes while down.
		if _, err := em.Emit(ctx, sid, events.TypeError, events.ErrorPayload{Message: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	wantSeqs(t, recv(t, ch, 3), 3, 5)

	bus.setDown(false)
	deadline := time.Now().Add(5 * time.Second)
	for bus.inner.Subscribers(sid) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if bus.inner.Subscribers(sid) != 1 {
		t.Fatalf("not resubscribed after %d attempts", bus.subscribeCount())
	}
	emit(t, em, 2)
	wantSeqs(t, recv(t, ch, 2), 6, 7)
	quiet(t, ch, 50*time.Millisecond)
}

func TestReplayCancelClosesAndUnsubscribes(t *testing.T) {
	store, bus := memevents.NewStore(), memevents.NewBus()
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := events.Replay(ctx, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	if bus.Subscribers(sid) != 1 {
		t.Fatal("not subscribed")
	}
	cancel()
	select {
	case _, ok := <-ch:
		if ok {
			t.Fatal("event after cancel")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("channel not closed after cancel")
	}
	deadline := time.Now().Add(5 * time.Second)
	for bus.Subscribers(sid) != 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if bus.Subscribers(sid) != 0 {
		t.Fatal("still subscribed after cancel")
	}
}

func TestReplayCancelWhileClientIsNotReading(t *testing.T) {
	store, bus := memevents.NewStore(), memevents.NewBus()
	em := &events.Emitter{Store: store, Bus: bus}
	emit(t, em, 100) // more than the output buffer
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := events.Replay(ctx, store, bus, sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	cancel()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return
			}
		case <-deadline:
			t.Fatal("channel not closed")
		}
	}
}

func TestReplayErrors(t *testing.T) {
	bus := newFlakyBus()
	bus.setDown(true)
	if _, err := events.Replay(context.Background(), memevents.NewStore(), bus, sid, 0); err == nil {
		t.Fatal("subscribe failure not reported")
	}

	mem := memevents.NewBus()
	store := &hookStore{Store: memevents.NewStore(), err: errors.New("db down")}
	if _, err := events.Replay(context.Background(), store, mem, sid, 0); err == nil {
		t.Fatal("store failure not reported")
	}
	if mem.Subscribers(sid) != 0 {
		t.Fatal("subscription leaked after store failure")
	}
}
