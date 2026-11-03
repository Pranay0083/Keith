package lease_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/redistest"
)

// Integration tests: KEITH_IT=1, against REDIS_URL (see redistest).

func newKey() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return lease.SessionKey("test-" + hex.EncodeToString(b))
}

func TestExclusive(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()
	key := newKey()

	l, err := m.Acquire(ctx, key, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if l.Key() != key || l.Token() == "" || l.Fence() < 1 || !l.Valid() {
		t.Fatalf("lease %+v", l)
	}
	if _, err := m.Acquire(ctx, key, 5*time.Second); !errors.Is(err, lease.ErrHeld) {
		t.Fatalf("second acquire err = %v, want ErrHeld", err)
	}
	if ok, err := l.Check(ctx); !ok || err != nil {
		t.Fatalf("Check = %v, %v", ok, err)
	}
	if err := l.Release(ctx); err != nil {
		t.Fatal(err)
	}
	if err := l.Release(ctx); err != nil {
		t.Errorf("second release = %v", err)
	}
	if l.Valid() {
		t.Error("released lease still valid")
	}
	select {
	case <-l.Done():
	default:
		t.Error("Done not closed after release")
	}
	if err := l.Renew(ctx); err == nil {
		t.Error("renewed a released lease")
	}

	l2, err := m.Acquire(ctx, key, 5*time.Second)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	defer l2.Release(ctx)
	if l2.Fence() <= l.Fence() || l2.Token() == l.Token() {
		t.Errorf("fence %d -> %d, tokens %s %s", l.Fence(), l2.Fence(), l.Token(), l2.Token())
	}
}

func TestConcurrentAcquireHasOneWinner(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()
	key := newKey()

	var wg sync.WaitGroup
	var mu sync.Mutex
	var won []*lease.Lease
	held := 0
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l, err := m.Acquire(ctx, key, 5*time.Second)
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				won = append(won, l)
			case errors.Is(err, lease.ErrHeld):
				held++
			default:
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if len(won) != 1 || held != 19 {
		t.Fatalf("%d winners, %d ErrHeld; want 1 and 19", len(won), held)
	}
	_ = won[0].Release(ctx)
}

// An expired lease can be taken over, and the old holder can neither
// renew nor release the new holder's lease.
func TestExpiryAndTakeover(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()
	key := newKey()

	old, err := m.Acquire(ctx, key, 150*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(250 * time.Millisecond)
	if old.Valid() {
		t.Error("lease valid past its deadline")
	}
	cur, err := m.Acquire(ctx, key, 5*time.Second)
	if err != nil {
		t.Fatalf("acquire after expiry: %v", err)
	}
	defer cur.Release(ctx)
	if cur.Fence() <= old.Fence() {
		t.Errorf("fence did not increase: %d -> %d", old.Fence(), cur.Fence())
	}

	if err := old.Renew(ctx); !errors.Is(err, lease.ErrLost) {
		t.Errorf("stale renew err = %v, want ErrLost", err)
	}
	if !errors.Is(old.Err(), lease.ErrLost) {
		t.Errorf("stale Err = %v", old.Err())
	}
	if err := old.Release(ctx); !errors.Is(err, lease.ErrLost) {
		t.Errorf("stale release err = %v, want ErrLost", err)
	}
	if ok, err := cur.Check(ctx); !ok || err != nil {
		t.Fatalf("stale holder disturbed the new lease: %v, %v", ok, err)
	}
	if _, err := m.Acquire(ctx, key, time.Second); !errors.Is(err, lease.ErrHeld) {
		t.Errorf("acquire while held = %v", err)
	}
}

func TestRenewExtends(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()
	key := newKey()
	l, err := m.Acquire(ctx, key, 300*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Release(ctx)
	for i := 0; i < 6; i++ { // 900ms in all, three TTLs
		time.Sleep(150 * time.Millisecond)
		before := l.Deadline()
		if err := l.Renew(ctx); err != nil {
			t.Fatalf("renew %d: %v", i, err)
		}
		if !l.Deadline().After(before) {
			t.Error("deadline did not move")
		}
	}
	if _, err := m.Acquire(ctx, key, time.Second); !errors.Is(err, lease.ErrHeld) {
		t.Fatalf("renewed lease was taken: %v", err)
	}
}

func TestKeepAlive(t *testing.T) {
	rdb := redistest.Client(t)
	m := lease.NewManager(rdb)
	ctx := context.Background()
	key := newKey()

	l, err := m.Acquire(ctx, key, 300*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	kctx, stop := context.WithCancel(ctx)
	defer stop()
	lost := l.KeepAlive(kctx)

	time.Sleep(time.Second) // over three TTLs
	if _, err := m.Acquire(ctx, key, time.Second); !errors.Is(err, lease.ErrHeld) {
		t.Fatalf("kept-alive lease was taken: %v", err)
	}
	if !l.Valid() {
		t.Fatal("kept-alive lease not valid")
	}

	// Someone else takes the key: KeepAlive reports the loss.
	if err := rdb.Set(ctx, "lease:{"+key+"}", "intruder", time.Second).Err(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-lost:
		if !errors.Is(err, lease.ErrLost) {
			t.Fatalf("KeepAlive reported %v, want ErrLost", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("loss not reported")
	}
	if _, open := <-lost; open {
		t.Error("channel not closed after the loss")
	}
	select {
	case <-l.Done():
	default:
		t.Error("Done not closed after loss")
	}
	if got, _ := rdb.Get(ctx, "lease:{"+key+"}").Result(); got != "intruder" {
		t.Errorf("lost lease touched the new value: %q", got)
	}
}

func TestKeepAliveStopsCleanly(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()

	// On ctx cancel: closes without an error.
	l, err := m.Acquire(ctx, newKey(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	kctx, stop := context.WithCancel(ctx)
	ch := l.KeepAlive(kctx)
	stop()
	select {
	case err, open := <-ch:
		if open {
			t.Fatalf("cancel reported %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("KeepAlive did not stop on cancel")
	}
	_ = l.Release(ctx)

	// On release: closes without an error.
	l, err = m.Acquire(ctx, newKey(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	ch = l.KeepAlive(ctx)
	if err := l.Release(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case err, open := <-ch:
		if open {
			t.Fatalf("release reported %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("KeepAlive did not stop on release")
	}
}

// Fencing tokens increase across every acquisition of a key, including
// takeovers after expiry, so a store can reject a stale holder's writes.
func TestFencingTokensIncrease(t *testing.T) {
	m := lease.NewManager(redistest.Client(t))
	ctx := context.Background()
	key := newKey()
	var last int64
	for i := 0; i < 5; i++ {
		l, err := m.Acquire(ctx, key, 50*time.Millisecond)
		if err != nil {
			t.Fatal(err)
		}
		if l.Fence() <= last {
			t.Fatalf("fence %d after %d", l.Fence(), last)
		}
		last = l.Fence()
		if i%2 == 0 {
			_ = l.Release(ctx)
		} else {
			time.Sleep(80 * time.Millisecond) // let it expire
		}
	}
	other, err := m.Acquire(ctx, newKey(), time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Release(ctx)
	if other.Fence() != 1 {
		t.Errorf("a new key's first fence = %d, want 1", other.Fence())
	}
}
