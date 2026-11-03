package lease

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

func TestKeys(t *testing.T) {
	if got := lockKey(SessionKey("abc")); got != "lease:{session:abc}" {
		t.Errorf("lockKey = %q", got)
	}
	if got := fenceKey(SessionKey("abc")); got != "lease:{session:abc}:fence" {
		t.Errorf("fenceKey = %q", got)
	}
}

func TestAcquireValidatesTTLAndReportsRedisErrors(t *testing.T) {
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", MaxRetries: -1, DialTimeout: time.Second})
	defer rdb.Close()
	m := NewManager(rdb)
	if _, err := m.Acquire(context.Background(), "k", time.Millisecond); err == nil {
		t.Error("tiny ttl accepted")
	}
	_, err := m.Acquire(context.Background(), "k", time.Second)
	if err == nil || errors.Is(err, ErrHeld) {
		t.Errorf("err = %v, want a connection error", err)
	}
}

// With Redis unreachable, KeepAlive keeps retrying until the deadline and
// then reports the lease lost: another worker may hold it by then.
func TestKeepAliveReportsLossWhenRedisIsUnreachable(t *testing.T) {
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", MaxRetries: -1, DialTimeout: 50 * time.Millisecond})
	defer rdb.Close()
	ttl := 300 * time.Millisecond
	l := &Lease{rdb: rdb, key: "k", token: "t", ttl: ttl, deadline: time.Now().Add(ttl), done: make(chan struct{})}

	start := time.Now()
	select {
	case err := <-l.KeepAlive(context.Background()):
		if !errors.Is(err, ErrLost) {
			t.Fatalf("err = %v, want ErrLost", err)
		}
		if waited := time.Since(start); waited < ttl/2 {
			t.Errorf("gave up after %v, before the deadline", waited)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("loss not reported")
	}
	if l.Valid() || !errors.Is(l.Err(), ErrLost) {
		t.Errorf("Valid=%v Err=%v after loss", l.Valid(), l.Err())
	}
}
