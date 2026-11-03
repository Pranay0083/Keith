package lease

import (
	"context"
	"errors"
	"fmt"
	"sync"
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

// scriptedRedis is a redis.Hook that answers the lease's commands from
// one in-memory key instead of a server, so tests can stop a script
// half-way and interleave Renew, Check and Release deterministically.
// Hooks run before a connection is taken, so no server is contacted.
type scriptedRedis struct {
	mu  sync.Mutex
	val string // the lock key's value; "" when the key is absent
	// onScript, if set, runs each script: apply performs its effect and
	// returns its result. nil applies it at once.
	onScript func(name string, apply func() int64) int64
	// down, if set, fails every script without running it, like an
	// unreachable server.
	down error
}

func newScriptedLease(t *testing.T, ttl time.Duration) (*Lease, *scriptedRedis) {
	t.Helper()
	fr := &scriptedRedis{val: "tok"}
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", MaxRetries: -1})
	rdb.AddHook(fr)
	t.Cleanup(func() { _ = rdb.Close() })
	return &Lease{rdb: rdb, key: "k", token: "tok", ttl: ttl, deadline: time.Now().Add(ttl), done: make(chan struct{})}, fr
}

func (f *scriptedRedis) DialHook(next redis.DialHook) redis.DialHook { return next }

func (f *scriptedRedis) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(context.Context, []redis.Cmder) error { return errors.New("scriptedRedis: no pipelines") }
}

func (f *scriptedRedis) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		args := cmd.Args()
		switch c := cmd.(type) {
		case *redis.StringCmd: // GET
			f.mu.Lock()
			v := f.val
			f.mu.Unlock()
			if v == "" {
				return redis.Nil
			}
			c.SetVal(v)
			return nil
		case *redis.Cmd: // EVALSHA hash numkeys key token [ttl]
			if len(args) < 5 || args[0] != "evalsha" {
				return fmt.Errorf("scriptedRedis: unexpected %v", args)
			}
			if f.down != nil {
				return f.down
			}
			token := args[4]
			var name string
			var apply func() int64
			switch args[1] {
			case renewScript.Hash():
				name = "renew"
				apply = func() int64 {
					f.mu.Lock()
					defer f.mu.Unlock()
					if f.val == token {
						return 1
					}
					return 0
				}
			case releaseScript.Hash():
				name = "release"
				apply = func() int64 {
					f.mu.Lock()
					defer f.mu.Unlock()
					if f.val == token {
						f.val = ""
						return 1
					}
					return 0
				}
			default:
				return fmt.Errorf("scriptedRedis: unexpected script %v", args)
			}
			n := apply
			if f.onScript != nil {
				n = func() int64 { return f.onScript(name, apply) }
			}
			c.SetVal(n())
			return nil
		}
		return fmt.Errorf("scriptedRedis: unexpected %v", args)
	}
}

// A renewal (or check) that finds the key gone because Release deleted it
// is not a loss: Release's own result decides.
func TestRenewRacingReleaseIsNotALoss(t *testing.T) {
	ctx := context.Background()
	l, fr := newScriptedLease(t, time.Minute)
	scriptDone, finish := make(chan struct{}), make(chan struct{})
	fr.onScript = func(name string, apply func() int64) int64 {
		n := apply()
		if name == "release" {
			close(scriptDone) // the key is gone; Release has not returned
			<-finish
		}
		return n
	}
	released := make(chan error, 1)
	go func() { released <- l.Release(ctx) }()
	<-scriptDone

	if err := l.Renew(ctx); err == nil || errors.Is(err, ErrLost) {
		t.Errorf("Renew during Release = %v, want the released error", err)
	}
	if ok, err := l.Check(ctx); ok || err != nil {
		t.Errorf("Check during Release = %v, %v", ok, err)
	}
	if err := l.Err(); err != nil {
		t.Errorf("Err during Release = %v, want nil until Release ends the lease", err)
	}
	close(finish)
	if err := <-released; err != nil {
		t.Fatalf("Release = %v", err)
	}
	if err := l.Err(); !errors.Is(err, errReleased) {
		t.Errorf("Err after Release = %v, want released", err)
	}
}

// KeepAlive's renewal in flight while Release runs: KeepAlive stops
// without reporting a loss.
func TestKeepAliveRacingReleaseStopsQuietly(t *testing.T) {
	ctx := context.Background()
	l, fr := newScriptedLease(t, 300*time.Millisecond)
	inFlight, resume := make(chan struct{}), make(chan struct{})
	var once sync.Once
	fr.onScript = func(name string, apply func() int64) int64 {
		if name == "renew" {
			once.Do(func() {
				close(inFlight)
				<-resume // Release runs to the end meanwhile
			})
		}
		return apply()
	}
	ch := l.KeepAlive(ctx)
	<-inFlight
	if err := l.Release(ctx); err != nil {
		t.Fatalf("Release = %v", err)
	}
	close(resume)
	select {
	case err, open := <-ch:
		if open {
			t.Fatalf("KeepAlive reported %v after a clean release", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("KeepAlive did not stop")
	}
	if err := l.Err(); !errors.Is(err, errReleased) {
		t.Errorf("Err = %v, want released", err)
	}
}

// A failed Release (Redis unreachable) leaves the lease as it was: a
// later loss is still a loss.
func TestFailedReleaseLeavesTheLeaseHeld(t *testing.T) {
	ctx := context.Background()
	l, fr := newScriptedLease(t, time.Minute)
	fr.down = errors.New("connection refused")
	if err := l.Release(ctx); err == nil || errors.Is(err, ErrLost) {
		t.Fatalf("Release with Redis down = %v", err)
	}
	if err := l.Err(); err != nil {
		t.Fatalf("Err after a failed Release = %v", err)
	}
	fr.down = nil
	fr.mu.Lock()
	fr.val = "someone-else"
	fr.mu.Unlock()
	if err := l.Renew(ctx); !errors.Is(err, ErrLost) {
		t.Errorf("Renew after takeover = %v, want ErrLost", err)
	}
	if !errors.Is(l.Err(), ErrLost) {
		t.Errorf("Err = %v, want ErrLost", l.Err())
	}
}
