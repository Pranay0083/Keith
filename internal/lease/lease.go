// Package lease is an exclusive, expiring Redis lease. The worker holds a
// session's lease while it runs the session, which makes it the session's
// only event writer.
//
// Acquire is SET NX PX with a random token. Renew and Release are
// check-and-set Lua scripts that act only while the key still holds that
// token, so a holder whose lease expired can never extend or delete its
// successor's. KeepAlive renews in the background and reports a loss.
//
// A lease can be lost without the holder noticing in time: a long GC
// pause or a network partition can outlast the TTL. Each acquisition
// therefore also gets a fencing token, a per-key counter that only goes
// up. A store that records the highest token it has seen for a key and
// rejects writes carrying a lower one is safe even then.
package lease

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

var (
	// ErrHeld is returned by Acquire when another owner holds the lease.
	ErrHeld = errors.New("lease: held by another owner")
	// ErrLost means the lease expired or was taken over: the key no longer
	// holds this lease's token.
	ErrLost = errors.New("lease: lost")
)

// MinTTL is the shortest lease TTL Acquire accepts.
const MinTTL = 10 * time.Millisecond

// SessionKey returns the lease key for a session.
func SessionKey(sessionID string) string { return "session:" + sessionID }

// Redis keys for a lease key. The hash tag keeps both on one Redis
// Cluster slot, which the acquire script needs.
func lockKey(key string) string  { return "lease:{" + key + "}" }
func fenceKey(key string) string { return "lease:{" + key + "}:fence" }

// acquireScript sets the lock if it is free and returns the next fencing
// token, or 0 if the lock is held. The fence counter never expires, so
// tokens keep increasing across holders.
var acquireScript = redis.NewScript(`
if redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2]) then
	return redis.call('INCR', KEYS[2])
end
return 0`)

// renewScript resets the TTL if the lock still holds the token.
var renewScript = redis.NewScript(`
if redis.call('GET', KEYS[1]) == ARGV[1] then
	return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0`)

// releaseScript deletes the lock if it still holds the token.
var releaseScript = redis.NewScript(`
if redis.call('GET', KEYS[1]) == ARGV[1] then
	return redis.call('DEL', KEYS[1])
end
return 0`)

// Manager acquires leases on one Redis.
type Manager struct {
	rdb redis.UniversalClient
}

// NewManager returns a Manager on rdb. It does not own rdb.
func NewManager(rdb redis.UniversalClient) *Manager { return &Manager{rdb: rdb} }

// Acquire takes the lease on key for ttl. It returns ErrHeld if someone
// else holds it.
func (m *Manager) Acquire(ctx context.Context, key string, ttl time.Duration) (*Lease, error) {
	if ttl < MinTTL {
		return nil, fmt.Errorf("lease: ttl %v is shorter than %v", ttl, MinTTL)
	}
	tok := make([]byte, 16)
	if _, err := rand.Read(tok); err != nil {
		return nil, fmt.Errorf("lease: token: %w", err)
	}
	l := &Lease{
		rdb:   m.rdb,
		key:   key,
		token: hex.EncodeToString(tok),
		ttl:   ttl,
		done:  make(chan struct{}),
	}
	start := time.Now()
	fence, err := acquireScript.Run(ctx, m.rdb, []string{lockKey(key), fenceKey(key)},
		l.token, ttl.Milliseconds()).Int64()
	if err != nil {
		return nil, fmt.Errorf("lease: acquire %s: %w", key, err)
	}
	if fence == 0 {
		return nil, fmt.Errorf("lease: acquire %s: %w", key, ErrHeld)
	}
	l.fence = fence
	l.deadline = start.Add(ttl)
	return l, nil
}

// Lease is a held lease. Its methods are safe for concurrent use.
type Lease struct {
	rdb   redis.UniversalClient
	key   string
	token string
	fence int64
	ttl   time.Duration

	mu       sync.Mutex
	deadline time.Time
	err      error // why the lease ended: ErrLost (wrapped) or errReleased
	done     chan struct{}
}

var errReleased = errors.New("lease: released")

// Key returns the lease key passed to Acquire.
func (l *Lease) Key() string { return l.key }

// Token returns the random token that identifies this holder.
func (l *Lease) Token() string { return l.token }

// Fence returns the fencing token: larger than that of every earlier
// holder of the key.
func (l *Lease) Fence() int64 { return l.fence }

// TTL returns the lease duration.
func (l *Lease) TTL() time.Duration { return l.ttl }

// Deadline returns when the lease expires unless renewed, measured from
// before the last successful acquire or renew was sent, so it is never
// later than Redis's own expiry (clock drift aside).
func (l *Lease) Deadline() time.Time {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.deadline
}

// Valid reports whether the lease is neither known to be lost or released
// nor past its Deadline. It does not contact Redis; Check does.
func (l *Lease) Valid() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err == nil && time.Now().Before(l.deadline)
}

// Done is closed when the lease is known to be lost, or is released.
func (l *Lease) Done() <-chan struct{} { return l.done }

// Err returns nil while the lease is held, an error wrapping ErrLost once
// it is known lost, or a "released" error after Release.
func (l *Lease) Err() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err
}

// end records why the lease ended and closes Done, once.
func (l *Lease) end(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.err == nil {
		l.err = err
		close(l.done)
	}
}

// Renew resets the lease's TTL. It returns an error wrapping ErrLost if the
// key no longer holds this lease's token, after which the lease is dead.
// Other errors (Redis unreachable) leave the lease as it was.
func (l *Lease) Renew(ctx context.Context) error {
	if err := l.Err(); err != nil {
		return err
	}
	start := time.Now()
	n, err := renewScript.Run(ctx, l.rdb, []string{lockKey(l.key)}, l.token, l.ttl.Milliseconds()).Int64()
	if err != nil {
		return fmt.Errorf("lease: renew %s: %w", l.key, err)
	}
	if n == 0 {
		err := fmt.Errorf("lease: renew %s: %w", l.key, ErrLost)
		l.end(err)
		return err
	}
	l.mu.Lock()
	l.deadline = start.Add(l.ttl)
	l.mu.Unlock()
	return nil
}

// Check asks Redis whether the key still holds this lease's token. A no
// marks the lease lost.
func (l *Lease) Check(ctx context.Context) (bool, error) {
	if l.Err() != nil {
		return false, nil
	}
	v, err := l.rdb.Get(ctx, lockKey(l.key)).Result()
	if errors.Is(err, redis.Nil) {
		v, err = "", nil
	}
	if err != nil {
		return false, fmt.Errorf("lease: check %s: %w", l.key, err)
	}
	if v != l.token {
		l.end(fmt.Errorf("lease: check %s: %w", l.key, ErrLost))
		return false, nil
	}
	return true, nil
}

// Release gives the lease up and stops KeepAlive. It returns an error
// wrapping ErrLost if the lease had already been lost; a successor's
// lease is never touched. Releasing twice is a no-op.
func (l *Lease) Release(ctx context.Context) error {
	if err := l.Err(); err != nil {
		if errors.Is(err, errReleased) {
			return nil
		}
		return err
	}
	n, err := releaseScript.Run(ctx, l.rdb, []string{lockKey(l.key)}, l.token).Int64()
	if err != nil {
		return fmt.Errorf("lease: release %s: %w", l.key, err)
	}
	l.end(errReleased)
	if n == 0 {
		return fmt.Errorf("lease: release %s: %w", l.key, ErrLost)
	}
	return nil
}

// KeepAlive renews the lease every TTL/3 until ctx ends or the lease is
// released or lost. If the lease is lost, it sends one error wrapping
// ErrLost on the returned channel; the channel is closed when KeepAlive
// stops, for whatever reason.
//
// Transient renewal failures are retried until the lease's Deadline
// passes; then the lease counts as lost even if Redis comes back, since
// another worker may already hold it. Stop work as soon as a loss is
// reported, or as soon as Valid is false.
func (l *Lease) KeepAlive(ctx context.Context) <-chan error {
	out := make(chan error, 1)
	go func() {
		defer close(out)
		interval := l.ttl / 3
		retry := min(interval, 100*time.Millisecond)
		wait := interval
		for {
			t := time.NewTimer(wait)
			select {
			case <-ctx.Done():
				t.Stop()
				return
			case <-l.done:
				t.Stop()
				if err := l.Err(); errors.Is(err, ErrLost) {
					out <- err
				}
				return
			case <-t.C:
			}

			deadline := l.Deadline()
			if !time.Now().Before(deadline) {
				err := fmt.Errorf("lease: %s expired before it could be renewed: %w", l.key, ErrLost)
				l.end(err)
				out <- err
				return
			}
			rctx, cancel := context.WithDeadline(ctx, deadline)
			err := l.Renew(rctx)
			cancel()
			switch {
			case err == nil:
				wait = interval
			case errors.Is(err, ErrLost):
				out <- err
				return
			case ctx.Err() != nil:
				return
			default:
				wait = retry // transient; try again before the deadline
			}
		}
	}()
	return out
}
