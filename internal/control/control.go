// Package control carries out-of-band signals from the control plane to
// the worker running a session, on the Redis pub/sub channel
// session:<id>:control.
//
// Today the only signal is cancel: Stop (and the installation.deleted
// webhook) set sessions.cancel_requested_at and then publish it, so a
// worker blocked in a long shell command can interrupt it at once instead
// of waiting for its next between-steps check.
//
// Pub/sub is best-effort. A signal published while no worker is
// subscribed, or during a reconnect, is lost. The database flag is the
// source of truth: the worker must keep checking
// store.IsCancelRequested between steps, and treat a signal only as a
// prompt to check now.
//
// Worker usage (wave D):
//
//	sigs, stop, err := ctl.Subscribe(ctx, sessionID)
//	defer stop()
//	go func() {
//		for sig := range sigs {
//			if sig == control.SignalCancel {
//				interruptRunningShell()
//			}
//		}
//	}()
package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/redis/go-redis/v9"
)

// Signal is a control message for a session's worker.
type Signal string

// SignalCancel asks the worker to stop the session now.
const SignalCancel Signal = "cancel"

// Channel returns a session's control channel. It does not collide with
// the event channel session:<id> (internal/events/redisbus), which is an
// exact-name subscription.
func Channel(sessionID string) string { return "session:" + sessionID + ":control" }

// Publisher sends signals.
type Publisher interface {
	Publish(ctx context.Context, sessionID string, sig Signal) error
}

// Subscriber receives signals.
type Subscriber interface {
	// Subscribe returns a channel of the session's signals published
	// after it returns. The func, or cancelling ctx, ends the
	// subscription and closes the channel.
	Subscribe(ctx context.Context, sessionID string) (<-chan Signal, func(), error)
}

// message is the JSON published on the channel.
type message struct {
	Signal Signal `json:"signal"`
}

// Redis is a Publisher and Subscriber on Redis pub/sub.
type Redis struct {
	rdb redis.UniversalClient
}

var (
	_ Publisher  = (*Redis)(nil)
	_ Subscriber = (*Redis)(nil)
)

// NewRedis returns a Redis on rdb. It does not own rdb.
func NewRedis(rdb redis.UniversalClient) *Redis { return &Redis{rdb: rdb} }

// Publish implements Publisher.
func (r *Redis) Publish(ctx context.Context, sessionID string, sig Signal) error {
	if sessionID == "" || sig == "" {
		return errors.New("control: empty session ID or signal")
	}
	b, _ := json.Marshal(message{Signal: sig})
	if err := r.rdb.Publish(ctx, Channel(sessionID), b).Err(); err != nil {
		return fmt.Errorf("control: publish %s: %w", sig, err)
	}
	return nil
}

// Subscribe implements Subscriber. It returns once Redis confirmed the
// subscription. Malformed messages are skipped.
func (r *Redis) Subscribe(ctx context.Context, sessionID string) (<-chan Signal, func(), error) {
	if sessionID == "" {
		return nil, nil, errors.New("control: empty session ID")
	}
	ps := r.rdb.Subscribe(ctx, Channel(sessionID))
	if _, err := ps.Receive(ctx); err != nil {
		_ = ps.Close()
		return nil, nil, fmt.Errorf("control: subscribe: %w", err)
	}
	ctx, cancel := context.WithCancel(ctx)
	out := make(chan Signal, 8)
	in := ps.Channel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer close(out)
		defer ps.Close()
		for {
			select {
			case <-ctx.Done():
				return
			case m, ok := <-in:
				if !ok {
					return
				}
				var msg message
				if json.Unmarshal([]byte(m.Payload), &msg) != nil || msg.Signal == "" {
					continue
				}
				select {
				case out <- msg.Signal:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	stop := func() {
		cancel()
		<-done
	}
	return out, stop, nil
}

// Mem is an in-memory Publisher and Subscriber for tests and
// single-process use. It records every published signal. The zero value
// is ready to use.
type Mem struct {
	mu        sync.Mutex
	subs      map[string]map[int]chan Signal
	next      int
	published []Published
}

// Published is one signal Mem saw.
type Published struct {
	SessionID string
	Signal    Signal
}

var (
	_ Publisher  = (*Mem)(nil)
	_ Subscriber = (*Mem)(nil)
)

// NewMem returns an empty Mem.
func NewMem() *Mem { return &Mem{} }

// Publish implements Publisher. A subscriber whose buffer is full misses
// the signal, as with Redis.
func (m *Mem) Publish(ctx context.Context, sessionID string, sig Signal) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.published = append(m.published, Published{sessionID, sig})
	for _, ch := range m.subs[sessionID] {
		select {
		case ch <- sig:
		default:
		}
	}
	return nil
}

// Subscribe implements Subscriber.
func (m *Mem) Subscribe(ctx context.Context, sessionID string) (<-chan Signal, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	ch := make(chan Signal, 8)
	m.mu.Lock()
	if m.subs == nil {
		m.subs = make(map[string]map[int]chan Signal)
	}
	if m.subs[sessionID] == nil {
		m.subs[sessionID] = make(map[int]chan Signal)
	}
	id := m.next
	m.next++
	m.subs[sessionID][id] = ch
	m.mu.Unlock()

	var once sync.Once
	done := make(chan struct{})
	stop := func() {
		once.Do(func() {
			m.mu.Lock()
			delete(m.subs[sessionID], id)
			m.mu.Unlock()
			// Publish sends only under m.mu, so nothing sends on ch now.
			close(ch)
			close(done)
		})
	}
	go func() {
		select {
		case <-ctx.Done():
			stop()
		case <-done:
		}
	}()
	return ch, stop, nil
}

// Published returns every signal published so far, in order.
func (m *Mem) Published() []Published {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Published(nil), m.published...)
}
