// Package redisbus is the Redis events.Bus: each session's events are
// published as JSON on the pub/sub channel session:<id>.
//
// Every Subscribe gets its own pub/sub connection, and the channel it
// returns is reconnect-safe in the sense the event log needs. A
// subscription never silently skips events. If its connection breaks,
// stops answering health-check pings, or the subscriber falls so far
// behind that its buffer fills, the channel is closed instead. The caller
// (events.Replay) then resubscribes and reads what it missed from the
// Store. go-redis's own transparent reconnect would hide exactly that gap.
package redisbus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/events"
)

// Defaults for a Bus's zero fields.
const (
	// DefaultBuffer is the per-subscriber channel size.
	DefaultBuffer = 1024
	// DefaultHealthCheck is how long a subscription waits without traffic
	// before pinging. A ping unanswered for another interval marks the
	// connection dead.
	DefaultHealthCheck = 15 * time.Second
	// DefaultSubscribeTimeout bounds the wait for Redis to confirm a
	// subscription.
	DefaultSubscribeTimeout = 10 * time.Second
)

// Channel returns the pub/sub channel of a session.
func Channel(sessionID string) string { return "session:" + sessionID }

// Bus is an events.Bus on Redis pub/sub.
type Bus struct {
	rdb redis.UniversalClient

	// Buffer is the per-subscriber channel size; 0 means DefaultBuffer.
	Buffer int
	// HealthCheck is the idle time before a subscription pings; 0 means
	// DefaultHealthCheck.
	HealthCheck time.Duration
	// SubscribeTimeout bounds Subscribe's wait for confirmation; 0 means
	// DefaultSubscribeTimeout.
	SubscribeTimeout time.Duration
}

var _ events.Bus = (*Bus)(nil)

// New returns a Bus on rdb. The Bus does not own rdb.
func New(rdb redis.UniversalClient) *Bus { return &Bus{rdb: rdb} }

// Publish implements events.Bus.
func (b *Bus) Publish(ctx context.Context, e events.Event) error {
	data, err := encode(e)
	if err != nil {
		return err
	}
	if err := b.rdb.Publish(ctx, Channel(e.SessionID), data).Err(); err != nil {
		return fmt.Errorf("redisbus: publish: %w", err)
	}
	return nil
}

// Subscribe implements events.Bus. It returns once Redis has confirmed
// the subscription, so every event published after it returns is
// delivered unless the channel closes first. The returned func and
// cancelling ctx both end the subscription; the func returns after the
// channel is closed.
func (b *Bus) Subscribe(ctx context.Context, sessionID string) (<-chan events.Event, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	ps := b.rdb.Subscribe(ctx, Channel(sessionID))
	if err := b.confirm(ctx, ps); err != nil {
		_ = ps.Close()
		return nil, nil, fmt.Errorf("redisbus: subscribe %s: %w", Channel(sessionID), err)
	}

	size := b.Buffer
	if size <= 0 {
		size = DefaultBuffer
	}
	health := b.HealthCheck
	if health <= 0 {
		health = DefaultHealthCheck
	}
	s := &subscription{
		ps:        ps,
		sessionID: sessionID,
		ch:        make(chan events.Event, size),
		stopping:  make(chan struct{}),
		exited:    make(chan struct{}),
	}
	go s.receive(ctx, health)
	go func() {
		select {
		case <-ctx.Done():
			s.stop()
		case <-s.exited:
		}
	}()
	return s.ch, s.cancel, nil
}

// confirm waits for the subscribe reply.
func (b *Bus) confirm(ctx context.Context, ps *redis.PubSub) error {
	timeout := b.SubscribeTimeout
	if timeout <= 0 {
		timeout = DefaultSubscribeTimeout
	}
	if dl, ok := ctx.Deadline(); ok {
		timeout = min(timeout, time.Until(dl))
	}
	deadline := time.Now().Add(timeout)
	for {
		left := time.Until(deadline)
		if left <= 0 {
			return errors.New("timed out waiting for confirmation")
		}
		msg, err := ps.ReceiveTimeout(ctx, left)
		if err != nil {
			return err
		}
		if sub, ok := msg.(*redis.Subscription); ok && sub.Kind == "subscribe" {
			return nil
		}
	}
}

type subscription struct {
	ps        *redis.PubSub
	sessionID string
	ch        chan events.Event

	stopOnce sync.Once
	stopping chan struct{} // closed when the subscription is ending
	exited   chan struct{} // closed after ch is closed
}

// stop ends the subscription without waiting. Closing the PubSub unblocks
// a pending receive.
func (s *subscription) stop() {
	s.stopOnce.Do(func() {
		close(s.stopping)
		_ = s.ps.Close()
	})
}

// cancel is the func Subscribe returns: stop, then wait for ch to close.
func (s *subscription) cancel() {
	s.stop()
	<-s.exited
}

func (s *subscription) receive(ctx context.Context, health time.Duration) {
	defer close(s.exited)
	defer close(s.ch)
	defer s.stop()

	pinged := false
	for {
		msg, err := s.ps.ReceiveTimeout(ctx, health)
		select {
		case <-s.stopping:
			return
		default:
		}
		if err != nil {
			if isTimeout(err) && !pinged && s.ps.Ping(ctx) == nil {
				pinged = true
				continue
			}
			return // broken connection, or no pong within the interval
		}
		pinged = false
		m, ok := msg.(*redis.Message)
		if !ok {
			continue // pong or subscription notice
		}
		ev, err := decode(m.Payload)
		if err != nil || ev.SessionID != s.sessionID {
			continue // not ours; never let a bad message end the stream
		}
		select {
		case s.ch <- ev:
		default:
			// The subscriber fell behind. Close rather than drop, so it
			// resynchronises from the Store.
			return
		}
	}
}

func isTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// encode marshals e with its payload bytes copied verbatim. Going through
// encoding/json would re-escape the payload (U+2028, and <, >, & unless
// disabled), and then a live event would differ from the same event read
// from the Store.
func encode(e events.Event) ([]byte, error) {
	payload := []byte(e.Payload)
	if len(payload) == 0 {
		payload = []byte("null")
	} else if !json.Valid(payload) {
		return nil, errors.New("redisbus: encode event: payload is not valid JSON")
	}
	head, err := json.Marshal(struct {
		SessionID string    `json:"session_id"`
		Seq       int64     `json:"seq"`
		TS        time.Time `json:"ts"`
		Type      string    `json:"type"`
	}{e.SessionID, e.Seq, e.TS, e.Type})
	if err != nil {
		return nil, fmt.Errorf("redisbus: encode event: %w", err)
	}
	out := make([]byte, 0, len(head)+len(payload)+12)
	out = append(out, head[:len(head)-1]...) // drop '}'
	out = append(out, `,"payload":`...)
	out = append(out, payload...)
	return append(out, '}'), nil
}

func decode(s string) (events.Event, error) {
	var e events.Event
	err := json.Unmarshal([]byte(s), &e)
	return e, err
}
