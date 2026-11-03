// Package memevents is an in-memory events.Store and events.Bus for tests
// and single-process development. Nothing is persisted.
package memevents

import (
	"context"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/events"
)

// Store is an in-memory events.Store. The zero value is ready to use.
type Store struct {
	// Now returns event timestamps; nil means time.Now.
	Now func() time.Time

	mu  sync.Mutex
	log map[string][]events.Event
}

var _ events.Store = (*Store)(nil)

// NewStore returns an empty Store.
func NewStore() *Store { return &Store{} }

// Append implements events.Store.
func (s *Store) Append(ctx context.Context, sessionID, typ string, payload any) (events.Event, error) {
	if err := ctx.Err(); err != nil {
		return events.Event{}, err
	}
	if events.IsEphemeral(typ) {
		return events.Event{}, events.ErrEphemeral
	}
	raw, err := events.MarshalPayload(payload)
	if err != nil {
		return events.Event{}, err
	}
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.log == nil {
		s.log = make(map[string][]events.Event)
	}
	ev := events.Event{
		SessionID: sessionID,
		Seq:       int64(len(s.log[sessionID])) + 1,
		TS:        now().UTC(),
		Type:      typ,
		Payload:   raw,
	}
	s.log[sessionID] = append(s.log[sessionID], ev)
	return ev, nil
}

// After implements events.Store.
func (s *Store) After(ctx context.Context, sessionID string, afterSeq int64, limit int) ([]events.Event, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []events.Event
	for _, ev := range s.log[sessionID] {
		if ev.Seq <= afterSeq {
			continue
		}
		out = append(out, ev)
		if limit > 0 && len(out) == limit {
			break
		}
	}
	return out, nil
}

// All returns every stored event of a session, for test assertions.
func (s *Store) All(sessionID string) []events.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]events.Event(nil), s.log[sessionID]...)
}

// Types returns the types of a session's stored events in order, for test
// assertions.
func (s *Store) Types(sessionID string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.log[sessionID]))
	for _, ev := range s.log[sessionID] {
		out = append(out, ev.Type)
	}
	return out
}

// DefaultBuffer is the per-subscriber channel size of a Bus.
const DefaultBuffer = 1024

// Bus is an in-memory events.Bus. Each subscriber gets a buffered
// channel; when it is full, further events for that subscriber are
// dropped, matching the best-effort contract. The zero value is ready to
// use.
type Bus struct {
	// Buffer is the per-subscriber channel size; 0 means DefaultBuffer.
	Buffer int

	mu     sync.Mutex
	nextID int
	subs   map[string]map[int]chan events.Event
}

var _ events.Bus = (*Bus)(nil)

// NewBus returns an empty Bus.
func NewBus() *Bus { return &Bus{} }

// Publish implements events.Bus.
func (b *Bus) Publish(ctx context.Context, e events.Event) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, ch := range b.subs[e.SessionID] {
		select {
		case ch <- e:
		default: // subscriber is behind; it recovers by replaying
		}
	}
	return nil
}

// Subscribe implements events.Bus.
func (b *Bus) Subscribe(ctx context.Context, sessionID string) (<-chan events.Event, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	size := b.Buffer
	if size <= 0 {
		size = DefaultBuffer
	}
	ch := make(chan events.Event, size)

	b.mu.Lock()
	if b.subs == nil {
		b.subs = make(map[string]map[int]chan events.Event)
	}
	if b.subs[sessionID] == nil {
		b.subs[sessionID] = make(map[int]chan events.Event)
	}
	id := b.nextID
	b.nextID++
	b.subs[sessionID][id] = ch
	b.mu.Unlock()

	var once sync.Once
	done := make(chan struct{})
	cancel := func() {
		once.Do(func() {
			b.mu.Lock()
			delete(b.subs[sessionID], id)
			if len(b.subs[sessionID]) == 0 {
				delete(b.subs, sessionID)
			}
			b.mu.Unlock()
			// Publish only sends while holding b.mu, so after the delete
			// nothing can send on ch.
			close(ch)
			close(done)
		})
	}
	if ctx.Done() != nil {
		go func() {
			select {
			case <-ctx.Done():
				cancel()
			case <-done:
			}
		}()
	}
	return ch, cancel, nil
}

// Subscribers returns how many live subscriptions a session has.
func (b *Bus) Subscribers(sessionID string) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.subs[sessionID])
}
