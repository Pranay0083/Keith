package events

import (
	"context"
	"fmt"
	"time"
)

// ReplayPageSize is how many stored events Replay reads per Store.After
// call.
const ReplayPageSize = 500

// Replay timing. Resubscribe attempts back off from replayRetryMin to
// replayRetryMax; while the Bus is down, each attempt also polls the Store
// so the client keeps receiving stored events.
const (
	replayRetryMin = 50 * time.Millisecond
	replayRetryMax = 5 * time.Second
	// replayGapRetries bounds how often a gap is re-read from the Store
	// before the live event that revealed it is forwarded anyway.
	replayGapRetries = 3
)

// Replay streams one session's events to one client, the way the
// WebSocket endpoint needs them: every stored event with seq > afterSeq
// in order, then live events, each seq at most once.
//
//  1. It subscribes to the Bus first, so nothing published while it reads
//     the Store is missed.
//  2. It sends stored events after afterSeq from Store.After, page by
//     page.
//  3. It forwards live events, dropping any seq <= the last seq sent.
//     Ephemeral events (seq 0) always pass.
//
// Bus delivery is best-effort, so Replay repairs it from the Store:
//   - A live event that skips ahead (seq > last+1) means the Bus dropped
//     something; Replay reads the gap from the Store first.
//   - If the subscription channel closes while ctx is live (the Bus
//     connection broke, or the subscriber fell behind), Replay
//     resubscribes with backoff, polling the Store while it waits, and
//     then reads everything it missed from the Store.
//
// The first Store page is read before Replay returns, so a Store failure
// is reported as an error. The channel closes when ctx is cancelled, or
// when a later Store read fails; the client should then reconnect with
// the last seq it received. Because live events buffered during the
// initial read are forwarded after it, a message.delta can arrive after
// the message.assistant that completed it; clients ignore deltas for
// messages they already have in full.
func Replay(ctx context.Context, store Store, bus Bus, sessionID string, afterSeq int64) (<-chan Event, error) {
	live, unsub, err := bus.Subscribe(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("events: replay %s: subscribe: %w", sessionID, err)
	}
	first, err := store.After(ctx, sessionID, afterSeq, ReplayPageSize)
	if err != nil {
		unsub()
		return nil, fmt.Errorf("events: replay %s: %w", sessionID, err)
	}
	r := &replayer{
		ctx:       ctx,
		store:     store,
		bus:       bus,
		sessionID: sessionID,
		out:       make(chan Event, 16),
		last:      afterSeq,
		live:      live,
		unsub:     unsub,
	}
	go r.run(first)
	return r.out, nil
}

type replayer struct {
	ctx       context.Context
	store     Store
	bus       Bus
	sessionID string
	out       chan Event
	// last is the highest seq sent.
	last  int64
	live  <-chan Event
	unsub func()
}

func (r *replayer) run(first []Event) {
	defer close(r.out)
	defer func() { r.unsub() }() // r.unsub changes on resubscribe

	if !r.sendAll(first) {
		return
	}
	if len(first) == ReplayPageSize && !r.catchUp() {
		return
	}
	for {
		select {
		case <-r.ctx.Done():
			return
		case ev, ok := <-r.live:
			if !ok {
				if r.ctx.Err() != nil || !r.resubscribe() {
					return
				}
				continue
			}
			if !r.forward(ev) {
				return
			}
		}
	}
}

// forward handles one live event.
func (r *replayer) forward(ev Event) bool {
	switch {
	case ev.Seq == 0:
		return r.send(ev)
	case ev.Seq <= r.last:
		return true // already sent
	case ev.Seq == r.last+1:
		return r.send(ev)
	}
	// The Bus skipped something; the Store has it, since events are
	// stored before they are published.
	for i := 0; i < replayGapRetries && ev.Seq > r.last+1; i++ {
		if i > 0 && !r.wait(replayRetryMin<<i) {
			return false
		}
		if !r.catchUp() {
			return false
		}
	}
	if ev.Seq > r.last {
		return r.send(ev)
	}
	return true
}

// catchUp sends every stored event after r.last. It returns false if ctx
// ended or the Store failed.
func (r *replayer) catchUp() bool {
	for {
		evs, err := r.store.After(r.ctx, r.sessionID, r.last, ReplayPageSize)
		if err != nil {
			return false
		}
		if !r.sendAll(evs) {
			return false
		}
		if len(evs) < ReplayPageSize {
			return true
		}
	}
}

// resubscribe replaces a closed subscription, retrying with backoff and
// polling the Store meanwhile, then catches up on what was missed.
func (r *replayer) resubscribe() bool {
	r.unsub()
	r.unsub = func() {}
	delay := replayRetryMin
	for {
		live, unsub, err := r.bus.Subscribe(r.ctx, r.sessionID)
		if err == nil {
			r.live, r.unsub = live, unsub
			return r.catchUp()
		}
		if !r.catchUp() || !r.wait(delay) {
			return false
		}
		delay = min(delay*2, replayRetryMax)
	}
}

// sendAll sends stored events in order, skipping any already sent.
func (r *replayer) sendAll(evs []Event) bool {
	for _, ev := range evs {
		if ev.Seq <= r.last {
			continue
		}
		if !r.send(ev) {
			return false
		}
	}
	return true
}

// send delivers ev to the client and records its seq.
func (r *replayer) send(ev Event) bool {
	select {
	case r.out <- ev:
		if ev.Seq > r.last {
			r.last = ev.Seq
		}
		return true
	case <-r.ctx.Done():
		return false
	}
}

func (r *replayer) wait(d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.ctx.Done():
		return false
	}
}
