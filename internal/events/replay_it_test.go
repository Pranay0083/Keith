package events_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/redistest"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

// Integration tests of the whole event path on Postgres and Redis:
// KEITH_IT=1, against DATABASE_URL and REDIS_URL.

type rig struct {
	store events.Store
	bus   *redisbus.Bus
	em    *events.Emitter
	rdb   *redis.Client
	sid   string
}

// newRig seeds a session and takes its lease, as the worker does before
// it writes events.
func newRig(t *testing.T, clientName string) *rig {
	t.Helper()
	s := storetest.New(t)
	f := storetest.Seed(t, s)
	redistest.Client(t) // skips without KEITH_IT, checks Redis is up
	opt, err := redis.ParseURL(redistest.RedisURL())
	if err != nil {
		t.Fatal(err)
	}
	opt.ClientName = clientName
	rdb := redis.NewClient(opt)
	t.Cleanup(func() { _ = rdb.Close() })

	l, err := lease.NewManager(rdb).Acquire(context.Background(), lease.SessionKey(f.Session.ID), 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = l.Release(context.Background()) })

	es := s.Events()
	bus := redisbus.New(rdb)
	return &rig{store: es, bus: bus, em: &events.Emitter{Store: es, Bus: bus}, rdb: rdb, sid: f.Session.ID}
}

func (r *rig) emit(t *testing.T, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if _, err := r.em.Emit(context.Background(), r.sid, events.TypeToolCallOutput,
			events.ToolCallOutputPayload{CallID: "c", Chunk: fmt.Sprintf("line %d <ok> &  \n", i)}); err != nil {
			t.Fatal(err)
		}
	}
}

func randName() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return "keith-b3-it-" + hex.EncodeToString(b)
}

// The reconnect case the UI depends on: events are appended while a
// subscriber is disconnected; reconnecting with the last seq it saw yields
// exactly the missing events, in order, with no duplicates, and then live
// events.
func TestReconnectReplayIT(t *testing.T) {
	r := newRig(t, randName())
	r.emit(t, 5)

	ctx1, cancel1 := context.WithCancel(context.Background())
	ch1, err := events.Replay(ctx1, r.store, r.bus, r.sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := recv(t, ch1, 5)
	r.emit(t, 3) // live
	got = append(got, recv(t, ch1, 3)...)
	wantSeqs(t, got, 1, 8)
	last := got[len(got)-1].Seq
	cancel1()
	for range ch1 {
	}

	// Disconnected: seven stored events and a delta nobody sees.
	r.emit(t, 7)
	if _, err := r.em.Emit(context.Background(), r.sid, events.TypeMessageDelta,
		events.MessageDeltaPayload{MessageID: "m", Text: "gone"}); err != nil {
		t.Fatal(err)
	}

	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	ch2, err := events.Replay(ctx2, r.store, r.bus, r.sid, last)
	if err != nil {
		t.Fatal(err)
	}
	missed := recv(t, ch2, 7)
	wantSeqs(t, missed, 9, 15)
	quiet(t, ch2, 200*time.Millisecond)

	r.emit(t, 3)
	live := recv(t, ch2, 3)
	wantSeqs(t, live, 16, 18)
	quiet(t, ch2, 200*time.Millisecond)

	// Everything the client saw is byte-identical to the stored log.
	stored, err := r.store.After(context.Background(), r.sid, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	seen := append(append(got, missed...), live...)
	if len(seen) != len(stored) {
		t.Fatalf("saw %d events, stored %d", len(seen), len(stored))
	}
	for i := range stored {
		if seen[i].Seq != stored[i].Seq || seen[i].Type != stored[i].Type || !seen[i].TS.Equal(stored[i].TS) ||
			string(seen[i].Payload) != string(stored[i].Payload) {
			t.Fatalf("event %d: saw %+v, stored %+v", i, seen[i], stored[i])
		}
	}
}

// A client that keeps disconnecting and reconnecting with its last seq,
// while the worker writes without pause, sees every event exactly once in
// order.
func TestReconnectReplayUnderLoadIT(t *testing.T) {
	r := newRig(t, randName())
	const total = 300

	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		for i := 0; i < total; i++ {
			if _, err := r.em.Emit(context.Background(), r.sid, events.TypeToolCallOutput,
				events.ToolCallOutputPayload{CallID: "c", Chunk: fmt.Sprint(i)}); err != nil {
				t.Error(err)
				return
			}
			if i%10 == 0 {
				_, _ = r.em.Emit(context.Background(), r.sid, events.TypeMessageDelta,
					events.MessageDeltaPayload{MessageID: "m", Text: "x"})
			}
			time.Sleep(time.Millisecond)
		}
	}()

	var seen []int64
	var last int64
	deadline := time.Now().Add(30 * time.Second)
	for reconnects := 0; last < total; reconnects++ {
		if time.Now().After(deadline) {
			t.Fatalf("stuck at seq %d after %d reconnects", last, reconnects)
		}
		ctx, cancel := context.WithCancel(context.Background())
		ch, err := events.Replay(ctx, r.store, r.bus, r.sid, last)
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		// Read a varying number of stored events, then drop the connection.
		budget := 7 + reconnects%23
		timeout := time.After(2 * time.Second)
	read:
		for budget > 0 {
			select {
			case ev, ok := <-ch:
				if !ok {
					break read
				}
				if ev.Seq == 0 {
					continue
				}
				seen = append(seen, ev.Seq)
				last = ev.Seq
				budget--
			case <-timeout:
				break read
			}
		}
		cancel()
		for range ch {
		}
	}
	<-writerDone
	if len(seen) != total {
		t.Fatalf("saw %d stored events, want %d", len(seen), total)
	}
	for i, s := range seen {
		if s != int64(i+1) {
			t.Fatalf("event %d has seq %d: duplicates or gaps", i, s)
		}
	}
}

// The server-side case on real Redis: the subscription's connection is
// killed under a live Replay while the worker keeps writing; Replay
// resubscribes and backfills, so the client sees no gap and no duplicate.
func TestReplaySurvivesRedisConnectionKillIT(t *testing.T) {
	name := randName()
	r := newRig(t, name)
	admin := redistest.Client(t)
	r.emit(t, 3)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := events.Replay(ctx, r.store, r.bus, r.sid, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantSeqs(t, recv(t, ch, 3), 1, 3)

	var wg sync.WaitGroup
	killed := 0
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 3; i++ {
			killed += killPubSub(t, admin, name)
			time.Sleep(30 * time.Millisecond)
		}
	}()
	r.emit(t, 40) // some land while the subscription is down
	wg.Wait()
	if killed == 0 {
		t.Fatal("no pub/sub connection was killed")
	}
	r.emit(t, 2)

	wantSeqs(t, recv(t, ch, 42), 4, 45)
	quiet(t, ch, 200*time.Millisecond)
}

// killPubSub kills the pub/sub connections named name and returns how
// many it killed.
func killPubSub(t *testing.T, admin *redis.Client, name string) int {
	list, err := admin.ClientList(context.Background()).Result()
	if err != nil {
		t.Error(err)
		return 0
	}
	n := 0
	for _, line := range strings.Split(list, "\n") {
		f := map[string]string{}
		for _, kv := range strings.Fields(line) {
			k, v, _ := strings.Cut(kv, "=")
			f[k] = v
		}
		if f["name"] == name && f["sub"] != "0" && f["sub"] != "" {
			if admin.Do(context.Background(), "CLIENT", "KILL", "ID", f["id"]).Err() == nil {
				n++
			}
		}
	}
	return n
}
