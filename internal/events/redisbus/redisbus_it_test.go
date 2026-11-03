package redisbus_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/redistest"
)

func randID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func ev(session string, seq int64) events.Event {
	return events.Event{
		SessionID: session, Seq: seq, TS: time.Now().UTC().Truncate(time.Microsecond),
		Type: events.TypeToolCallOutput, Payload: json.RawMessage(fmt.Sprintf(`{"call_id":"c","chunk":"%d"}`, seq)),
	}
}

func recvN(t *testing.T, ch <-chan events.Event, n int) []events.Event {
	t.Helper()
	var out []events.Event
	timeout := time.After(5 * time.Second)
	for len(out) < n {
		select {
		case e, ok := <-ch:
			if !ok {
				t.Fatalf("closed after %d of %d", len(out), n)
			}
			out = append(out, e)
		case <-timeout:
			t.Fatalf("timed out after %d of %d", len(out), n)
		}
	}
	return out
}

func waitClosed(t *testing.T, ch <-chan events.Event, d time.Duration) int {
	t.Helper()
	n := 0
	timeout := time.After(d)
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return n
			}
			n++
		case <-timeout:
			t.Fatalf("channel not closed within %v", d)
		}
	}
}

func numSub(t *testing.T, rdb *redis.Client, session string) int64 {
	t.Helper()
	m, err := rdb.PubSubNumSub(context.Background(), redisbus.Channel(session)).Result()
	if err != nil {
		t.Fatal(err)
	}
	return m[redisbus.Channel(session)]
}

func TestPublishSubscribe(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a, b := randID(), randID()

	subA1, stopA1, err := bus.Subscribe(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	defer stopA1()
	subA2, stopA2, err := bus.Subscribe(ctx, a)
	if err != nil {
		t.Fatal(err)
	}
	defer stopA2()
	subB, stopB, err := bus.Subscribe(ctx, b)
	if err != nil {
		t.Fatal(err)
	}
	defer stopB()
	if n := numSub(t, rdb, a); n != 2 {
		t.Fatalf("NUMSUB = %d, want 2", n)
	}

	want := []events.Event{ev(a, 1), ev(a, 2), {SessionID: a, Type: events.TypeMessageDelta,
		TS: time.Now().UTC(), Payload: json.RawMessage(`{"message_id":"m","text":"<&>"}`)}}
	for _, e := range want {
		if err := bus.Publish(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	for _, sub := range []<-chan events.Event{subA1, subA2} {
		got := recvN(t, sub, len(want))
		for i := range want {
			if got[i].Seq != want[i].Seq || got[i].Type != want[i].Type || !got[i].TS.Equal(want[i].TS) ||
				string(got[i].Payload) != string(want[i].Payload) {
				t.Errorf("event %d = %+v, want %+v", i, got[i], want[i])
			}
		}
	}
	select {
	case e := <-subB:
		t.Fatalf("session B got %+v", e)
	case <-time.After(100 * time.Millisecond):
	}
}

// Subscribe returns only after Redis confirms, so an immediate publish
// is never lost.
func TestSubscribeIsConfirmed(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	ctx := context.Background()
	for i := 0; i < 20; i++ {
		s := randID()
		ch, stop, err := bus.Subscribe(ctx, s)
		if err != nil {
			t.Fatal(err)
		}
		if err := bus.Publish(ctx, ev(s, 1)); err != nil {
			t.Fatal(err)
		}
		recvN(t, ch, 1)
		stop()
	}
}

func TestCancelClosesAndUnsubscribes(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	s := randID()

	ch, stop, err := bus.Subscribe(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	stop()
	select {
	case _, ok := <-ch:
		if ok {
			t.Fatal("event after stop")
		}
	default:
		t.Fatal("channel still open when stop returned")
	}
	stop() // idempotent

	ctx, cancel := context.WithCancel(context.Background())
	ch, _, err = bus.Subscribe(ctx, s)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	waitClosed(t, ch, 5*time.Second)

	deadline := time.Now().Add(5 * time.Second)
	for numSub(t, rdb, s) != 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if n := numSub(t, rdb, s); n != 0 {
		t.Fatalf("NUMSUB = %d after cancel", n)
	}
	if _, _, err := bus.Subscribe(ctx, s); err == nil {
		t.Fatal("Subscribe with a cancelled ctx succeeded")
	}
}

// A broken connection closes the channel rather than reconnecting
// silently, so the subscriber knows to resync from the Store.
func TestKilledConnectionClosesChannel(t *testing.T) {
	admin := redistest.Client(t)
	name := "keith-b3-test-" + randID()
	opt, err := redis.ParseURL(redistest.RedisURL())
	if err != nil {
		t.Fatal(err)
	}
	opt.ClientName = name
	rdb := redis.NewClient(opt)
	defer rdb.Close()
	bus := redisbus.New(rdb)
	s := randID()

	ch, stop, err := bus.Subscribe(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	if err := bus.Publish(context.Background(), ev(s, 1)); err != nil {
		t.Fatal(err)
	}
	recvN(t, ch, 1)

	killed := killClients(t, admin, name)
	if killed == 0 {
		t.Fatal("found no connection to kill")
	}
	waitClosed(t, ch, 5*time.Second)
}

// killClients kills every connection named name.
func killClients(t *testing.T, admin *redis.Client, name string) int {
	t.Helper()
	ctx := context.Background()
	list, err := admin.ClientList(ctx).Result()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, line := range strings.Split(list, "\n") {
		fields := map[string]string{}
		for _, kv := range strings.Fields(line) {
			k, v, _ := strings.Cut(kv, "=")
			fields[k] = v
		}
		if fields["name"] != name {
			continue
		}
		if err := admin.Do(ctx, "CLIENT", "KILL", "ID", fields["id"]).Err(); err != nil {
			t.Fatal(err)
		}
		n++
	}
	return n
}

func TestSlowSubscriberIsClosedNotSkipped(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	bus.Buffer = 4
	s := randID()
	ch, stop, err := bus.Subscribe(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	for i := int64(1); i <= 20; i++ {
		if err := bus.Publish(context.Background(), ev(s, i)); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(200 * time.Millisecond)
	// What was buffered arrives in order, then the channel closes.
	var got []int64
	timeout := time.After(5 * time.Second)
	for {
		select {
		case e, ok := <-ch:
			if !ok {
				if len(got) == 0 || len(got) > 4 {
					t.Fatalf("got %v before close, want 1..4 events", got)
				}
				for i, seq := range got {
					if seq != int64(i+1) {
						t.Fatalf("got %v, want a prefix of 1..4", got)
					}
				}
				return
			}
			got = append(got, e.Seq)
		case <-timeout:
			t.Fatalf("overflowing subscription not closed; got %v", got)
		}
	}
}

func TestIdleSubscriptionStaysOpen(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	bus.HealthCheck = 50 * time.Millisecond
	s := randID()
	ch, stop, err := bus.Subscribe(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	select {
	case e, ok := <-ch:
		t.Fatalf("idle subscription produced %+v (open=%v)", e, ok)
	case <-time.After(500 * time.Millisecond): // ten health checks
	}
	if err := bus.Publish(context.Background(), ev(s, 1)); err != nil {
		t.Fatal(err)
	}
	recvN(t, ch, 1)
}

func TestIgnoresForeignMessages(t *testing.T) {
	rdb := redistest.Client(t)
	bus := redisbus.New(rdb)
	s := randID()
	ch, stop, err := bus.Subscribe(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	ctx := context.Background()
	if err := rdb.Publish(ctx, redisbus.Channel(s), "not json").Err(); err != nil {
		t.Fatal(err)
	}
	if err := bus.Publish(ctx, events.Event{SessionID: "other", Seq: 9, Type: events.TypeError,
		Payload: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	// Cross-post: an event for "other" on s's channel is dropped.
	if err := rdb.Publish(ctx, redisbus.Channel(s), `{"session_id":"other","seq":3,"type":"error","payload":{}}`).Err(); err != nil {
		t.Fatal(err)
	}
	if err := bus.Publish(ctx, ev(s, 1)); err != nil {
		t.Fatal(err)
	}
	got := recvN(t, ch, 1)
	if got[0].SessionID != s || got[0].Seq != 1 {
		t.Fatalf("got %+v", got[0])
	}
}
