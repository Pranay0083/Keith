package memevents

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
)

func TestStoreAppendAndAfter(t *testing.T) {
	ctx := context.Background()
	s := NewStore()
	for i := 0; i < 5; i++ {
		if _, err := s.Append(ctx, "a", events.TypeMessageUser, events.MessageUserPayload{Text: "hi"}); err != nil {
			t.Fatal(err)
		}
	}
	ev, err := s.Append(ctx, "b", events.TypeSessionStatus, json.RawMessage(`{"status":"queued"}`))
	if err != nil {
		t.Fatal(err)
	}
	if ev.Seq != 1 || string(ev.Payload) != `{"status":"queued"}` {
		t.Errorf("session b first event = %+v", ev)
	}

	got, err := s.After(ctx, "a", 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Seq != 3 || got[2].Seq != 5 {
		t.Errorf("After(2) = %+v", got)
	}
	got, _ = s.After(ctx, "a", 0, 2)
	if len(got) != 2 || got[1].Seq != 2 {
		t.Errorf("After(0, limit 2) = %+v", got)
	}
	got, _ = s.After(ctx, "a", 5, 0)
	if len(got) != 0 {
		t.Errorf("After(5) = %+v", got)
	}

	if _, err := s.Append(ctx, "a", events.TypeMessageDelta, nil); !errors.Is(err, events.ErrEphemeral) {
		t.Errorf("ephemeral append err = %v", err)
	}
	ev, _ = s.Append(ctx, "a", events.TypeError, nil)
	if string(ev.Payload) != `{}` {
		t.Errorf("nil payload stored as %s", ev.Payload)
	}
}

func TestBusFanOutAndUnsubscribe(t *testing.T) {
	ctx := context.Background()
	b := NewBus()
	ch1, cancel1, _ := b.Subscribe(ctx, "s")
	ch2, cancel2, _ := b.Subscribe(ctx, "s")
	other, cancelOther, _ := b.Subscribe(ctx, "t")
	defer cancelOther()

	if err := b.Publish(ctx, events.Event{SessionID: "s", Seq: 1}); err != nil {
		t.Fatal(err)
	}
	for _, ch := range []<-chan events.Event{ch1, ch2} {
		select {
		case e := <-ch:
			if e.Seq != 1 {
				t.Errorf("got seq %d", e.Seq)
			}
		case <-time.After(time.Second):
			t.Fatal("subscriber got nothing")
		}
	}
	select {
	case e := <-other:
		t.Errorf("other session received %+v", e)
	default:
	}

	cancel1()
	cancel1() // idempotent
	if _, ok := <-ch1; ok {
		t.Error("channel open after unsubscribe")
	}
	if b.Subscribers("s") != 1 {
		t.Errorf("Subscribers = %d, want 1", b.Subscribers("s"))
	}
	cancel2()
	if b.Subscribers("s") != 0 {
		t.Errorf("Subscribers = %d, want 0", b.Subscribers("s"))
	}
}

func TestBusContextCancelEndsSubscription(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	b := NewBus()
	ch, _, err := b.Subscribe(ctx, "s")
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("unexpected event")
		}
	case <-time.After(time.Second):
		t.Fatal("channel not closed after ctx cancel")
	}
}

func TestBusDropsForSlowSubscriber(t *testing.T) {
	ctx := context.Background()
	b := &Bus{Buffer: 1}
	ch, cancel, _ := b.Subscribe(ctx, "s")
	defer cancel()
	b.Publish(ctx, events.Event{SessionID: "s", Seq: 1})
	b.Publish(ctx, events.Event{SessionID: "s", Seq: 2}) // dropped, must not block
	if e := <-ch; e.Seq != 1 {
		t.Errorf("got seq %d, want 1", e.Seq)
	}
}
