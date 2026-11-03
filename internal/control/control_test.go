package control_test

import (
	"context"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/redistest"
)

func TestChannel(t *testing.T) {
	if got := control.Channel("abc"); got != "session:abc:control" {
		t.Fatalf("Channel = %q", got)
	}
}

func recv(t *testing.T, ch <-chan control.Signal) control.Signal {
	t.Helper()
	select {
	case s, ok := <-ch:
		if !ok {
			t.Fatal("channel closed")
		}
		return s
	case <-time.After(5 * time.Second):
		t.Fatal("no signal")
	}
	return ""
}

func testPubSub(t *testing.T, pub control.Publisher, sub control.Subscriber) {
	ctx := context.Background()
	id := "s-" + time.Now().Format("150405.000000000")
	sigs, stop, err := sub.Subscribe(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	other, stopOther, err := sub.Subscribe(ctx, id+"-other")
	if err != nil {
		t.Fatal(err)
	}
	defer stopOther()

	if err := pub.Publish(ctx, id, control.SignalCancel); err != nil {
		t.Fatal(err)
	}
	if got := recv(t, sigs); got != control.SignalCancel {
		t.Fatalf("got %q", got)
	}
	select {
	case s := <-other:
		t.Fatalf("other session got %q", s)
	case <-time.After(100 * time.Millisecond):
	}

	stop()
	if _, ok := <-sigs; ok {
		t.Fatal("channel open after stop")
	}
	stop() // idempotent

	// Cancelling ctx also ends a subscription.
	cctx, cancel := context.WithCancel(ctx)
	sigs2, stop2, err := sub.Subscribe(cctx, id)
	if err != nil {
		t.Fatal(err)
	}
	defer stop2()
	cancel()
	select {
	case _, ok := <-sigs2:
		if ok {
			t.Fatal("unexpected signal")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("channel not closed after ctx cancel")
	}
}

func TestMem(t *testing.T) {
	m := control.NewMem()
	testPubSub(t, m, m)
	if p := m.Published(); len(p) != 1 || p[0].Signal != control.SignalCancel {
		t.Fatalf("published %+v", p)
	}
}

func TestRedis(t *testing.T) {
	rdb := redistest.Client(t)
	r := control.NewRedis(rdb)
	testPubSub(t, r, r)

	// Garbage on the channel is skipped.
	ctx := context.Background()
	sigs, stop, err := r.Subscribe(ctx, "garbage-test")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	rdb.Publish(ctx, control.Channel("garbage-test"), "not json")
	if err := r.Publish(ctx, "garbage-test", control.SignalCancel); err != nil {
		t.Fatal(err)
	}
	if got := recv(t, sigs); got != control.SignalCancel {
		t.Fatalf("got %q", got)
	}
	if err := r.Publish(ctx, "", control.SignalCancel); err == nil {
		t.Error("empty session ID accepted")
	}
}
