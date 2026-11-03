package pgevents_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/pgevents"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/store/storetest"
)

// Integration tests: KEITH_IT=1, against DATABASE_URL (see storetest).

func setup(t *testing.T) (*store.Store, *pgevents.Store, string) {
	t.Helper()
	s := storetest.New(t)
	f := storetest.Seed(t, s)
	return s, pgevents.New(s.Pool()), f.Session.ID
}

func TestAppendAndAfter(t *testing.T) {
	_, es, sid := setup(t)
	ctx := context.Background()

	var appended []events.Event
	for i, p := range []struct {
		typ     string
		payload any
	}{
		{events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning}},
		{events.TypeMessageUser, events.MessageUserPayload{MessageID: "m1", Text: "fix <b>it</b> & ship"}},
		{events.TypeToolCallStarted, events.ToolCallStartedPayload{
			CallID: "c1", Tool: "shell_exec", Input: json.RawMessage(`{"command": "npm test",  "shell_id":"dev"}`),
		}},
		{events.TypeToolCallOutput, events.ToolCallOutputPayload{CallID: "c1", Chunk: "ok\x00\x1b[32m✓\x1b[0m\n"}},
		{events.TypeError, nil},
	} {
		ev, err := es.Append(ctx, sid, p.typ, p.payload)
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
		if ev.Seq != int64(i+1) || ev.SessionID != sid || ev.Type != p.typ || ev.TS.IsZero() || ev.TS.Location() != time.UTC {
			t.Fatalf("append %d = %+v", i, ev)
		}
		appended = append(appended, ev)
	}

	all, err := es.After(ctx, sid, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	// What Append returned is exactly what After reads back.
	if !reflect.DeepEqual(all, appended) {
		t.Fatalf("After = %+v\nAppend returned %+v", all, appended)
	}
	if string(all[4].Payload) != `{}` {
		t.Errorf("nil payload stored as %s", all[4].Payload)
	}
	p, err := events.Decode(all[3])
	if err != nil {
		t.Fatal(err)
	}
	if got := p.(*events.ToolCallOutputPayload).Chunk; got != "ok�\x1b[32m✓\x1b[0m\n" {
		t.Errorf("chunk = %q", got)
	}
	p, err = events.Decode(all[1])
	if err != nil {
		t.Fatal(err)
	}
	if got := p.(*events.MessageUserPayload).Text; got != "fix <b>it</b> & ship" {
		t.Errorf("text = %q", got)
	}

	page, err := es.After(ctx, sid, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(page) != 2 || page[0].Seq != 3 || page[1].Seq != 4 {
		t.Errorf("After(2, limit 2) = %+v", page)
	}
	if rest, _ := es.After(ctx, sid, 5, 0); len(rest) != 0 {
		t.Errorf("After(5) = %+v", rest)
	}

	if _, err := es.Append(ctx, sid, events.TypeMessageDelta, events.MessageDeltaPayload{}); !errors.Is(err, events.ErrEphemeral) {
		t.Errorf("delta err = %v", err)
	}
	if _, err := es.Append(ctx, "00000000-0000-4000-8000-000000000000", events.TypeError, nil); err == nil {
		t.Error("append to a missing session succeeded")
	}
}

func TestAppendFixedClock(t *testing.T) {
	_, es, sid := setup(t)
	at := time.Date(2026, 11, 3, 10, 15, 3, 123456789, time.FixedZone("x", 3600))
	es.Now = func() time.Time { return at }
	ev, err := es.Append(context.Background(), sid, events.TypeError, events.ErrorPayload{Message: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if want := at.UTC().Truncate(time.Microsecond); !ev.TS.Equal(want) {
		t.Errorf("ts = %v, want %v", ev.TS, want)
	}
}

func TestConcurrentAppendIsGapFree(t *testing.T) {
	_, es, sid := setup(t)
	ctx := context.Background()
	const writers, each = 20, 10

	var wg sync.WaitGroup
	errs := make(chan error, writers*each)
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < each; i++ {
				if _, err := es.Append(ctx, sid, events.TypeToolCallOutput,
					events.ToolCallOutputPayload{CallID: "c", Chunk: "x"}); err != nil {
					errs <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}

	all, err := es.After(ctx, sid, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != writers*each {
		t.Fatalf("stored %d events, want %d", len(all), writers*each)
	}
	for i, ev := range all {
		if ev.Seq != int64(i+1) {
			t.Fatalf("event %d has seq %d: seqs are not unique and gap-free", i, ev.Seq)
		}
	}
}

func TestAppendInTransaction(t *testing.T) {
	s, es, sid := setup(t)
	ctx := context.Background()
	if _, err := es.Append(ctx, sid, events.TypeError, nil); err != nil {
		t.Fatal(err)
	}

	// A lost seq race inside a transaction retries in a savepoint instead
	// of aborting the transaction: tx A holds seq 2 while tx B appends.
	done := make(chan events.Event, 1)
	err := s.WithTx(ctx, func(txA *store.Store) error {
		evA, err := txA.Events().Append(ctx, sid, events.TypeError, nil)
		if err != nil || evA.Seq != 2 {
			t.Fatalf("tx A append = %+v, %v", evA, err)
		}
		go func() {
			var ev events.Event
			err := s.WithTx(ctx, func(txB *store.Store) error {
				var err error
				ev, err = txB.Events().Append(ctx, sid, events.TypeError, nil)
				return err
			})
			if err != nil {
				t.Errorf("tx B: %v", err)
			}
			done <- ev
		}()
		time.Sleep(300 * time.Millisecond) // let B block on A's seq 2
		return nil                         // commit A; B conflicts, then retries
	})
	if err != nil {
		t.Fatal(err)
	}
	if evB := <-done; evB.Seq != 3 {
		t.Fatalf("tx B got seq %d, want 3", evB.Seq)
	}
	all, err := es.After(ctx, sid, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 || all[2].Seq != 3 {
		t.Fatalf("events = %+v", all)
	}
}
