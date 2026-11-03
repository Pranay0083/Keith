package events_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/memevents"
)

func TestEveryTypeHasAPayload(t *testing.T) {
	seen := map[string]bool{}
	for _, typ := range events.Types {
		if seen[typ] {
			t.Errorf("duplicate type %q", typ)
		}
		seen[typ] = true
		p, ok := events.NewPayload(typ)
		if !ok || p == nil {
			t.Errorf("no payload for %q", typ)
			continue
		}
		if reflect.TypeOf(p).Kind() != reflect.Pointer || reflect.TypeOf(p).Elem().Kind() != reflect.Struct {
			t.Errorf("payload for %q is %T, want pointer to struct", typ, p)
		}
	}
	if len(events.Types) != 12 {
		t.Errorf("len(Types) = %d, want 12", len(events.Types))
	}
	if _, ok := events.NewPayload("nope"); ok {
		t.Error("unknown type has a payload")
	}
}

// Payloads become TypeScript types, so every field must be exported,
// JSON-tagged and free of interfaces.
func TestPayloadsArePlainStructs(t *testing.T) {
	for _, typ := range events.Types {
		p, _ := events.NewPayload(typ)
		st := reflect.TypeOf(p).Elem()
		for i := 0; i < st.NumField(); i++ {
			f := st.Field(i)
			if !f.IsExported() || f.Anonymous {
				t.Errorf("%s.%s: unexported or embedded field", st.Name(), f.Name)
			}
			if f.Tag.Get("json") == "" {
				t.Errorf("%s.%s: missing json tag", st.Name(), f.Name)
			}
			if f.Type.Kind() == reflect.Interface || f.Type.Kind() == reflect.Pointer {
				t.Errorf("%s.%s: interface or pointer field", st.Name(), f.Name)
			}
		}
	}
}

func TestIsEphemeral(t *testing.T) {
	for _, typ := range events.Types {
		if got, want := events.IsEphemeral(typ), typ == events.TypeMessageDelta; got != want {
			t.Errorf("IsEphemeral(%q) = %v", typ, got)
		}
	}
}

func TestStatusTerminal(t *testing.T) {
	terminal := map[events.SessionStatus]bool{
		events.StatusCompleted: true, events.StatusFailed: true,
		events.StatusCancelled: true, events.StatusNoChanges: true,
	}
	if len(events.Statuses) != 9 {
		t.Errorf("len(Statuses) = %d, want 9", len(events.Statuses))
	}
	for _, s := range events.Statuses {
		if s.Terminal() != terminal[s] {
			t.Errorf("%s.Terminal() = %v", s, s.Terminal())
		}
	}
}

func TestEventJSONShape(t *testing.T) {
	ev := events.Event{
		SessionID: "s_123", Seq: 42,
		TS:      time.Date(2026, 11, 2, 10, 15, 3, 0, time.UTC),
		Type:    events.TypeToolCallStarted,
		Payload: json.RawMessage(`{"call_id":"c_9","tool":"shell_exec","input":{"shell_id":"dev","command":"npm test"}}`),
	}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"session_id":"s_123","seq":42,"ts":"2026-11-02T10:15:03Z","type":"tool_call.started","payload":{"call_id":"c_9","tool":"shell_exec","input":{"shell_id":"dev","command":"npm test"}}}`
	if string(b) != want {
		t.Errorf("got  %s\nwant %s", b, want)
	}
	p, err := events.Decode(ev)
	if err != nil {
		t.Fatal(err)
	}
	started := p.(*events.ToolCallStartedPayload)
	if started.CallID != "c_9" || started.Tool != "shell_exec" {
		t.Errorf("decoded = %+v", started)
	}
}

func TestEmitterStoresThenPublishes(t *testing.T) {
	ctx := context.Background()
	store := memevents.NewStore()
	bus := memevents.NewBus()
	em := &events.Emitter{Store: store, Bus: bus}

	ch, unsubscribe, err := bus.Subscribe(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()

	ev, err := em.Emit(ctx, "s1", events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning})
	if err != nil {
		t.Fatal(err)
	}
	if ev.Seq != 1 {
		t.Errorf("seq = %d, want 1", ev.Seq)
	}
	delta, err := em.Emit(ctx, "s1", events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m1", Text: "he"})
	if err != nil {
		t.Fatal(err)
	}
	if delta.Seq != 0 {
		t.Errorf("ephemeral seq = %d, want 0", delta.Seq)
	}

	if got := store.Types("s1"); !reflect.DeepEqual(got, []string{events.TypeSessionStatus}) {
		t.Errorf("stored types = %v, want only session.status", got)
	}
	for _, want := range []string{events.TypeSessionStatus, events.TypeMessageDelta} {
		select {
		case got := <-ch:
			if got.Type != want {
				t.Errorf("published %s, want %s", got.Type, want)
			}
		case <-time.After(time.Second):
			t.Fatalf("no %s published", want)
		}
	}
}

type failingBus struct{}

func (failingBus) Publish(context.Context, events.Event) error { return errors.New("redis down") }

func (failingBus) Subscribe(context.Context, string) (<-chan events.Event, func(), error) {
	return nil, nil, errors.New("redis down")
}

func TestEmitterPublishFailureKeepsStoredEvent(t *testing.T) {
	store := memevents.NewStore()
	em := &events.Emitter{Store: store, Bus: failingBus{}}
	ev, err := em.Emit(context.Background(), "s1", events.TypeError, events.ErrorPayload{Message: "x"})
	if !errors.Is(err, events.ErrPublish) {
		t.Fatalf("err = %v, want ErrPublish", err)
	}
	if ev.Seq != 1 || len(store.All("s1")) != 1 {
		t.Errorf("event not stored: %+v", ev)
	}
}
