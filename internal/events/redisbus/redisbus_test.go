package redisbus

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/events"
)

func TestChannel(t *testing.T) {
	if got := Channel("abc"); got != "session:abc" {
		t.Errorf("Channel = %q", got)
	}
}

func TestEncodeDecodeKeepsPayloadBytes(t *testing.T) {
	e := events.Event{
		SessionID: "s",
		Seq:       7,
		TS:        time.Date(2026, 11, 3, 10, 0, 0, 123456000, time.UTC),
		Type:      events.TypeMessageUser,
		Payload:   json.RawMessage("{\"message_id\":\"m\",\"text\":\"<b>&</b> \u2713 \u2028 \\u2029\"}"),
	}
	data, err := encode(e)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decode(string(data))
	if err != nil {
		t.Fatal(err)
	}
	if got.SessionID != e.SessionID || got.Seq != e.Seq || !got.TS.Equal(e.TS) || got.Type != e.Type {
		t.Errorf("decoded %+v, want %+v", got, e)
	}
	// Payload bytes are copied verbatim (no HTML or U+2028 escaping), so
	// they match what pgevents returns.
	if string(got.Payload) != string(e.Payload) {
		t.Errorf("payload = %s, want %s", got.Payload, e.Payload)
	}
}

func TestRedisDownFailsFast(t *testing.T) {
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", MaxRetries: -1, DialTimeout: time.Second})
	defer rdb.Close()
	bus := New(rdb)
	ctx := context.Background()
	if _, _, err := bus.Subscribe(ctx, "s"); err == nil {
		t.Error("Subscribe succeeded without Redis")
	}
	if err := bus.Publish(ctx, events.Event{SessionID: "s", Type: events.TypeError}); err == nil {
		t.Error("Publish succeeded without Redis")
	}
}

func TestEncodeRejectsInvalidPayload(t *testing.T) {
	if _, err := encode(events.Event{SessionID: "s", Payload: json.RawMessage(`{"a":`)}); err == nil {
		t.Error("invalid payload encoded")
	}
	data, err := encode(events.Event{SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(data) {
		t.Errorf("nil payload encodes to invalid JSON: %s", data)
	}
}
