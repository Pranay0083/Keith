package tmuxshell

import (
	"testing"
	"time"
)

func TestDeadline(t *testing.T) {
	t0 := time.Unix(1000, 0)
	at := func(s float64) time.Time { return t0.Add(time.Duration(s * float64(time.Second))) }

	t.Run("silent command times out at its timeout", func(t *testing.T) {
		d := newDeadline(t0, 10*time.Second, 3*time.Second, time.Minute)
		d.observe(at(5), false)
		if d.expired(at(9.9)) || !d.expired(at(10)) {
			t.Fatal("want expiry at exactly 10s")
		}
	})
	t.Run("output before the timeout does not extend", func(t *testing.T) {
		d := newDeadline(t0, 10*time.Second, 3*time.Second, time.Minute)
		d.observe(at(2), true)
		if !d.expired(at(10)) {
			t.Fatal("early output must not extend the timeout")
		}
	})
	t.Run("output near the end extends by the silence limit", func(t *testing.T) {
		d := newDeadline(t0, 10*time.Second, 3*time.Second, time.Minute)
		d.observe(at(9), true) // → 12
		if d.expired(at(11.9)) {
			t.Fatal("expired before 12s")
		}
		d.observe(at(11.5), true) // → 14.5
		if d.expired(at(14.4)) || !d.expired(at(14.5)) {
			t.Fatal("want expiry at 14.5s")
		}
		if r := d.remaining(at(13)); r != 1500*time.Millisecond {
			t.Fatalf("remaining = %v", r)
		}
	})
	t.Run("hard max caps a chatty command", func(t *testing.T) {
		d := newDeadline(t0, 10*time.Second, 3*time.Second, 20*time.Second)
		for s := 0.0; s < 30; s += 0.5 {
			d.observe(at(s), true)
			if s < 20 && d.expired(at(s)) {
				t.Fatalf("expired at %vs", s)
			}
		}
		if !d.expired(at(20)) {
			t.Fatal("hard max not enforced")
		}
	})
	t.Run("timeout above max is clamped", func(t *testing.T) {
		d := newDeadline(t0, time.Hour, time.Second, 5*time.Second)
		if !d.expired(at(5)) || d.remaining(at(6)) != 0 {
			t.Fatal("timeout not clamped to max")
		}
	})
}
