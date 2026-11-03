package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/redistest"
	"github.com/Pranay0083/keith/internal/store"
)

const wsSession = "3c4b0000-0000-4000-8000-000000000001"

// streamServer serves streamEvents for wsSession straight from the
// harness's memevents store and bus, skipping the database checks.
func streamServer(t *testing.T, h *harness, exp time.Time, recheck func(context.Context) int) *httptest.Server {
	t.Helper()
	if recheck == nil {
		recheck = func(context.Context) int { return 0 }
	}
	ts := httptest.NewServer(h.srv.withRequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		after, _ := strconv.ParseInt(r.URL.Query().Get("after_seq"), 10, 64)
		if err := h.srv.streamEvents(w, r, wsSession, after, exp, recheck); err != nil {
			h.srv.writeError(w, r, err)
		}
	})))
	t.Cleanup(ts.Close)
	return ts
}

func wsURL(base, path string) string { return "ws" + strings.TrimPrefix(base, "http") + path }

func dialWS(t *testing.T, url string, hdr http.Header) *websocket.Conn {
	t.Helper()
	if hdr == nil {
		hdr = http.Header{}
	}
	if hdr.Get("Origin") == "" {
		hdr.Set("Origin", testWebOrigin)
	}
	c, resp, err := websocket.DefaultDialer.Dial(url, hdr)
	if err != nil {
		status := 0
		if resp != nil {
			status = resp.StatusCode
		}
		t.Fatalf("dial %s: %v (status %d)", url, err, status)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func readEvent(t *testing.T, c *websocket.Conn) events.Event {
	t.Helper()
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	typ, b, err := c.ReadMessage()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if typ != websocket.TextMessage {
		t.Fatalf("frame type %d", typ)
	}
	var ev events.Event
	if err := json.Unmarshal(b, &ev); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return ev
}

// wantSeqs reads len(want) events and checks their seqs.
func wantSeqs(t *testing.T, c *websocket.Conn, want ...int64) []events.Event {
	t.Helper()
	var got []events.Event
	for _, w := range want {
		ev := readEvent(t, c)
		if ev.Seq != w {
			t.Fatalf("seq %d (%s), want %d", ev.Seq, ev.Type, w)
		}
		got = append(got, ev)
	}
	return got
}

// wantQuiet checks nothing arrives for a while. The read timeout leaves
// the connection unusable, so call it last.
func wantQuiet(t *testing.T, c *websocket.Conn) {
	t.Helper()
	_ = c.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
	if _, b, err := c.ReadMessage(); err == nil {
		t.Fatalf("unexpected frame %s", b)
	}
}

// wantClose reads until the server's close frame and checks its code.
func wantClose(t *testing.T, c *websocket.Conn, code int) {
	t.Helper()
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	for {
		_, _, err := c.ReadMessage()
		if err == nil {
			continue
		}
		var ce *websocket.CloseError
		if !errors.As(err, &ce) || ce.Code != code {
			t.Fatalf("close: %v, want code %d", err, code)
		}
		return
	}
}

func emit(t *testing.T, em *events.Emitter, typ string, payload any) events.Event {
	t.Helper()
	ev, err := em.Emit(context.Background(), wsSession, typ, payload)
	if err != nil {
		t.Fatal(err)
	}
	return ev
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestStreamReplayLiveReconnect(t *testing.T) {
	h := newHarness(t, nil)
	em := &events.Emitter{Store: h.evs, Bus: h.bus}
	emit(t, em, events.TypeMessageUser, events.MessageUserPayload{MessageID: "m_1", Text: "fix it"})
	emit(t, em, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusProvisioning})
	emit(t, em, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning})
	ts := streamServer(t, h, time.Time{}, nil)

	// Replay after_seq=1, then live.
	c := dialWS(t, wsURL(ts.URL, "/?after_seq=1"), nil)
	got := wantSeqs(t, c, 2, 3)
	if got[0].SessionID != wsSession || got[0].Type != events.TypeSessionStatus || !strings.Contains(string(got[0].Payload), "provisioning") {
		t.Fatalf("event %+v", got[0])
	}
	emit(t, em, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m_a", Text: "Hel"})
	emit(t, em, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m_a", Text: "Hello"})
	live := wantSeqs(t, c, 0, 4)
	if live[0].Type != events.TypeMessageDelta || live[1].Type != events.TypeMessageAssistant {
		t.Fatalf("live %+v", live)
	}
	wantQuiet(t, c)

	// The client goes away; the subscription ends.
	c.Close()
	waitFor(t, "unsubscribe", func() bool { return h.bus.Subscribers(wsSession) == 0 })
	emit(t, em, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c1", Tool: "shell_exec", Input: json.RawMessage(`{}`)})

	// Reconnect with the last seq: only what it missed, then live, and
	// never a seq it already had.
	c2 := dialWS(t, wsURL(ts.URL, "/?after_seq=4"), nil)
	wantSeqs(t, c2, 5)
	emit(t, em, events.TypeToolCallFinished, events.ToolCallFinishedPayload{CallID: "c1", Tool: "shell_exec"})
	wantSeqs(t, c2, 6)

	// A second tab from 0 sees everything in order.
	c3 := dialWS(t, wsURL(ts.URL, "/"), nil)
	wantSeqs(t, c3, 1, 2, 3, 4, 5, 6)
	// Client frames are ignored.
	if err := c3.WriteMessage(websocket.TextMessage, []byte(`{"hello":1}`)); err != nil {
		t.Fatal(err)
	}
	emit(t, em, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusCompleted})
	wantSeqs(t, c3, 7)
	wantSeqs(t, c2, 7)
	// Nothing else, and never an old seq. (A read timeout breaks a
	// gorilla connection, so this comes last.)
	wantQuiet(t, c2)
}

func TestStreamCloses(t *testing.T) {
	t.Run("shutdown 1001", func(t *testing.T) {
		h := newHarness(t, nil)
		ts := streamServer(t, h, time.Time{}, nil)
		c := dialWS(t, wsURL(ts.URL, "/"), nil)
		waitFor(t, "subscribe", func() bool { return h.bus.Subscribers(wsSession) == 1 })
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		go func() { _ = h.srv.Shutdown(ctx) }()
		wantClose(t, c, closeGoingAway)
	})
	t.Run("cookie expiry 4401", func(t *testing.T) {
		h := newHarness(t, nil)
		ts := streamServer(t, h, time.Now().Add(300*time.Millisecond), nil)
		wantClose(t, dialWS(t, wsURL(ts.URL, "/"), nil), closeUnauthorized)
	})
	t.Run("lost access 4403", func(t *testing.T) {
		h := newHarness(t, nil)
		h.srv.wsRecheckEvery = 50 * time.Millisecond
		var calls atomic.Int32
		ts := streamServer(t, h, time.Time{}, func(context.Context) int {
			if calls.Add(1) >= 2 {
				return closeForbidden
			}
			return 0
		})
		wantClose(t, dialWS(t, wsURL(ts.URL, "/"), nil), closeForbidden)
	})
	t.Run("replay failure 1011", func(t *testing.T) {
		h := newHarness(t, nil)
		fb := &flakyBus{ch: make(chan events.Event)}
		h.srv.bus = fb
		h.srv.events = &failingStore{Store: h.evs}
		ts := streamServer(t, h, time.Time{}, nil)
		c := dialWS(t, wsURL(ts.URL, "/"), nil)
		close(fb.ch) // the live channel drops; resubscribe and the store fail
		wantClose(t, c, closeServerError)
	})
	t.Run("pings", func(t *testing.T) {
		h := newHarness(t, nil)
		h.srv.wsPingEvery = 30 * time.Millisecond
		ts := streamServer(t, h, time.Time{}, nil)
		c := dialWS(t, wsURL(ts.URL, "/"), nil)
		var pings atomic.Int32
		c.SetPingHandler(func(string) error {
			pings.Add(1)
			return c.WriteControl(websocket.PongMessage, nil, time.Now().Add(time.Second))
		})
		go func() {
			for {
				if _, _, err := c.ReadMessage(); err != nil {
					return
				}
			}
		}()
		waitFor(t, "pings", func() bool { return pings.Load() >= 2 })
	})
	t.Run("bad origin refused by upgrader", func(t *testing.T) {
		h := newHarness(t, nil)
		ts := streamServer(t, h, time.Time{}, nil)
		_, resp, err := websocket.DefaultDialer.Dial(wsURL(ts.URL, "/"), http.Header{"Origin": {"https://evil.test"}})
		if err == nil || resp == nil || resp.StatusCode != http.StatusForbidden {
			t.Fatalf("dial from bad origin: %v %v", err, resp)
		}
	})
}

// flakyBus hands out one subscription the test controls, then fails.
type flakyBus struct {
	ch    chan events.Event
	calls atomic.Int32
}

func (b *flakyBus) Publish(context.Context, events.Event) error { return nil }

func (b *flakyBus) Subscribe(context.Context, string) (<-chan events.Event, func(), error) {
	if b.calls.Add(1) == 1 {
		return b.ch, func() {}, nil
	}
	return nil, nil, errors.New("bus down")
}

// failingStore answers the first After and fails afterwards.
type failingStore struct {
	events.Store
	calls atomic.Int32
}

func (s *failingStore) After(ctx context.Context, id string, after int64, limit int) ([]events.Event, error) {
	if s.calls.Add(1) == 1 {
		return s.Store.After(ctx, id, after, limit)
	}
	return nil, errors.New("db down")
}

// --- The full endpoint, on Postgres and Redis (KEITH_IT=1) ---

func TestEventsEndpointIT(t *testing.T) {
	w := newWorld(t)
	rdb := redistest.Client(t)
	w.srv.events = w.st.Events()
	w.srv.bus = redisbus.New(rdb)
	ts := httptest.NewServer(w.h)
	t.Cleanup(ts.Close)
	ctx := context.Background()

	x, err := w.st.CreateSession(ctx, store.CreateSessionParams{
		OrgID: w.org.ID, CreatedBy: w.member.ID, Prompt: "p", Provider: "anthropic", Model: "m", RepoIDs: []string{w.repo.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	em := &events.Emitter{Store: w.st.Events(), Bus: w.srv.bus}
	emitIT := func(typ string, payload any) {
		t.Helper()
		if _, err := em.Emit(ctx, x.ID, typ, payload); err != nil {
			t.Fatal(err)
		}
	}
	emitIT(events.TypeMessageUser, events.MessageUserPayload{MessageID: "m_1", Text: "p"})
	emitIT(events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusProvisioning})

	path := "/api/sessions/" + x.ID + "/events"
	cookie := func(userID string) http.Header {
		v, _ := w.srv.signer.signSession(userID, time.Now(), time.Now().Add(time.Hour))
		return http.Header{"Cookie": {sessionCookie + "=" + v}, "Origin": {testWebOrigin}}
	}

	// Pre-upgrade checks answer plain HTTP JSON errors.
	upgrade := func(hdr http.Header) http.Header {
		hdr.Set("Connection", "Upgrade")
		hdr.Set("Upgrade", "websocket")
		hdr.Set("Sec-WebSocket-Version", "13")
		hdr.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
		return hdr
	}
	check := func(p string, hdr http.Header, status int, code string) {
		t.Helper()
		r := httptest.NewRequest("GET", p, nil)
		r.Header = upgrade(hdr)
		rec := httptest.NewRecorder()
		w.h.ServeHTTP(rec, r)
		wantError(t, rec, status, code)
	}
	check(path, http.Header{"Origin": {testWebOrigin}}, 401, codeUnauthenticated)
	bad := cookie(w.member.ID)
	bad.Set("Origin", "https://evil.test")
	check(path, bad, 403, codeBadOrigin)
	noOrigin := cookie(w.member.ID)
	noOrigin.Del("Origin")
	check(path, noOrigin, 403, codeBadOrigin)
	check(path, cookie(w.outsider.ID), 404, codeNotFound)
	check("/api/sessions/00000000-0000-4000-8000-000000000000/events", cookie(w.member.ID), 404, codeNotFound)
	check(path+"?after_seq=-1", cookie(w.member.ID), 400, codeBadRequest)
	check(path+"?after_seq=x", cookie(w.member.ID), 400, codeBadRequest)
	plain := httptest.NewRequest("GET", path, nil)
	plain.Header = cookie(w.member.ID)
	rec := httptest.NewRecorder()
	w.h.ServeHTTP(rec, plain)
	wantError(t, rec, 400, codeBadRequest)

	// Replay, live, reconnect.
	c := dialWS(t, wsURL(ts.URL, path+"?after_seq=1"), cookie(w.member.ID))
	wantSeqs(t, c, 2)
	emitIT(events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m_a", Text: "Hi"})
	emitIT(events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m_a", Text: "Hi there"})
	wantSeqs(t, c, 0, 3)
	c.Close()
	emitIT(events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning})
	c = dialWS(t, wsURL(ts.URL, path+"?after_seq=3"), cookie(w.member.ID))
	wantSeqs(t, c, 4)
	emitIT(events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusCompleted})
	wantSeqs(t, c, 5)
	wantQuiet(t, c)

	// Removing the member closes the stream with 4403.
	w.srv.wsRecheckEvery = 50 * time.Millisecond
	c2 := dialWS(t, wsURL(ts.URL, path+"?after_seq=5"), cookie(w.member.ID))
	if err := w.st.RemoveOrgMember(ctx, w.org.ID, w.member.ID); err != nil {
		t.Fatal(err)
	}
	wantClose(t, c2, closeForbidden)
}
