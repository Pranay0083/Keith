package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/websocket"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store"
)

// WebSocket timing (docs/api.md, Events).
const (
	wsPingEvery = 25 * time.Second
	wsPongWait  = 60 * time.Second
	wsWriteWait = 10 * time.Second
	// wsRecheckEvery is how often an open stream re-checks that the
	// session exists and the user may still see it.
	wsRecheckEvery = 30 * time.Second
	// wsReadLimit caps a client frame; the client sends nothing.
	wsReadLimit = 4096
)

// WebSocket close codes (docs/api.md).
const (
	closeNormal       = websocket.CloseNormalClosure     // 1000
	closeGoingAway    = websocket.CloseGoingAway         // 1001
	closeServerError  = websocket.CloseInternalServerErr // 1011
	closeResync       = 4000
	closeUnauthorized = 4401
	closeForbidden    = 4403
	closeGone         = 4404
)

type wsUpgrader = websocket.Upgrader

func newUpgrader(webOrigin string) wsUpgrader {
	return websocket.Upgrader{
		HandshakeTimeout: 10 * time.Second,
		ReadBufferSize:   1024,
		WriteBufferSize:  16 << 10,
		// handleEvents has already answered 403 for a wrong Origin; this
		// is the upgrader's own check, kept as a second line.
		CheckOrigin: func(r *http.Request) bool {
			o := r.Header.Get("Origin")
			return o != "" && o == webOrigin
		},
	}
}

// handleEvents is GET /api/sessions/{id}/events?after_seq=N, the session
// event stream. The cookie (401) is checked by authed; then Origin (403),
// membership (404) and after_seq (400), all before the upgrade.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) error {
	if !s.allowedOrigin(r) {
		return errBadOrigin
	}
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	afterSeq := int64(0)
	if v := r.URL.Query().Get("after_seq"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n < 0 {
			return errBadRequest("after_seq must be an integer >= 0.")
		}
		afterSeq = n
	}
	if !websocket.IsWebSocketUpgrade(r) {
		return errBadRequest("This endpoint needs a WebSocket upgrade.")
	}
	user := currentUser(r)
	recheck := func(ctx context.Context) int {
		if _, err := s.store.GetUser(ctx, user.ID); errors.Is(err, store.ErrNotFound) {
			return closeUnauthorized
		}
		cur, err := s.store.GetSession(ctx, x.ID)
		if errors.Is(err, store.ErrNotFound) {
			return closeGone
		}
		if err != nil {
			return 0 // transient; check again later
		}
		ok, err := s.store.HasOrgRole(ctx, cur.OrgID, user.ID, store.RoleMember)
		if err == nil && !ok {
			return closeForbidden
		}
		return 0
	}
	return s.streamEvents(w, r, x.ID, afterSeq, sessionExpiry(r), recheck)
}

// streamEvents upgrades the connection and streams the session's events
// from events.Replay: stored events after afterSeq in order, then live
// ones (message.delta included, with seq 0), each stored seq once.
//
// It pings every 25 s and closes when no pong arrives within 60 s, when
// the client closes, at the session cookie's expiry (4401), when recheck
// reports lost access (4403/4404/4401), on Server.Shutdown (1001), and
// when the replay ends on a store error (1011). Client frames are read
// and discarded. If the upgrade fails the upgrader has already written
// the HTTP error.
func (s *Server) streamEvents(w http.ResponseWriter, r *http.Request, sessionID string, afterSeq int64,
	cookieExp time.Time, recheck func(context.Context) int) error {
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Replay subscribes, then reads the first page: a store error is
	// still a plain HTTP 500 here.
	evs, err := events.Replay(ctx, s.events, s.bus, sessionID, afterSeq)
	if err != nil {
		return err
	}
	s.wsMu.Lock()
	if s.closing {
		s.wsMu.Unlock()
		return errorf(http.StatusServiceUnavailable, codeInternal, "The server is shutting down. Reconnect shortly.")
	}
	s.wsWG.Add(1)
	s.wsMu.Unlock()
	defer s.wsWG.Done()
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return nil // the upgrader answered
	}
	defer conn.Close()
	log := s.log(r).With("session_id", sessionID)

	// Reader: discards client frames, extends the deadline on pong, and
	// cancels the stream when the client goes away.
	conn.SetReadLimit(wsReadLimit)
	_ = conn.SetReadDeadline(time.Now().Add(wsPongWait))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(wsPongWait))
	})
	readDone := make(chan struct{})
	go func() {
		defer close(readDone)
		defer cancel()
		for {
			if _, _, err := conn.NextReader(); err != nil {
				return
			}
		}
	}()

	closeWith := func(code int, text string) {
		_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(code, text), time.Now().Add(wsWriteWait))
	}
	ping := time.NewTicker(s.wsPing())
	defer ping.Stop()
	check := time.NewTicker(s.wsRecheck())
	defer check.Stop()
	var expired <-chan time.Time
	if !cookieExp.IsZero() {
		t := time.NewTimer(time.Until(cookieExp))
		defer t.Stop()
		expired = t.C
	}

	for {
		select {
		case ev, ok := <-evs:
			if !ok {
				if ctx.Err() == nil {
					// The replay stopped on a store error; the client
					// reconnects with its last seq.
					log.Warn("event replay ended")
					closeWith(closeServerError, "replay failed")
				}
				return nil
			}
			b, err := json.Marshal(ev)
			if err != nil {
				closeWith(closeServerError, "encode failed")
				return nil
			}
			_ = conn.SetWriteDeadline(time.Now().Add(wsWriteWait))
			if err := conn.WriteMessage(websocket.TextMessage, b); err != nil {
				return nil // slow or gone client; it reconnects with after_seq
			}
		case <-ping.C:
			if err := conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(wsWriteWait)); err != nil {
				return nil
			}
		case <-check.C:
			if code := recheck(ctx); code != 0 {
				closeWith(code, "access changed")
				return nil
			}
		case <-expired:
			closeWith(closeUnauthorized, "session expired")
			return nil
		case <-s.shutdown:
			closeWith(closeGoingAway, "server shutting down")
			return nil
		case <-readDone:
			return nil
		case <-ctx.Done():
			return nil
		}
	}
}

// wsPing is the ping interval; tests shorten it.
func (s *Server) wsPing() time.Duration {
	if s.wsPingEvery > 0 {
		return s.wsPingEvery
	}
	return wsPingEvery
}

// wsRecheck is the access re-check interval; tests shorten it.
func (s *Server) wsRecheck() time.Duration {
	if s.wsRecheckEvery > 0 {
		return s.wsRecheckEvery
	}
	return wsRecheckEvery
}
