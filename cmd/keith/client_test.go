package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/Pranay0083/keith/internal/events"
)

const (
	testCookie = "signed.cookie-value_123"
	testOrigin = "http://web.test"
	testOrg1   = "11111111-1111-4111-8111-111111111111"
	testOrg2   = "22222222-2222-4222-8222-222222222222"
	testSessA  = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	testSessB  = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
)

func testClient(t *testing.T, srvURL string) *apiClient {
	t.Helper()
	c, err := newAPIClient(envFrom(map[string]string{
		envSessionCookie: sessionCookie + "=" + testCookie, // a pasted name=value works too
		envAPIURL:        srvURL + "/",
		envWebOrigin:     testOrigin + "/",
	}))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func authed(t *testing.T, r *http.Request) bool {
	c, err := r.Cookie(sessionCookie)
	return err == nil && c.Value == testCookie
}

func writeAPIError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "message": msg}})
}

func TestNewAPIClient(t *testing.T) {
	for _, c := range []struct {
		env  map[string]string
		want string
	}{
		{map[string]string{}, "KEITH_SESSION_COOKIE is not set"},
		{map[string]string{envSessionCookie: "a; b=c"}, "bare value"},
		{map[string]string{envSessionCookie: "x", envAPIURL: "ftp://api"}, "KEITH_API_URL"},
	} {
		if _, err := newAPIClient(envFrom(c.env)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("env %v: err = %v, want %q", c.env, err, c.want)
		}
	}
	c, err := newAPIClient(envFrom(map[string]string{envSessionCookie: "x"}))
	if err != nil || c.base.String() != defaultAPIURL || c.origin != defaultWebOrigin {
		t.Fatalf("defaults: %+v, %v", c, err)
	}
	if got := c.eventsURL(testSessA, 7); got != "ws://localhost:8080/api/sessions/"+testSessA+"/events?after_seq=7" {
		t.Errorf("events URL = %s", got)
	}
	c, _ = newAPIClient(envFrom(map[string]string{envSessionCookie: "x", envAPIURL: "https://api.keith.dev/base"}))
	if got := c.eventsURL(testSessA, 0); got != "wss://api.keith.dev/base/api/sessions/"+testSessA+"/events?after_seq=0" {
		t.Errorf("events URL = %s", got)
	}
}

func TestListSessions(t *testing.T) {
	var mu sync.Mutex
	var calls []string
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/me", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls = append(calls, "me")
		mu.Unlock()
		if !authed(t, r) {
			writeAPIError(w, 401, "unauthenticated", "Sign in first.")
			return
		}
		io.WriteString(w, `{"user":{"id":"u","login":"octocat","name":"","avatar_url":""},"orgs":[{"id":"`+testOrg1+`","name":"octocat","role":"owner"},{"id":"`+testOrg2+`","name":"zeta","role":"member"}]}`)
	})
	mux.HandleFunc("GET /api/orgs/{org}/sessions", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls = append(calls, r.PathValue("org")+"?"+r.URL.RawQuery)
		mu.Unlock()
		if !authed(t, r) {
			writeAPIError(w, 401, "unauthenticated", "Sign in first.")
			return
		}
		io.WriteString(w, `{"sessions":[
			{"id":"`+testSessA+`","org_id":"`+testOrg1+`","status":"completed","prompt":"Fix the flaky date test in utils\nand more","issue_url":null,
			 "provider":"anthropic","model":"m","created_by":null,"cancel_requested":false,"created_at":"2026-11-02T10:15:00Z","finished_at":"2026-11-02T10:30:00Z",
			 "repos":[{"repo_id":"r1","full_name":"acme/web","branch":"agent/x","pr_number":12,"pr_url":"https://github.com/acme/web/pull/12"},
			          {"repo_id":"r2","full_name":"acme/api","branch":"agent/x","pr_number":null,"pr_url":null}]},
			{"id":"`+testSessB+`","org_id":"`+testOrg1+`","status":"running","prompt":"","issue_url":"https://github.com/acme/web/issues/42",
			 "provider":"gemini","model":"g","created_by":null,"cancel_requested":false,"created_at":"2026-11-02T11:00:00Z","finished_at":null,"repos":[]}
		],"next_cursor":"abc"}`)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	var out bytes.Buffer
	if err := listSessions(context.Background(), testClient(t, srv.URL), &out, sessionsOptions{Limit: 5}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"ID", "STATUS", "PROMPT", testSessA, "completed", "acme/web #12, acme/api", "Fix the flaky date test in utils …",
		testSessB, "running", "https://github.com/acme/web/issues/42", "(more sessions"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	mu.Lock()
	if len(calls) != 2 || calls[0] != "me" || calls[1] != testOrg1+"?limit=5" {
		t.Errorf("calls = %v", calls)
	}
	calls = nil
	mu.Unlock()

	// --org skips /api/me.
	out.Reset()
	if err := listSessions(context.Background(), testClient(t, srv.URL), &out, sessionsOptions{Org: testOrg2, Limit: 25}); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	if len(calls) != 1 || calls[0] != testOrg2+"?limit=25" {
		t.Errorf("calls = %v", calls)
	}
	mu.Unlock()

	// Bad flags and a stale cookie.
	if err := listSessions(context.Background(), testClient(t, srv.URL), &out, sessionsOptions{Limit: 0}); err == nil {
		t.Error("limit 0 accepted")
	}
	if err := listSessions(context.Background(), testClient(t, srv.URL), &out, sessionsOptions{Org: "nope", Limit: 5}); err == nil {
		t.Error("bad org accepted")
	}
	stale, _ := newAPIClient(envFrom(map[string]string{envSessionCookie: "stale", envAPIURL: srv.URL}))
	err := listSessions(context.Background(), stale, &out, sessionsOptions{Limit: 5})
	if err == nil || !strings.Contains(err.Error(), "401 unauthenticated: Sign in first.") || !strings.Contains(err.Error(), envSessionCookie) {
		t.Errorf("stale cookie: %v", err)
	}
}

// ---- logs ----

type wsServer struct {
	t   *testing.T
	srv *httptest.Server
	// serve handles one accepted connection (the n-th, from 0).
	serve func(n int, conn *websocket.Conn, afterSeq string)

	mu    sync.Mutex
	after []string
}

func newWSServer(t *testing.T, serve func(n int, conn *websocket.Conn, afterSeq string)) *wsServer {
	s := &wsServer{t: t, serve: serve}
	up := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return r.Header.Get("Origin") == testOrigin }}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/sessions/{id}/events", func(w http.ResponseWriter, r *http.Request) {
		if !authed(t, r) {
			writeAPIError(w, 401, "unauthenticated", "Sign in first.")
			return
		}
		if r.Header.Get("Origin") != testOrigin {
			writeAPIError(w, 403, "bad_origin", "Bad origin.")
			return
		}
		if r.PathValue("id") != testSessA {
			writeAPIError(w, 404, "not_found", "Not found.")
			return
		}
		conn, err := up.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		s.mu.Lock()
		n := len(s.after)
		s.after = append(s.after, r.URL.Query().Get("after_seq"))
		s.mu.Unlock()
		s.serve(n, conn, r.URL.Query().Get("after_seq"))
	})
	s.srv = httptest.NewServer(mux)
	t.Cleanup(s.srv.Close)
	return s
}

func (s *wsServer) afterSeqs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.after...)
}

func send(t *testing.T, conn *websocket.Conn, evs ...events.Event) {
	for _, e := range evs {
		e.SessionID = testSessA
		b, _ := json.Marshal(e)
		if err := conn.WriteMessage(websocket.TextMessage, b); err != nil {
			t.Errorf("send: %v", err)
		}
	}
}

// waitClosed keeps the connection open, as the server does after a
// terminal status, until the client goes away.
func waitClosed(conn *websocket.Conn) {
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
	}
}

func fastLogs(follow bool) logsOptions {
	o := defaultLogsOptions()
	o.Follow = follow
	o.Idle = 5 * time.Second
	o.TerminalIdle = 100 * time.Millisecond
	o.Backoff = func(int) time.Duration { return 5 * time.Millisecond }
	return o
}

func TestLogsExitsAfterTerminalStatus(t *testing.T) {
	s := newWSServer(t, func(n int, conn *websocket.Conn, _ string) {
		send(t, conn,
			ev(1, events.TypeMessageUser, events.MessageUserPayload{MessageID: "u", Text: "Fix it"}),
			ev(0, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: "m", Text: "Look"}),
			ev(2, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: "m", Text: "Looking."}),
			ev(3, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusFailed, Reason: "step_cap"}),
			ev(4, events.TypeError, events.ErrorPayload{Code: "step_cap", Message: "Too many steps."}),
		)
		waitClosed(conn)
	})
	var out, info bytes.Buffer
	p := newEventPrinter(&out, &info, false)
	start := time.Now()
	if err := streamLogs(context.Background(), testClient(t, s.srv.URL), testSessA, p, fastLogs(false)); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("took %s: did not stop soon after the terminal status", d)
	}
	want := "› Fix it\nLooking.\n● failed (step_cap)\n✗ error (step_cap): Too many steps.\n"
	if out.String() != want {
		t.Errorf("output:\n%q\nwant\n%q", out.String(), want)
	}
	if got := s.afterSeqs(); len(got) != 1 || got[0] != "0" {
		t.Errorf("after_seq = %v", got)
	}
}

func TestLogsReconnectsWithAfterSeq(t *testing.T) {
	s := newWSServer(t, func(n int, conn *websocket.Conn, after string) {
		switch n {
		case 0:
			send(t, conn,
				ev(1, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning}),
				ev(2, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c", Tool: "shell_exec", Input: json.RawMessage(`{"command":"ls"}`)}))
			conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseServiceRestart, "restart"))
		default:
			// A replayed duplicate is dropped; then the rest.
			send(t, conn,
				ev(2, events.TypeToolCallStarted, events.ToolCallStartedPayload{CallID: "c", Tool: "shell_exec", Input: json.RawMessage(`{"command":"ls"}`)}),
				ev(3, events.TypeToolCallFinished, events.ToolCallFinishedPayload{CallID: "c", Tool: "shell_exec", DurationMS: 100}),
				ev(4, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusCompleted}))
			waitClosed(conn)
		}
	})
	var out, info bytes.Buffer
	p := newEventPrinter(&out, &info, false)
	if err := streamLogs(context.Background(), testClient(t, s.srv.URL), testSessA, p, fastLogs(false)); err != nil {
		t.Fatal(err)
	}
	if want := "● running\n▶ shell_exec  ls\n✓ shell_exec (0.1s)\n● completed\n"; out.String() != want {
		t.Errorf("output:\n%q\nwant\n%q", out.String(), want)
	}
	if got := s.afterSeqs(); len(got) != 2 || got[0] != "0" || got[1] != "2" {
		t.Errorf("after_seq = %v", got)
	}
	if !strings.Contains(info.String(), "reconnecting") {
		t.Errorf("info = %q", info.String())
	}
}

func TestLogsIdleExitWithoutFollow(t *testing.T) {
	s := newWSServer(t, func(n int, conn *websocket.Conn, _ string) {
		send(t, conn, ev(1, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusRunning}))
		waitClosed(conn)
	})
	o := fastLogs(false)
	o.Idle = 100 * time.Millisecond
	var out bytes.Buffer
	if err := streamLogs(context.Background(), testClient(t, s.srv.URL), testSessA, newEventPrinter(&out, io.Discard, true), o); err != nil {
		t.Fatal(err)
	}
	var e events.Event
	if err := json.Unmarshal(out.Bytes(), &e); err != nil || e.Seq != 1 || e.Type != events.TypeSessionStatus {
		t.Errorf("--json output = %q (%v)", out.String(), err)
	}
}

func TestLogsFollowUntilCancelled(t *testing.T) {
	s := newWSServer(t, func(n int, conn *websocket.Conn, _ string) {
		send(t, conn, ev(1, events.TypeSessionStatus, events.SessionStatusPayload{Status: events.StatusCompleted}))
		waitClosed(conn)
	})
	ctx, cancel := context.WithCancel(context.Background())
	out := &syncBuffer{}
	p := newEventPrinter(out, io.Discard, false)
	done := make(chan error, 1)
	go func() { done <- streamLogs(ctx, testClient(t, s.srv.URL), testSessA, p, fastLogs(true)) }()
	for deadline := time.Now().Add(5 * time.Second); !strings.Contains(out.String(), "● completed"); time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("no event; output %q", out.String())
		}
	}
	select {
	case err := <-done:
		t.Fatalf("follow returned at a terminal status: %v", err)
	case <-time.After(300 * time.Millisecond):
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("follow did not stop on cancel")
	}
}

func TestLogsPermanentErrors(t *testing.T) {
	s := newWSServer(t, func(n int, conn *websocket.Conn, _ string) {
		conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(4404, "deleted"))
	})
	p := newEventPrinter(io.Discard, io.Discard, false)
	err := streamLogs(context.Background(), testClient(t, s.srv.URL), testSessA, p, fastLogs(true))
	if err == nil || !strings.Contains(err.Error(), "the session was deleted") {
		t.Errorf("4404: %v", err)
	}
	if n := len(s.afterSeqs()); n != 1 {
		t.Errorf("reconnected after 4404: %d connections", n)
	}

	// Handshake failures are not retried, even with -f.
	stale, _ := newAPIClient(envFrom(map[string]string{envSessionCookie: "stale", envAPIURL: s.srv.URL, envWebOrigin: testOrigin}))
	err = streamLogs(context.Background(), stale, testSessA, p, fastLogs(true))
	if err == nil || !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), envSessionCookie) {
		t.Errorf("401: %v", err)
	}
	wrongOrigin, _ := newAPIClient(envFrom(map[string]string{envSessionCookie: testCookie, envAPIURL: s.srv.URL, envWebOrigin: "http://elsewhere"}))
	err = streamLogs(context.Background(), wrongOrigin, testSessA, p, fastLogs(true))
	if err == nil || !strings.Contains(err.Error(), "bad_origin") || !strings.Contains(err.Error(), envWebOrigin) {
		t.Errorf("bad origin: %v", err)
	}
	err = streamLogs(context.Background(), testClient(t, s.srv.URL), testSessB, p, fastLogs(true))
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("unknown session: %v", err)
	}
}

func TestClientCommandsRegistered(t *testing.T) {
	for _, name := range []string{"run", "sessions", "logs"} {
		cmd, _, err := newRootCmd().Find([]string{name})
		if err != nil || cmd.Name() != name {
			t.Fatalf("%s: %v %v", name, cmd, err)
		}
		if name != "run" && !strings.Contains(cmd.Short, "(experimental)") {
			t.Errorf("%s Short = %q", name, cmd.Short)
		}
	}
	root := newRootCmd()
	root.SetArgs([]string{"logs", "not-a-uuid"})
	root.SetOut(io.Discard)
	root.SetErr(io.Discard)
	if err := root.Execute(); err == nil || !strings.Contains(err.Error(), "not a session ID") {
		t.Errorf("logs bad id: %v", err)
	}
}
