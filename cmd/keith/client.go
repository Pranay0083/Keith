package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/spf13/cobra"

	"github.com/Pranay0083/keith/internal/events"
)

// Environment of the experimental client commands.
const (
	envAPIURL        = "KEITH_API_URL"
	envWebOrigin     = "KEITH_WEB_ORIGIN"
	envSessionCookie = "KEITH_SESSION_COOKIE"
	defaultAPIURL    = "http://localhost:8080"
	defaultWebOrigin = "http://localhost:3000"
	sessionCookie    = "keith_session"
)

const clientEnvHelp = `Environment:
  KEITH_SESSION_COOKIE  required: the value of the keith_session cookie, copied
                        from the browser after signing in to the web app.
  KEITH_API_URL         the API's base URL (default http://localhost:8080).
  KEITH_WEB_ORIGIN      the API's WEB_ORIGIN, sent as Origin on the event
                        WebSocket (default http://localhost:3000).`

// apiClient calls the Keith API as a signed-in browser would.
type apiClient struct {
	base   *url.URL
	origin string
	cookie string
	hc     *http.Client
}

func newAPIClient(getenv func(string) string) (*apiClient, error) {
	cookie := strings.TrimSpace(getenv(envSessionCookie))
	cookie = strings.TrimPrefix(cookie, sessionCookie+"=")
	if cookie == "" {
		return nil, fmt.Errorf("%s is not set: sign in to the web app and copy the value of its %s cookie", envSessionCookie, sessionCookie)
	}
	if strings.ContainsAny(cookie, "; \t\r\n\"") {
		return nil, fmt.Errorf("%s must be the bare value of the %s cookie", envSessionCookie, sessionCookie)
	}
	raw := strings.TrimSpace(getenv(envAPIURL))
	if raw == "" {
		raw = defaultAPIURL
	}
	u, err := url.Parse(strings.TrimRight(raw, "/"))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("%s: %q is not an http(s) URL", envAPIURL, raw)
	}
	origin := strings.TrimRight(strings.TrimSpace(getenv(envWebOrigin)), "/")
	if origin == "" {
		origin = defaultWebOrigin
	}
	return &apiClient{base: u, origin: origin, cookie: cookie, hc: &http.Client{Timeout: 30 * time.Second}}, nil
}

// apiError is a non-2xx answer from the API.
type apiError struct {
	Status  int
	Code    string
	Message string
}

func (e *apiError) Error() string {
	msg := fmt.Sprintf("API answered %d", e.Status)
	if e.Code != "" {
		msg += " " + e.Code
	}
	if e.Message != "" {
		msg += ": " + e.Message
	}
	switch {
	case e.Status == http.StatusUnauthorized:
		msg += " (" + envSessionCookie + " is missing, expired or revoked: sign in again and copy the " + sessionCookie + " cookie)"
	case e.Code == "bad_origin":
		msg += " (set " + envWebOrigin + " to the API's WEB_ORIGIN)"
	}
	return msg
}

func readAPIError(resp *http.Response) error {
	var body struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	e := &apiError{Status: resp.StatusCode}
	if json.Unmarshal(b, &body) == nil {
		e.Code, e.Message = body.Error.Code, body.Error.Message
	}
	return e
}

func (c *apiClient) url(path string, q url.Values) string {
	u := *c.base
	u.Path = strings.TrimRight(u.Path, "/") + path
	u.RawQuery = q.Encode()
	return u.String()
}

// get fetches path and decodes the JSON answer into v.
func (c *apiClient) get(ctx context.Context, path string, q url.Values, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url(path, q), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Cookie", sessionCookie+"="+c.cookie)
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("GET %s: %w", path, readAPIError(resp))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(v); err != nil {
		return fmt.Errorf("GET %s: decode: %w", path, err)
	}
	return nil
}

// ---- keith sessions ----

type apiMe struct {
	User struct {
		Login string `json:"login"`
	} `json:"user"`
	Orgs []struct {
		ID   string `json:"id"`
		Name string `json:"name"`
		Role string `json:"role"`
	} `json:"orgs"`
}

type apiSession struct {
	ID         string               `json:"id"`
	OrgID      string               `json:"org_id"`
	Status     events.SessionStatus `json:"status"`
	Prompt     string               `json:"prompt"`
	IssueURL   *string              `json:"issue_url"`
	Provider   string               `json:"provider"`
	Model      string               `json:"model"`
	CreatedAt  time.Time            `json:"created_at"`
	FinishedAt *time.Time           `json:"finished_at"`
	Repos      []struct {
		FullName string  `json:"full_name"`
		Branch   string  `json:"branch"`
		PRNumber *int    `json:"pr_number"`
		PRURL    *string `json:"pr_url"`
	} `json:"repos"`
}

type apiSessionPage struct {
	Sessions   []apiSession `json:"sessions"`
	NextCursor *string      `json:"next_cursor"`
}

type sessionsOptions struct {
	Org   string
	Limit int
}

func newSessionsCmd() *cobra.Command {
	var o sessionsOptions
	cmd := &cobra.Command{
		Use:   "sessions",
		Short: "List an org's sessions (experimental)",
		Long: `(experimental, for development) sessions lists the newest sessions of an org
through the API, as the web app does: ID, status, creation time, repos (with
the pull request number once one is open) and the start of the prompt.
Without --org it uses your first org.

` + clientEnvHelp,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			c, err := newAPIClient(os.Getenv)
			if err != nil {
				return err
			}
			return listSessions(cmd.Context(), c, cmd.OutOrStdout(), o)
		},
	}
	cmd.Flags().StringVar(&o.Org, "org", "", "org ID (default: your first org)")
	cmd.Flags().IntVar(&o.Limit, "limit", 25, "how many sessions to show (1-100)")
	return cmd
}

func listSessions(ctx context.Context, c *apiClient, out io.Writer, o sessionsOptions) error {
	if o.Limit < 1 || o.Limit > 100 {
		return errors.New("sessions: --limit must be between 1 and 100")
	}
	org := strings.TrimSpace(o.Org)
	if org == "" {
		var me apiMe
		if err := c.get(ctx, "/api/me", nil, &me); err != nil {
			return fmt.Errorf("sessions: %w", err)
		}
		if len(me.Orgs) == 0 {
			return errors.New("sessions: you belong to no org")
		}
		org = me.Orgs[0].ID
	} else if _, err := uuid.Parse(org); err != nil {
		return fmt.Errorf("sessions: --org %q is not an org ID", org)
	}
	var page apiSessionPage
	q := url.Values{"limit": {strconv.Itoa(o.Limit)}}
	if err := c.get(ctx, "/api/orgs/"+url.PathEscape(org)+"/sessions", q, &page); err != nil {
		return fmt.Errorf("sessions: %w", err)
	}
	if len(page.Sessions) == 0 {
		fmt.Fprintln(out, "No sessions yet.")
		return nil
	}
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tSTATUS\tCREATED\tREPOS\tPROMPT")
	for _, s := range page.Sessions {
		var repos []string
		for _, r := range s.Repos {
			name := r.FullName
			if r.PRNumber != nil {
				name += fmt.Sprintf(" #%d", *r.PRNumber)
			}
			repos = append(repos, name)
		}
		prompt := oneLine(s.Prompt, 60)
		if prompt == "" && s.IssueURL != nil {
			prompt = *s.IssueURL
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", s.ID, s.Status, s.CreatedAt.Local().Format("2006-01-02 15:04"),
			strings.Join(repos, ", "), strings.ReplaceAll(prompt, "\t", " "))
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	if page.NextCursor != nil {
		fmt.Fprintf(out, "(more sessions: raise --limit, up to 100)\n")
	}
	return nil
}

// ---- keith logs ----

type logsOptions struct {
	Follow bool
	JSON   bool
	// Idle ends a non-follow run when no event arrives for this long;
	// TerminalIdle replaces it once the session's status is terminal.
	Idle         time.Duration
	TerminalIdle time.Duration
	// FollowTimeout is how long a follow connection may stay silent
	// (the server pings every 25 s) before it is replaced.
	FollowTimeout time.Duration
	// Backoff is the delay before reconnect attempt n (1-based).
	Backoff func(n int) time.Duration
	// MaxRetries bounds reconnects without --follow.
	MaxRetries int
}

func defaultLogsOptions() logsOptions {
	return logsOptions{
		Idle:          time.Second,
		TerminalIdle:  250 * time.Millisecond,
		FollowTimeout: 90 * time.Second,
		Backoff:       reconnectDelay,
		MaxRetries:    3,
	}
}

// reconnectDelay is docs/api.md's client backoff: min(15 s, 0.5 s × 2^n)
// scaled by a random factor in [0.5, 1).
func reconnectDelay(n int) time.Duration {
	d := min(15*time.Second, time.Duration(float64(500*time.Millisecond)*float64(int64(1)<<min(n, 10))))
	return time.Duration(float64(d) * (0.5 + rand.Float64()/2))
}

func newLogsCmd() *cobra.Command {
	o := defaultLogsOptions()
	cmd := &cobra.Command{
		Use:   "logs [-f] <session-id>",
		Short: "Print a session's event stream (experimental)",
		Long: `(experimental, for development) logs prints a session's events from the API's
event WebSocket, formatted like keith run: stored events first, then live ones.
Without -f it exits once the stream goes quiet (about a second after the stored
events, sooner once the session has finished). With -f it keeps following,
reconnecting where it left off, until Ctrl-C.

` + clientEnvHelp,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id := strings.TrimSpace(args[0])
			if _, err := uuid.Parse(id); err != nil {
				return fmt.Errorf("logs: %q is not a session ID", id)
			}
			c, err := newAPIClient(os.Getenv)
			if err != nil {
				return err
			}
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			p := newEventPrinter(cmd.OutOrStdout(), cmd.ErrOrStderr(), o.JSON)
			return streamLogs(ctx, c, id, p, o)
		},
	}
	cmd.Flags().BoolVarP(&o.Follow, "follow", "f", false, "keep following live events until Ctrl-C")
	cmd.Flags().BoolVar(&o.JSON, "json", false, "print raw events as JSON lines")
	return cmd
}

// eventsURL is the session's event WebSocket URL.
func (c *apiClient) eventsURL(sessionID string, afterSeq int64) string {
	u := *c.base
	switch u.Scheme {
	case "https":
		u.Scheme = "wss"
	default:
		u.Scheme = "ws"
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/api/sessions/" + url.PathEscape(sessionID) + "/events"
	u.RawQuery = url.Values{"after_seq": {strconv.FormatInt(afterSeq, 10)}}.Encode()
	return u.String()
}

func (c *apiClient) dialEvents(ctx context.Context, sessionID string, afterSeq int64) (*websocket.Conn, error) {
	h := http.Header{}
	h.Set("Cookie", sessionCookie+"="+c.cookie)
	h.Set("Origin", c.origin)
	d := websocket.Dialer{HandshakeTimeout: 15 * time.Second, Proxy: http.ProxyFromEnvironment}
	conn, resp, err := d.DialContext(ctx, c.eventsURL(sessionID, afterSeq), h)
	if err != nil {
		if resp != nil && resp.StatusCode != http.StatusSwitchingProtocols {
			defer resp.Body.Close()
			return nil, &permanentError{readAPIError(resp)}
		}
		return nil, err
	}
	return conn, nil
}

// permanentError is a failure reconnecting cannot fix.
type permanentError struct{ err error }

func (e *permanentError) Error() string { return e.err.Error() }
func (e *permanentError) Unwrap() error { return e.err }

// closeReasons are the close codes that mean "do not reconnect".
var closeReasons = map[int]string{
	4401: "the session cookie expired or was revoked: sign in again and update " + envSessionCookie,
	4403: "you no longer have access to this session's org",
	4404: "the session was deleted",
}

// streamLogs prints the session's events until the stream goes quiet
// (or, with Follow, until ctx ends), reconnecting with after_seq when the
// connection drops.
func streamLogs(ctx context.Context, c *apiClient, sessionID string, p *eventPrinter, o logsOptions) error {
	attempt := 0
	for {
		conn, err := c.dialEvents(ctx, sessionID, p.Last())
		if err == nil {
			start := time.Now()
			var quiet bool
			quiet, err = readEvents(ctx, conn, p, o)
			conn.Close()
			if quiet {
				return nil
			}
			var ce *websocket.CloseError
			if errors.As(err, &ce) {
				if why, ok := closeReasons[ce.Code]; ok {
					return fmt.Errorf("logs: %s", why)
				}
			}
			if time.Since(start) > 5*time.Second {
				attempt = 0
			}
		}
		if ctx.Err() != nil {
			return nil
		}
		var perm *permanentError
		if errors.As(err, &perm) {
			return fmt.Errorf("logs: %w", perm.err)
		}
		attempt++
		if !o.Follow && attempt > o.MaxRetries {
			return fmt.Errorf("logs: %w", err)
		}
		delay := o.Backoff(attempt)
		p.Say("(connection lost: %v; reconnecting in %s)", err, delay.Round(10*time.Millisecond))
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil
		case <-t.C:
		}
	}
}

// readEvents prints events from one connection. It returns quiet=true
// when a non-follow stream went quiet (done), else the error that ended
// the connection.
func readEvents(ctx context.Context, conn *websocket.Conn, p *eventPrinter, o logsOptions) (quiet bool, err error) {
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	if o.Follow {
		conn.SetPingHandler(func(data string) error {
			_ = conn.SetReadDeadline(time.Now().Add(o.FollowTimeout))
			err := conn.WriteControl(websocket.PongMessage, []byte(data), time.Now().Add(5*time.Second))
			var ne net.Error
			if errors.Is(err, websocket.ErrCloseSent) || (errors.As(err, &ne) && ne.Timeout()) {
				return nil
			}
			return err
		})
	}
	for {
		wait := o.FollowTimeout
		if !o.Follow {
			wait = o.Idle
			if p.Status().Terminal() {
				wait = o.TerminalIdle
			}
		}
		_ = conn.SetReadDeadline(time.Now().Add(wait))
		_, data, err := conn.ReadMessage()
		if err != nil {
			var ne net.Error
			if !o.Follow && errors.As(err, &ne) && ne.Timeout() {
				return true, nil
			}
			return false, err
		}
		var ev events.Event
		if json.Unmarshal(data, &ev) != nil || ev.Type == "" {
			continue
		}
		p.Print(ev)
	}
}
