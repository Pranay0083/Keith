// Package queue is the asynq job queue the API and the worker share.
//
// The API enqueues three task types through Client:
//
//   - run_session (RunSessionPayload): a new session. One per session,
//     ever: the task ID is "run_session:<session_id>", so a second enqueue
//     is a no-op.
//   - resume_session (ResumeSessionPayload): wake a worker for a session
//     nobody is running: a follow-up message, an answer, or a Stop while
//     the session is queued or waiting_for_user. Deduplicated per
//     (session, reason, input) while one is pending, running or retrying.
//   - github_webhook (GitHubWebhookPayload): a verified GitHub App webhook
//     for internal/hooks.ProcessGitHub. One per delivery ID.
//
// The worker runs a Server (NewServer) with a mux from NewMux.
//
// # Handler contract (wave D)
//
// resume_session is a wakeup, not a command: several may arrive for one
// session, and one may arrive while another worker is still running it.
// The handler must therefore:
//
//  1. Try to take the session lease. If another worker holds it, return
//     an error wrapping ErrBusy: the task is retried after BusyRetryDelay
//     without using up its retries. Never drop the task, or input sent as
//     that worker finishes could wait forever.
//  2. With the lease, drain store.ListPendingInputs in order and honour
//     sessions.cancel_requested_at, then return nil. If there is nothing
//     to do (the input was already consumed by a running worker), return
//     nil.
//
// run_session follows the same lease rule. Payload errors wrap
// asynq.SkipRetry, so a malformed task is archived, not retried.
package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/hibiken/asynq"
)

// Task types.
const (
	TypeRunSession    = "run_session"
	TypeResumeSession = "resume_session"
	TypeGitHubWebhook = "github_webhook"
)

// Queues. Session tasks run for minutes to hours; webhook tasks take
// milliseconds. Run them on separate Servers (or give webhooks spare
// concurrency) so long sessions never starve uninstall handling.
const (
	QueueSessions = "sessions"
	QueueWebhooks = "webhooks"
)

// Task settings.
const (
	// SessionTimeout bounds one run_session or resume_session attempt.
	// The worker enforces its own, shorter per-session limits.
	SessionTimeout = 6 * time.Hour
	// SessionMaxRetry is how often a failed session task is retried. A
	// retry resumes from agent_state; ErrBusy retries do not count.
	SessionMaxRetry = 3
	// WebhookTimeout and WebhookMaxRetry are for github_webhook.
	WebhookTimeout  = 2 * time.Minute
	WebhookMaxRetry = 10
	// Retention keeps finished tasks for inspection, and keeps the
	// run_session task ID taken for that long.
	Retention = 7 * 24 * time.Hour
	// ResumeUniqueTTL bounds resume_session deduplication.
	ResumeUniqueTTL = time.Hour
	// BusyRetryDelay is the retry delay after ErrBusy.
	BusyRetryDelay = 5 * time.Second
)

// ErrBusy means the session's lease is held by another worker. A handler
// returns an error wrapping it to have the task retried after
// BusyRetryDelay without counting a failure.
var ErrBusy = errors.New("queue: session is busy")

// RunSessionPayload starts a new session.
type RunSessionPayload struct {
	SessionID string `json:"session_id"`
}

// ResumeReason is why a resume_session was enqueued.
type ResumeReason string

const (
	// ResumeMessage: a message to a session in a terminal status.
	ResumeMessage ResumeReason = "message"
	// ResumeAnswer: an answer to the open question.
	ResumeAnswer ResumeReason = "answer"
	// ResumeCancel: Stop while no worker held the session (queued or
	// waiting_for_user), or the App was uninstalled.
	ResumeCancel ResumeReason = "cancel"
)

// Valid reports whether r is a defined reason.
func (r ResumeReason) Valid() bool {
	return r == ResumeMessage || r == ResumeAnswer || r == ResumeCancel
}

// ResumeSessionPayload wakes a worker for a session.
type ResumeSessionPayload struct {
	SessionID string       `json:"session_id"`
	Reason    ResumeReason `json:"reason"`
	// InputID is the session_inputs ID for message and answer; "" for
	// cancel. The worker drains every pending input, not just this one.
	InputID string `json:"input_id,omitempty"`
}

// GitHubWebhookPayload is a GitHub App webhook whose signature the API
// verified.
type GitHubWebhookPayload struct {
	// DeliveryID is X-GitHub-Delivery.
	DeliveryID string `json:"delivery_id"`
	// Event is X-GitHub-Event.
	Event string `json:"event"`
	// Body is the raw request body (base64 in the task's JSON).
	Body []byte `json:"body"`
}

func (p RunSessionPayload) validate() error {
	if p.SessionID == "" {
		return errors.New("run_session: empty session_id")
	}
	return nil
}

func (p ResumeSessionPayload) validate() error {
	switch {
	case p.SessionID == "":
		return errors.New("resume_session: empty session_id")
	case !p.Reason.Valid():
		return fmt.Errorf("resume_session: invalid reason %q", p.Reason)
	case p.Reason == ResumeCancel && p.InputID != "":
		return errors.New("resume_session: cancel takes no input_id")
	case p.Reason != ResumeCancel && p.InputID == "":
		return fmt.Errorf("resume_session: %s needs an input_id", p.Reason)
	}
	return nil
}

func (p GitHubWebhookPayload) validate() error {
	switch {
	case p.DeliveryID == "":
		return errors.New("github_webhook: empty delivery_id")
	case p.Event == "":
		return errors.New("github_webhook: empty event")
	}
	return nil
}

// Enqueuer is what the API and the webhook processor need from Client.
// Each method returns nil when the task already exists (a duplicate).
type Enqueuer interface {
	EnqueueRunSession(ctx context.Context, p RunSessionPayload) error
	EnqueueResumeSession(ctx context.Context, p ResumeSessionPayload) error
	EnqueueGitHubWebhook(ctx context.Context, p GitHubWebhookPayload) error
}

// Client enqueues tasks. It is safe for concurrent use.
type Client struct {
	c *asynq.Client
}

var _ Enqueuer = (*Client)(nil)

// NewClient returns a Client on rdb (a redis.UniversalClient). It does
// not own rdb: Close leaves it open.
func NewClient(rdb asynqRedis) *Client {
	return &Client{c: asynq.NewClientFromRedisClient(rdb)}
}

// Close releases the Client.
func (c *Client) Close() error { return c.c.Close() }

// Ping checks the connection to Redis.
func (c *Client) Ping() error { return c.c.Ping() }

// RunSessionTaskID is the task ID of a session's run_session task.
func RunSessionTaskID(sessionID string) string { return TypeRunSession + ":" + sessionID }

// WebhookTaskID is the task ID of a delivery's github_webhook task.
func WebhookTaskID(deliveryID string) string { return TypeGitHubWebhook + ":" + deliveryID }

// EnqueueRunSession enqueues the session's run_session task, once.
func (c *Client) EnqueueRunSession(ctx context.Context, p RunSessionPayload) error {
	if err := p.validate(); err != nil {
		return fmt.Errorf("queue: %w", err)
	}
	return c.enqueue(ctx, TypeRunSession, p,
		asynq.Queue(QueueSessions), asynq.TaskID(RunSessionTaskID(p.SessionID)),
		asynq.MaxRetry(SessionMaxRetry), asynq.Timeout(SessionTimeout), asynq.Retention(Retention))
}

// EnqueueResumeSession enqueues a resume_session task unless an identical
// one is pending, running or retrying.
func (c *Client) EnqueueResumeSession(ctx context.Context, p ResumeSessionPayload) error {
	if err := p.validate(); err != nil {
		return fmt.Errorf("queue: %w", err)
	}
	return c.enqueue(ctx, TypeResumeSession, p,
		asynq.Queue(QueueSessions), asynq.Unique(ResumeUniqueTTL),
		asynq.MaxRetry(SessionMaxRetry), asynq.Timeout(SessionTimeout), asynq.Retention(Retention))
}

// EnqueueGitHubWebhook enqueues a delivery's github_webhook task, once.
func (c *Client) EnqueueGitHubWebhook(ctx context.Context, p GitHubWebhookPayload) error {
	if err := p.validate(); err != nil {
		return fmt.Errorf("queue: %w", err)
	}
	return c.enqueue(ctx, TypeGitHubWebhook, p,
		asynq.Queue(QueueWebhooks), asynq.TaskID(WebhookTaskID(p.DeliveryID)),
		asynq.MaxRetry(WebhookMaxRetry), asynq.Timeout(WebhookTimeout), asynq.Retention(Retention))
}

func (c *Client) enqueue(ctx context.Context, typ string, payload any, opts ...asynq.Option) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("queue: encode %s: %w", typ, err)
	}
	_, err = c.c.EnqueueContext(ctx, asynq.NewTask(typ, b), opts...)
	if errors.Is(err, asynq.ErrDuplicateTask) || errors.Is(err, asynq.ErrTaskIDConflict) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("queue: enqueue %s: %w", typ, err)
	}
	return nil
}
