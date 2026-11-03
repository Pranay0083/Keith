package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
)

// asynqRedis is the Redis client type asynq accepts.
type asynqRedis = redis.UniversalClient

// ServerConfig configures a worker's Server.
type ServerConfig struct {
	// Concurrency is how many tasks run at once; 0 means 10.
	Concurrency int
	// Queues maps queue names to priority weights; nil means both queues,
	// webhooks weighted higher.
	Queues map[string]int
	// Logger receives asynq's logs; nil means slog.Default().
	Logger *slog.Logger
	// ShutdownTimeout is how long Shutdown waits for running tasks; 0
	// means 30 s. Session handlers should save state and return when
	// their context is cancelled.
	ShutdownTimeout time.Duration
	// ErrorHandler, if set, is called for every task error.
	ErrorHandler func(ctx context.Context, task *asynq.Task, err error)
}

// NewServer returns an asynq Server on rdb with this package's retry
// policy: ErrBusy retries after BusyRetryDelay without counting a
// failure; anything else uses asynq's exponential backoff. Start it with
// srv.Start(NewMux(handlers)) and stop it with srv.Shutdown().
func NewServer(rdb redis.UniversalClient, cfg ServerConfig) *asynq.Server {
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 10
	}
	if cfg.Queues == nil {
		cfg.Queues = map[string]int{QueueWebhooks: 2, QueueSessions: 1}
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.ShutdownTimeout <= 0 {
		cfg.ShutdownTimeout = 30 * time.Second
	}
	ac := asynq.Config{
		Concurrency:     cfg.Concurrency,
		Queues:          cfg.Queues,
		Logger:          slogLogger{cfg.Logger},
		ShutdownTimeout: cfg.ShutdownTimeout,
		RetryDelayFunc:  RetryDelay,
		IsFailure:       IsFailure,
	}
	if cfg.ErrorHandler != nil {
		ac.ErrorHandler = asynq.ErrorHandlerFunc(cfg.ErrorHandler)
	}
	return asynq.NewServerFromRedisClient(rdb, ac)
}

// RetryDelay is the Server's retry delay: BusyRetryDelay after ErrBusy,
// asynq's default otherwise.
func RetryDelay(n int, err error, t *asynq.Task) time.Duration {
	if errors.Is(err, ErrBusy) {
		return BusyRetryDelay
	}
	return asynq.DefaultRetryDelayFunc(n, err, t)
}

// IsFailure reports whether a task error counts toward its retry limit.
// ErrBusy does not.
func IsFailure(err error) bool { return err != nil && !errors.Is(err, ErrBusy) }

// Handlers are the worker's task handlers. A nil handler is not
// registered; its tasks then fail with "handler not found" and retry.
type Handlers struct {
	RunSession    func(ctx context.Context, p RunSessionPayload) error
	ResumeSession func(ctx context.Context, p ResumeSessionPayload) error
	GitHubWebhook func(ctx context.Context, p GitHubWebhookPayload) error
}

// NewMux returns a mux that decodes each task's payload and calls its
// handler. A payload that does not decode or validate fails with an error
// wrapping asynq.SkipRetry.
func NewMux(h Handlers) *asynq.ServeMux {
	mux := asynq.NewServeMux()
	if h.RunSession != nil {
		mux.Handle(TypeRunSession, Handler(h.RunSession, RunSessionPayload.validate))
	}
	if h.ResumeSession != nil {
		mux.Handle(TypeResumeSession, Handler(h.ResumeSession, ResumeSessionPayload.validate))
	}
	if h.GitHubWebhook != nil {
		mux.Handle(TypeGitHubWebhook, Handler(h.GitHubWebhook, GitHubWebhookPayload.validate))
	}
	return mux
}

// Handler adapts a typed handler to asynq: it decodes the payload into P,
// checks it with validate (if non-nil) and calls fn.
func Handler[P any](fn func(context.Context, P) error, validate func(P) error) asynq.Handler {
	return asynq.HandlerFunc(func(ctx context.Context, t *asynq.Task) error {
		var p P
		if err := json.Unmarshal(t.Payload(), &p); err != nil {
			return fmt.Errorf("queue: decode %s payload: %v: %w", t.Type(), err, asynq.SkipRetry)
		}
		if validate != nil {
			if err := validate(p); err != nil {
				return fmt.Errorf("queue: %v: %w", err, asynq.SkipRetry)
			}
		}
		return fn(ctx, p)
	})
}

// slogLogger adapts slog to asynq.Logger.
type slogLogger struct{ l *slog.Logger }

func (s slogLogger) Debug(args ...any) { s.l.Debug(fmt.Sprint(args...), "component", "asynq") }
func (s slogLogger) Info(args ...any)  { s.l.Info(fmt.Sprint(args...), "component", "asynq") }
func (s slogLogger) Warn(args ...any)  { s.l.Warn(fmt.Sprint(args...), "component", "asynq") }
func (s slogLogger) Error(args ...any) { s.l.Error(fmt.Sprint(args...), "component", "asynq") }

// Fatal logs at error level and panics; asynq calls it only on
// unrecoverable setup errors. It does not os.Exit, so deferred cleanup in
// the worker still runs.
func (s slogLogger) Fatal(args ...any) {
	msg := fmt.Sprint(args...)
	s.l.Error(msg, "component", "asynq", "fatal", true)
	panic("asynq: " + msg)
}
