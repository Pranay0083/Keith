package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/redis/go-redis/v9"
	"github.com/spf13/cobra"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/hooks"
	"github.com/Pranay0083/keith/internal/lease"
	"github.com/Pranay0083/keith/internal/llm"
	llmproviders "github.com/Pranay0083/keith/internal/llm/providers"
	"github.com/Pranay0083/keith/internal/queue"
	sbproviders "github.com/Pranay0083/keith/internal/sandbox/providers"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
	"github.com/Pranay0083/keith/internal/worker"
)

// webhookConcurrency is the webhook server's concurrency: webhooks take
// milliseconds, so a few workers keep up.
const webhookConcurrency = 4

func newWorkerCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "worker",
		Short: "Run the agent worker that executes sessions",
		Long: "Run the worker: it takes run_session and resume_session jobs from the queue, runs each " +
			"session's sandbox and agent loop, processes GitHub App webhooks and deletes idle sandboxes. " +
			"Configuration comes from the environment (see internal/config).",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			logger := slog.New(slog.NewJSONHandler(os.Stderr, nil)).With("service", "worker")
			return runWorker(ctx, logger, config.Load)
		},
	}
}

// runWorker loads and validates the configuration, then runs the session
// and webhook queue servers and the reaper until ctx is cancelled, then
// shuts them down gracefully.
func runWorker(ctx context.Context, logger *slog.Logger, load func() (*config.Config, error)) error {
	cfg, err := load()
	if err != nil {
		return err
	}
	if err := cfg.Validate(config.RoleWorker); err != nil {
		return err
	}
	logger.Info("starting", "version", version, "config", cfg.String())

	st, err := store.Open(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer st.Close()
	if err := st.Migrate(ctx); err != nil {
		return err
	}

	opt, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		// The parse error can echo the URL, password included.
		return errors.New("worker: REDIS_URL is not a valid Redis URL")
	}
	rdb := redis.NewClient(opt)
	defer rdb.Close()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("worker: ping redis: %w", err)
	}

	keyring, err := secrets.NewKeyring(cfg.SecretsKey)
	if err != nil {
		return err
	}
	app, err := github.NewApp(github.AppConfig{
		AppID:         cfg.GitHubAppID,
		ClientID:      cfg.GitHubAppClientID,
		ClientSecret:  cfg.GitHubAppClientSecret,
		PrivateKeyPEM: cfg.GitHubAppPrivateKey,
	})
	if err != nil {
		return err
	}
	sandboxes, err := sbproviders.New(cfg)
	if err != nil {
		return err
	}
	qc := queue.NewClient(rdb)
	defer qc.Close()
	ctl := control.NewRedis(rdb)

	opts := worker.OptionsFromConfig(cfg)
	if cfg.SandboxProvider == config.SandboxProviderDocker {
		// Only containers this kind of worker made are reaped as orphans.
		opts.SandboxLabels = map[string]string{"keith.role": "worker"}
		opts.OrphanLabels = map[string]string{"keith.role": "worker"}
	}
	w, err := worker.New(worker.Deps{
		Store:     st,
		Events:    st.Events(),
		Bus:       redisbus.New(rdb),
		Leases:    worker.NewRedisLeaser(lease.NewManager(rdb)),
		Control:   ctl,
		Queue:     qc,
		Sandboxes: sandboxes,
		GitHub:    worker.NewAppGitHub(app),
		LLM:       func(name string) (llm.Provider, error) { return llmproviders.New(name, cfg) },
		Keyring:   keyring,
		Logger:    logger,
	}, opts)
	if err != nil {
		return err
	}

	// Session tasks run for minutes to hours and webhook tasks for
	// milliseconds, so each queue gets its own server: long sessions can
	// never starve uninstall handling.
	// A session task asynq is about to archive (retries used up) would
	// leave its session live with nothing to run it: the worker fails it,
	// or enqueues a fresh wakeup after ErrBusy.
	sessions := queue.NewServer(rdb, queue.ServerConfig{
		Concurrency:  cfg.WorkerConcurrency,
		Queues:       map[string]int{queue.QueueSessions: 1},
		Logger:       logger,
		ErrorHandler: queue.SessionErrorHandler(w.TaskExhausted),
	})
	webhooks := queue.NewServer(rdb, queue.ServerConfig{
		Concurrency: webhookConcurrency,
		Queues:      map[string]int{queue.QueueWebhooks: 1},
		Logger:      logger,
	})
	// On SIGTERM running sessions stop at once: each saves its state,
	// parks its sandbox, releases its lease and returns ErrBusy, so the
	// task is retried (on another worker) without waiting for asynq's
	// shutdown timeout.
	h := w.Handlers()
	h.RunSession = stopOnShutdown(ctx, h.RunSession)
	h.ResumeSession = stopOnShutdown(ctx, h.ResumeSession)
	if err := sessions.Start(queue.NewMux(h)); err != nil {
		return fmt.Errorf("worker: start session server: %w", err)
	}
	if err := webhooks.Start(queue.NewMux(queue.Handlers{
		GitHubWebhook: hooks.Handler(hooks.Deps{Store: st, GitHub: app, Queue: qc, Control: ctl, Logger: logger}),
	})); err != nil {
		sessions.Shutdown()
		return fmt.Errorf("worker: start webhook server: %w", err)
	}

	reapCtx, stopReaper := context.WithCancel(context.WithoutCancel(ctx))
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		w.RunReaper(reapCtx)
	}()
	logger.Info("running", "concurrency", cfg.WorkerConcurrency, "sandbox_provider", sandboxes.Name())

	<-ctx.Done()
	logger.Info("shutting down")
	// Shutdown stops taking tasks and cancels running ones after its
	// timeout; a cancelled session saves its state, parks its sandbox
	// and is retried by another worker.
	sessions.Shutdown()
	webhooks.Shutdown()
	stopReaper()
	wg.Wait()
	logger.Info("stopped")
	return nil
}

// stopOnShutdown cancels a handler's context when shutdown is.
func stopOnShutdown[P any](shutdown context.Context, fn func(context.Context, P) error) func(context.Context, P) error {
	return func(ctx context.Context, p P) error {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		stop := context.AfterFunc(shutdown, cancel)
		defer stop()
		return fn(ctx, p)
	}
}
