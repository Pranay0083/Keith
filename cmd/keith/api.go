package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/spf13/cobra"

	"github.com/Pranay0083/keith/internal/api"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events/redisbus"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/secrets"
	"github.com/Pranay0083/keith/internal/store"
)

// shutdownTimeout bounds graceful shutdown after SIGINT or SIGTERM.
const shutdownTimeout = 25 * time.Second

func newAPICmd() *cobra.Command {
	return &cobra.Command{
		Use:   "api",
		Short: "Run the HTTP and WebSocket API service",
		Long: "Run the control-plane API: GitHub sign-in and App setup, webhooks, sessions, " +
			"repos, secrets and the session event stream. Configuration comes from the " +
			"environment (see internal/config).",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			logger := slog.New(slog.NewJSONHandler(os.Stderr, nil)).With("service", "api")
			return runAPI(ctx, logger, config.Load)
		},
	}
}

// runAPI loads and validates the configuration, migrates the database,
// and serves until ctx is cancelled, then shuts down gracefully.
func runAPI(ctx context.Context, logger *slog.Logger, load func() (*config.Config, error)) error {
	cfg, err := load()
	if err != nil {
		return err
	}
	if err := cfg.Validate(config.RoleAPI); err != nil {
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
		return errors.New("api: REDIS_URL is not a valid Redis URL")
	}
	rdb := redis.NewClient(opt)
	defer rdb.Close()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("api: ping redis: %w", err)
	}

	keyring, err := secrets.NewKeyring(cfg.SecretsKey)
	if err != nil {
		return err
	}
	appCfg := github.AppConfig{
		AppID:         cfg.GitHubAppID,
		ClientID:      cfg.GitHubAppClientID,
		ClientSecret:  cfg.GitHubAppClientSecret,
		PrivateKeyPEM: cfg.GitHubAppPrivateKey,
	}
	// Sign-in sends redirect_uri; the install-time authorization went to
	// the App's own callback, so its code is exchanged without one.
	signInCfg := appCfg
	signInCfg.RedirectURL = cfg.PublicURL + "/auth/github/callback"
	signInApp, err := github.NewApp(signInCfg)
	if err != nil {
		return err
	}
	setupApp, err := github.NewApp(appCfg)
	if err != nil {
		return err
	}

	qc := queue.NewClient(rdb)
	defer qc.Close()

	srv, err := api.New(api.Deps{
		Config:      cfg,
		Store:       st,
		Bus:         redisbus.New(rdb),
		Queue:       qc,
		Control:     control.NewRedis(rdb),
		GitHub:      signInApp,
		SetupGitHub: setupApp,
		Dedupe:      api.NewRedisDeduper(rdb),
		Keyring:     keyring,
		Ready: func(ctx context.Context) error {
			if err := st.Pool().Ping(ctx); err != nil {
				return fmt.Errorf("postgres: %w", err)
			}
			if err := rdb.Ping(ctx).Err(); err != nil {
				return fmt.Errorf("redis: %w", err)
			}
			return nil
		},
		Logger: logger,
	})
	if err != nil {
		return err
	}

	hs := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// Bodies are at most 1 MiB (25 MB for webhooks). WebSockets set
		// their own deadlines after the upgrade.
		ReadTimeout:    60 * time.Second,
		WriteTimeout:   2 * time.Minute,
		IdleTimeout:    2 * time.Minute,
		MaxHeaderBytes: 64 << 10,
		ErrorLog:       slog.NewLogLogger(logger.Handler(), slog.LevelWarn),
	}
	errc := make(chan error, 1)
	go func() {
		logger.Info("listening", "addr", cfg.ListenAddr)
		errc <- hs.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("api: serve: %w", err)
		}
		return nil
	case <-ctx.Done():
	}

	logger.Info("shutting down")
	sctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	// Stop new requests and drain plain ones, then close WebSockets with
	// 1001 (http.Server does not track hijacked connections).
	herr := hs.Shutdown(sctx)
	werr := srv.Shutdown(sctx)
	if err := errors.Join(herr, werr); err != nil {
		return fmt.Errorf("api: shutdown: %w", err)
	}
	logger.Info("stopped")
	return nil
}
