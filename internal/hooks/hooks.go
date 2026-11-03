// Package hooks processes GitHub App webhooks for the worker, and holds
// the two control-plane actions the API and the webhooks share: syncing
// an installation's repos and stopping a session.
//
// The API verifies a webhook's signature, dedupes it on its delivery ID
// and enqueues a github_webhook task; the worker's handler for that task
// calls ProcessGitHub:
//
//	queue.Handlers{GitHubWebhook: func(ctx context.Context, p queue.GitHubWebhookPayload) error {
//		return hooks.ProcessGitHub(ctx, deps, p.Event, p.Body)
//	}}
//
// Processing is idempotent, so a retried task is harmless.
package hooks

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
)

// GitHub is the part of *github.App the processor uses.
type GitHub interface {
	ListInstallationRepos(ctx context.Context, installationID int64) ([]github.Repo, error)
}

// Deps are what processing and stopping need.
type Deps struct {
	Store   *store.Store
	GitHub  GitHub
	Queue   queue.Enqueuer
	Control control.Publisher
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

func (d Deps) logger() *slog.Logger {
	if d.Logger != nil {
		return d.Logger
	}
	return slog.Default()
}

// SyncInstallationRepos lists an installation's repos on GitHub, upserts
// each one as accessible under the installation's org, and marks the
// installation's other repos inaccessible, in one transaction. The
// installation must already be linked (store.UpsertInstallation). It
// returns how many repos GitHub listed.
func SyncInstallationRepos(ctx context.Context, st *store.Store, gh GitHub, installationID int64) (int, error) {
	repos, err := gh.ListInstallationRepos(ctx, installationID)
	if err != nil {
		return 0, fmt.Errorf("hooks: sync repos of installation %d: %w", installationID, err)
	}
	err = st.WithTx(ctx, func(tx *store.Store) error {
		keep := make([]int64, 0, len(repos))
		for _, r := range repos {
			if r.ID <= 0 || r.FullName == "" {
				continue
			}
			if _, err := tx.UpsertRepo(ctx, store.UpsertRepoParams{
				InstallationID: installationID, GitHubRepoID: r.ID, FullName: r.FullName,
				DefaultBranch: r.DefaultBranch, Private: r.Private,
			}); err != nil {
				return err
			}
			keep = append(keep, r.ID)
		}
		_, err := tx.SetInstallationReposInaccessibleExcept(ctx, installationID, keep)
		return err
	})
	if err != nil {
		return 0, fmt.Errorf("hooks: sync repos of installation %d: %w", installationID, err)
	}
	return len(repos), nil
}

// ErrSessionFinished is returned by StopSession for a session in a
// terminal status (the API's 409 not_running).
var ErrSessionFinished = errors.New("hooks: session is not running")

// StopSession asks a live session to stop:
//
//  1. In one transaction it locks the session row, refuses a terminal
//     session (ErrSessionFinished) and sets cancel_requested_at
//     (idempotent). The lock makes the status it read current: the
//     worker cannot change it until the flag is committed.
//  2. It publishes control.SignalCancel so a worker running a shell
//     command interrupts it now. A publish failure is only logged: the
//     flag is the source of truth.
//  3. If no worker holds the session (queued, or waiting_for_user), it
//     enqueues resume_session (cancel) so a worker records the
//     cancellation.
//
// Worker contract: after the worker writes waiting_for_user, and before
// it releases the lease, it must check IsCancelRequested again; a Stop
// that saw "running" relied on it.
func StopSession(ctx context.Context, d Deps, sessionID string) error {
	var status events.SessionStatus
	err := d.Store.WithTx(ctx, func(tx *store.Store) error {
		sess, err := tx.GetSessionForUpdate(ctx, sessionID)
		if err != nil {
			return err
		}
		if sess.Status.Terminal() {
			return ErrSessionFinished
		}
		status = sess.Status
		return tx.RequestCancel(ctx, sessionID)
	})
	if err != nil {
		return err
	}
	if err := d.Control.Publish(ctx, sessionID, control.SignalCancel); err != nil {
		d.logger().Warn("publish cancel signal", "session_id", sessionID, "err", err)
	}
	if status == events.StatusQueued || status == events.StatusWaitingForUser {
		if err := d.Queue.EnqueueResumeSession(ctx, queue.ResumeSessionPayload{
			SessionID: sessionID, Reason: queue.ResumeCancel,
		}); err != nil {
			return fmt.Errorf("hooks: stop session %s: %w", sessionID, err)
		}
	}
	return nil
}
