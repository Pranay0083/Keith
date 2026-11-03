package hooks

import (
	"context"
	"errors"
	"fmt"

	"github.com/hibiken/asynq"

	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
)

// ErrMalformed means a webhook body did not parse. Retrying cannot help.
var ErrMalformed = errors.New("hooks: malformed webhook payload")

// Handler returns the github_webhook task handler for the worker:
//
//	queue.NewMux(queue.Handlers{GitHubWebhook: hooks.Handler(deps), ...})
//
// A malformed payload is archived (asynq.SkipRetry); other errors retry.
func Handler(d Deps) func(context.Context, queue.GitHubWebhookPayload) error {
	return func(ctx context.Context, p queue.GitHubWebhookPayload) error {
		err := ProcessGitHub(ctx, d, p.Event, p.Body)
		if errors.Is(err, ErrMalformed) {
			return fmt.Errorf("delivery %s: %w: %w", p.DeliveryID, err, asynq.SkipRetry)
		}
		if err != nil {
			return fmt.Errorf("delivery %s: %w", p.DeliveryID, err)
		}
		return nil
	}
}

// ProcessGitHub applies one verified GitHub App webhook. The signature
// was checked by the API before the task was enqueued. It is idempotent.
//
//   - installation created: if the installation is already linked to an
//     org (the setup handler ran first), refresh it and sync its repos;
//     otherwise do nothing, since only the setup handler may link it.
//   - installation deleted: mark it deleted and its repos inaccessible,
//     then stop every live session on its repos (StopSession).
//   - installation suspend / unsuspend: set suspended / active.
//   - installation_repositories added: sync the installation's repos from
//     GitHub (the payload lacks default branches).
//   - installation_repositories removed: mark those repos inaccessible.
//
// Unknown events and actions, and installations this server never
// linked, are ignored.
func ProcessGitHub(ctx context.Context, d Deps, eventType string, body []byte) error {
	switch eventType {
	case github.EventInstallation:
		ev, err := github.ParseInstallationEvent(body)
		if err != nil {
			return fmt.Errorf("%w: %v", ErrMalformed, err)
		}
		return d.installation(ctx, ev)
	case github.EventInstallationRepositories:
		ev, err := github.ParseInstallationRepositoriesEvent(body)
		if err != nil {
			return fmt.Errorf("%w: %v", ErrMalformed, err)
		}
		return d.installationRepos(ctx, ev)
	}
	return nil
}

func (d Deps) installation(ctx context.Context, ev github.InstallationEvent) error {
	log := d.logger().With("installation_id", ev.InstallationID, "action", ev.Action)
	switch ev.Action {
	case "created":
		inst, err := d.Store.GetInstallation(ctx, ev.InstallationID)
		if errors.Is(err, store.ErrNotFound) {
			log.Info("installation not linked yet; the setup handler links it")
			return nil
		}
		if err != nil {
			return err
		}
		login := ev.Account.Login
		if login == "" {
			login = inst.AccountLogin
		}
		if _, err := d.Store.UpsertInstallation(ctx, store.UpsertInstallationParams{
			OrgID: inst.OrgID, InstallationID: ev.InstallationID, AccountLogin: login, Status: store.InstallationActive,
		}); err != nil {
			return err
		}
		_, err = SyncInstallationRepos(ctx, d.Store, d.GitHub, ev.InstallationID)
		return err

	case "deleted":
		err := d.Store.SetInstallationStatus(ctx, ev.InstallationID, store.InstallationDeleted)
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		n, err := d.Store.SetInstallationReposInaccessible(ctx, ev.InstallationID)
		if err != nil {
			return err
		}
		live, err := d.Store.ListLiveSessionsByInstallation(ctx, ev.InstallationID)
		if err != nil {
			return err
		}
		var errs []error
		for _, x := range live {
			if err := StopSession(ctx, d, x.ID); err != nil && !errors.Is(err, ErrSessionFinished) && !errors.Is(err, store.ErrNotFound) {
				errs = append(errs, err)
			}
		}
		log.Info("installation deleted", "repos_marked", n, "sessions_stopped", len(live)-len(errs))
		return errors.Join(errs...)

	case "suspend", "unsuspend":
		status := store.InstallationSuspended
		if ev.Action == "unsuspend" {
			status = store.InstallationActive
		}
		err := d.Store.SetInstallationStatus(ctx, ev.InstallationID, status)
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		return err
	}
	return nil
}

func (d Deps) installationRepos(ctx context.Context, ev github.InstallationRepositoriesEvent) error {
	if _, err := d.Store.GetInstallation(ctx, ev.InstallationID); errors.Is(err, store.ErrNotFound) {
		return nil
	} else if err != nil {
		return err
	}
	switch ev.Action {
	case "added":
		_, err := SyncInstallationRepos(ctx, d.Store, d.GitHub, ev.InstallationID)
		return err
	case "removed":
		ids := make([]int64, 0, len(ev.Removed))
		for _, r := range ev.Removed {
			ids = append(ids, r.ID)
		}
		_, err := d.Store.SetReposInaccessible(ctx, ev.InstallationID, ids)
		return err
	}
	return nil
}
