package github

import (
	"context"
	"errors"
	"fmt"

	gh "github.com/google/go-github/v93/github"
)

// Installation is an installation of the App on a user or org account.
type Installation struct {
	ID           int64
	AccountID    int64
	AccountLogin string
	// AccountType is "Organization" or "User".
	AccountType string
	// RepositorySelection is "all" or "selected".
	RepositorySelection string
	Suspended           bool
}

// Repo is a repository an installation can access.
type Repo struct {
	ID            int64
	FullName      string
	DefaultBranch string
	Private       bool
}

// GetInstallation returns an installation of this App
// (GET /app/installations/{id}). A missing one is ErrNotFound.
func (a *App) GetInstallation(ctx context.Context, installationID int64) (Installation, error) {
	if installationID <= 0 {
		return Installation{}, errors.New("github: installation ID must be positive")
	}
	c, err := a.appClient()
	if err != nil {
		return Installation{}, err
	}
	in, resp, err := c.Apps.GetInstallation(ctx, installationID)
	if err != nil {
		return Installation{}, wrapErr(fmt.Sprintf("get installation %d", installationID), resp, err)
	}
	return Installation{
		ID:                  in.GetID(),
		AccountID:           in.GetAccount().GetID(),
		AccountLogin:        in.GetAccount().GetLogin(),
		AccountType:         in.GetAccount().GetType(),
		RepositorySelection: in.GetRepositorySelection(),
		Suspended:           in.SuspendedAt != nil,
	}, nil
}

// ListInstallationRepos returns every repo the installation can access
// (GET /installation/repositories, paginated).
//
// Listing needs an installation token for the whole installation, so
// this mints one with metadata: read only, uses it in-process and
// revokes it. It is the only token this package mints without a repo
// list, and it never leaves this function.
func (a *App) ListInstallationRepos(ctx context.Context, installationID int64) ([]Repo, error) {
	if installationID <= 0 {
		return nil, errors.New("github: installation ID must be positive")
	}
	read := string(Read)
	tok, err := a.mintToken(ctx, installationID, &gh.InstallationTokenOptions{
		Permissions: &gh.InstallationPermissions{Metadata: &read},
	})
	if err != nil {
		return nil, err
	}
	defer func() {
		rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), revokeTimeout)
		defer cancel()
		_ = a.RevokeToken(rctx, tok.Value)
	}()

	c, err := a.client(tok.Value)
	if err != nil {
		return nil, err
	}
	var repos []Repo
	opts := &gh.ListOptions{PerPage: 100}
	for range maxPages {
		list, resp, err := c.Apps.ListRepos(ctx, opts)
		if err != nil {
			return nil, wrapErr(fmt.Sprintf("list repos of installation %d", installationID), resp, err)
		}
		for _, r := range list.Repositories {
			repos = append(repos, Repo{
				ID:            r.GetID(),
				FullName:      r.GetFullName(),
				DefaultBranch: r.GetDefaultBranch(),
				Private:       r.GetPrivate(),
			})
		}
		if resp.NextPage == 0 {
			return repos, nil
		}
		opts.Page = resp.NextPage
	}
	return nil, fmt.Errorf("github: list repos of installation %d: more than %d pages", installationID, maxPages)
}
