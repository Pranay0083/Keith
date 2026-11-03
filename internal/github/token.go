package github

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"time"

	gh "github.com/google/go-github/v93/github"
)

// Access is a permission level on an installation token.
type Access string

const (
	Read  Access = "read"
	Write Access = "write"
)

// Permissions are the repository permissions requested for an
// installation token. Unset fields are not requested. A token can never
// exceed what the installation granted the App.
type Permissions struct {
	Contents     Access
	Metadata     Access
	PullRequests Access
	Issues       Access
	Checks       Access
	Actions      Access
}

var (
	// SandboxPermissions is the sandbox token: clone and push, nothing
	// else.
	SandboxPermissions = Permissions{Contents: Write}
	// PushPermissions is the worker's push token: it updates a session's
	// agent/<session-id> branch from outside the sandbox (ReceivePack),
	// one repo per token, revoked right after the push. It never enters
	// the sandbox.
	PushPermissions = Permissions{Contents: Write}
	// WorkerPermissions is the worker's own token for reading issues and
	// opening PRs from outside the sandbox.
	WorkerPermissions = Permissions{
		Contents:     Read,
		PullRequests: Write,
		Issues:       Read,
		Metadata:     Read,
	}
)

// IsZero reports whether no permission is set.
func (p Permissions) IsZero() bool { return p == Permissions{} }

func (p Permissions) validate() error {
	if p.IsZero() {
		return errors.New("github: no permissions requested")
	}
	for name, v := range map[string]Access{
		"contents": p.Contents, "metadata": p.Metadata, "pull_requests": p.PullRequests,
		"issues": p.Issues, "checks": p.Checks, "actions": p.Actions,
	} {
		if v != "" && v != Read && v != Write {
			return fmt.Errorf("github: permission %s: invalid access %q", name, v)
		}
	}
	if p.Metadata == Write {
		return errors.New("github: permission metadata is read-only")
	}
	return nil
}

func (p Permissions) gh() *gh.InstallationPermissions {
	ptr := func(a Access) *string {
		if a == "" {
			return nil
		}
		s := string(a)
		return &s
	}
	return &gh.InstallationPermissions{
		Contents:     ptr(p.Contents),
		Metadata:     ptr(p.Metadata),
		PullRequests: ptr(p.PullRequests),
		Issues:       ptr(p.Issues),
		Checks:       ptr(p.Checks),
		Actions:      ptr(p.Actions),
	}
}

// Token is an installation access token. String, GoString and LogValue
// redact the value; read Value only where the token is used.
type Token struct {
	Value     string
	ExpiresAt time.Time
}

func (t Token) String() string {
	return "github.Token{Value:REDACTED ExpiresAt:" + t.ExpiresAt.UTC().Format(time.RFC3339) + "}"
}

// GoString keeps %#v from printing the token.
func (t Token) GoString() string { return t.String() }

// LogValue keeps slog from printing the token.
func (t Token) LogValue() slog.Value { return slog.StringValue(t.String()) }

// ErrNoRepos is returned when a token is requested without repos.
var ErrNoRepos = errors.New("github: an installation token must be scoped to at least one repo")

var (
	ownerRe    = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$`)
	repoNameRe = regexp.MustCompile(`^[A-Za-z0-9._-]{1,100}$`)
)

func validOwner(s string) bool { return ownerRe.MatchString(s) }

func validRepoName(s string) bool {
	return repoNameRe.MatchString(s) && s != "." && s != ".."
}

// splitRepo splits and checks owner/name.
func splitRepo(full string) (owner, name string, err error) {
	owner, name, ok := strings.Cut(full, "/")
	if !ok || !validOwner(owner) || !validRepoName(name) {
		return "", "", fmt.Errorf("github: %q is not an owner/name repo", full)
	}
	return owner, name, nil
}

// repoNames turns repos (owner/name or bare names) into the bare names
// the access_tokens endpoint takes, in the given order. Names resolve
// within the installation's account, so every owner given must match.
func repoNames(repos []string) ([]string, error) {
	if len(repos) == 0 {
		return nil, ErrNoRepos
	}
	names := make([]string, 0, len(repos))
	owner := ""
	for _, r := range repos {
		o, n, hasOwner := strings.Cut(r, "/")
		if !hasOwner {
			o, n = "", r
		}
		if !validRepoName(n) || (hasOwner && !validOwner(o)) {
			return nil, fmt.Errorf("github: invalid repo %q", r)
		}
		if o != "" {
			if owner == "" {
				owner = o
			} else if !strings.EqualFold(owner, o) {
				return nil, fmt.Errorf("github: repo %q is not owned by %q; one token covers one installation", r, owner)
			}
		}
		names = append(names, n)
	}
	return names, nil
}

// InstallationToken mints a 1-hour installation token limited to repos
// (owner/name or bare names, all in the installation's account) and
// perms. A nil or empty repos, or zero perms, is an error: this never
// mints an unscoped token.
func (a *App) InstallationToken(ctx context.Context, installationID int64, repos []string, perms Permissions) (Token, error) {
	if installationID <= 0 {
		return Token{}, errors.New("github: installation ID must be positive")
	}
	names, err := repoNames(repos)
	if err != nil {
		return Token{}, err
	}
	if err := perms.validate(); err != nil {
		return Token{}, err
	}
	return a.mintToken(ctx, installationID, &gh.InstallationTokenOptions{
		Repositories: names,
		Permissions:  perms.gh(),
	})
}

func (a *App) mintToken(ctx context.Context, installationID int64, opts *gh.InstallationTokenOptions) (Token, error) {
	c, err := a.appClient()
	if err != nil {
		return Token{}, err
	}
	t, resp, err := c.Apps.CreateInstallationToken(ctx, installationID, opts)
	if err != nil {
		return Token{}, wrapErr(fmt.Sprintf("create token for installation %d", installationID), resp, err)
	}
	if t.GetToken() == "" || t.ExpiresAt == nil {
		return Token{}, fmt.Errorf("github: create token for installation %d: response has no token or expiry", installationID)
	}
	return Token{Value: t.GetToken(), ExpiresAt: t.GetExpiresAt().Time}, nil
}

// RevokeToken revokes an installation token (DELETE /installation/token,
// authenticated with the token itself). A token GitHub already rejects
// (expired or revoked) counts as revoked.
func (a *App) RevokeToken(ctx context.Context, token string) error {
	if token == "" {
		return errors.New("github: revoke: empty token")
	}
	c, err := a.client(token)
	if err != nil {
		return err
	}
	resp, err := c.Apps.RevokeInstallationToken(ctx)
	if err != nil {
		if statusCode(resp) == http.StatusUnauthorized {
			return nil
		}
		return wrapErr("revoke installation token", resp, err)
	}
	return nil
}
