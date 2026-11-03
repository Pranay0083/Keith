package github

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	gh "github.com/google/go-github/v93/github"
)

// maxPages bounds every paginated listing, at 100 items per page.
const maxPages = 100

// UserToken is a user access token from the App's OAuth flow. String,
// GoString and LogValue redact it.
type UserToken struct {
	AccessToken string
	TokenType   string
	Scope       string
	// ExpiresAt and the refresh fields are zero when the App has user
	// token expiration turned off.
	ExpiresAt             time.Time
	RefreshToken          string
	RefreshTokenExpiresAt time.Time
}

func (t UserToken) String() string {
	exp := "never"
	if !t.ExpiresAt.IsZero() {
		exp = t.ExpiresAt.UTC().Format(time.RFC3339)
	}
	return "github.UserToken{AccessToken:REDACTED ExpiresAt:" + exp + "}"
}

// GoString keeps %#v from printing the token.
func (t UserToken) GoString() string { return t.String() }

// LogValue keeps slog from printing the token.
func (t UserToken) LogValue() slog.Value { return slog.StringValue(t.String()) }

// User is a GitHub account.
type User struct {
	ID    int64
	Login string
	Name  string
	// Email is the public profile email; it is often empty.
	Email     string
	AvatarURL string
}

func (a *App) requireOAuthClient() error {
	if a.clientID == "" || a.clientSecret == "" {
		return errors.New("github: OAuth needs the App's client ID and client secret")
	}
	return nil
}

// AuthCodeURL is the URL that starts the OAuth flow, for sign-in. state
// must be a fresh unguessable value tied to the browser session, checked
// again on the callback; an empty state is an error.
func (a *App) AuthCodeURL(state string) (string, error) {
	if err := a.requireOAuthClient(); err != nil {
		return "", err
	}
	if state == "" {
		return "", errors.New("github: AuthCodeURL needs a state value")
	}
	q := url.Values{"client_id": {a.clientID}, "state": {state}}
	if a.redirectURL != "" {
		q.Set("redirect_uri", a.redirectURL)
	}
	return a.webBase + "/login/oauth/authorize?" + q.Encode(), nil
}

var slugRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// InstallURL is the URL that starts an App installation. GitHub carries
// state through the install and, with "Request user authorization during
// installation" on, sends it back to the callback with the OAuth code and
// installation_id. Check state there as for sign-in, then Exchange the
// code and call UserCanAccessInstallation.
func (a *App) InstallURL(appSlug, state string) (string, error) {
	if !slugRe.MatchString(appSlug) {
		return "", fmt.Errorf("github: invalid App slug %q", appSlug)
	}
	if state == "" {
		return "", errors.New("github: InstallURL needs a state value")
	}
	return a.webBase + "/apps/" + appSlug + "/installations/new?" + url.Values{"state": {state}}.Encode(), nil
}

// Slug returns the App's slug (GET /app), the name in its github.com URL.
func (a *App) Slug(ctx context.Context) (string, error) {
	c, err := a.appClient()
	if err != nil {
		return "", err
	}
	app, resp, err := c.Apps.Get(ctx, "")
	if err != nil {
		return "", wrapErr("get app", resp, err)
	}
	return app.GetSlug(), nil
}

// Exchange trades an OAuth code, from sign-in or from the install-time
// user authorization, for a user access token.
func (a *App) Exchange(ctx context.Context, code string) (UserToken, error) {
	if err := a.requireOAuthClient(); err != nil {
		return UserToken{}, err
	}
	if code == "" {
		return UserToken{}, errors.New("github: oauth exchange: empty code")
	}
	form := url.Values{
		"client_id":     {a.clientID},
		"client_secret": {a.clientSecret},
		"code":          {code},
	}
	if a.redirectURL != "" {
		form.Set("redirect_uri", a.redirectURL)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.webBase+"/login/oauth/access_token", strings.NewReader(form.Encode()))
	if err != nil {
		return UserToken{}, fmt.Errorf("github: oauth exchange: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", userAgent)

	resp, err := a.hc.Do(req)
	if err != nil {
		return UserToken{}, fmt.Errorf("github: oauth exchange: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return UserToken{}, fmt.Errorf("github: oauth exchange: status %s", resp.Status)
	}
	var body struct {
		AccessToken           string `json:"access_token"`
		TokenType             string `json:"token_type"`
		Scope                 string `json:"scope"`
		ExpiresIn             int64  `json:"expires_in"`
		RefreshToken          string `json:"refresh_token"`
		RefreshTokenExpiresIn int64  `json:"refresh_token_expires_in"`
		Error                 string `json:"error"`
		ErrorDescription      string `json:"error_description"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<16)).Decode(&body); err != nil {
		return UserToken{}, fmt.Errorf("github: oauth exchange: decode response: %w", err)
	}
	if body.Error != "" {
		return UserToken{}, fmt.Errorf("github: oauth exchange: %s: %s", body.Error, body.ErrorDescription)
	}
	if body.AccessToken == "" {
		return UserToken{}, errors.New("github: oauth exchange: response has no access token")
	}

	now := a.clock.Now()
	t := UserToken{
		AccessToken:  body.AccessToken,
		TokenType:    body.TokenType,
		Scope:        body.Scope,
		RefreshToken: body.RefreshToken,
	}
	if body.ExpiresIn > 0 {
		t.ExpiresAt = now.Add(time.Duration(body.ExpiresIn) * time.Second)
	}
	if body.RefreshTokenExpiresIn > 0 {
		t.RefreshTokenExpiresAt = now.Add(time.Duration(body.RefreshTokenExpiresIn) * time.Second)
	}
	return t, nil
}

// GetUser returns the user a user access token belongs to (GET /user).
func (a *App) GetUser(ctx context.Context, userToken string) (User, error) {
	if userToken == "" {
		return User{}, errors.New("github: get user: empty token")
	}
	c, err := a.client(userToken)
	if err != nil {
		return User{}, err
	}
	u, resp, err := c.Users.Get(ctx, "")
	if err != nil {
		return User{}, wrapErr("get user", resp, err)
	}
	return User{
		ID:        u.GetID(),
		Login:     u.GetLogin(),
		Name:      u.GetName(),
		Email:     u.GetEmail(),
		AvatarURL: u.GetAvatarURL(),
	}, nil
}

// UserCanAccessInstallation reports whether the user behind userToken
// has access to installationID of this App, by paging through
// GET /user/installations. The setup handler must check this before it
// links an installation to an org: anyone can call the setup URL with
// someone else's installation_id. An invalid token is an error, not
// false.
func (a *App) UserCanAccessInstallation(ctx context.Context, userToken string, installationID int64) (bool, error) {
	if userToken == "" {
		return false, errors.New("github: list user installations: empty token")
	}
	if installationID <= 0 {
		return false, errors.New("github: installation ID must be positive")
	}
	c, err := a.client(userToken)
	if err != nil {
		return false, err
	}
	opts := &gh.ListOptions{PerPage: 100}
	for range maxPages {
		insts, resp, err := c.Apps.ListUserInstallations(ctx, opts)
		if err != nil {
			return false, wrapErr("list user installations", resp, err)
		}
		for _, in := range insts {
			// /user/installations lists only this App's installations
			// for a token this App issued; the app_id check is a second
			// line of defence.
			if in.GetID() == installationID && (in.GetAppID() == 0 || in.GetAppID() == a.appID) {
				return true, nil
			}
		}
		if resp.NextPage == 0 {
			return false, nil
		}
		opts.Page = resp.NextPage
	}
	return false, fmt.Errorf("github: list user installations: more than %d pages", maxPages)
}
