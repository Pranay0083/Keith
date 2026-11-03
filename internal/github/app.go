package github

import (
	"crypto/rsa"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/bradleyfalzon/ghinstallation/v2"
	jwt "github.com/golang-jwt/jwt/v4"
	gh "github.com/google/go-github/v93/github"
)

// Default GitHub endpoints.
const (
	DefaultAPIBaseURL = "https://api.github.com/"
	DefaultWebBaseURL = "https://github.com"
)

const (
	// The App JWT is backdated by a minute against clock drift and
	// expires 9 minutes from now; GitHub rejects an exp more than 10
	// minutes ahead.
	jwtBackdate = 60 * time.Second
	jwtLifetime = 540 * time.Second

	defaultHTTPTimeout = 30 * time.Second
	userAgent          = "keith"
)

// ErrNotFound marks a 404 from GitHub: the installation, repo, issue or
// pull request does not exist, or the token cannot see it.
var ErrNotFound = errors.New("not found")

// Clock is the time source for JWTs and token refresh. Tests inject a
// fake one.
type Clock interface {
	Now() time.Time
	// After returns a channel that receives once d has elapsed.
	After(d time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                         { return time.Now() }
func (realClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// AppConfig configures an App.
type AppConfig struct {
	// AppID is the GitHub App's numeric ID (GITHUB_APP_ID). Required.
	AppID int64
	// ClientID and ClientSecret are the App's OAuth client
	// (GITHUB_APP_CLIENT_ID, GITHUB_APP_CLIENT_SECRET). The OAuth methods
	// need both. When ClientID is set it is also the App JWT's issuer, as
	// GitHub recommends; otherwise AppID is.
	ClientID     string
	ClientSecret string
	// PrivateKeyPEM is the App's RSA private key in PEM form (PKCS#1 or
	// PKCS#8). Required.
	PrivateKeyPEM []byte
	// RedirectURL is the OAuth callback URL sent with the authorize and
	// token requests. Optional when the App has a single callback URL.
	RedirectURL string

	// APIBaseURL is the REST API base, default DefaultAPIBaseURL.
	APIBaseURL string
	// WebBaseURL is the github.com web base used for OAuth and install
	// URLs, default DefaultWebBaseURL.
	WebBaseURL string
	// HTTPClient makes every request. The default has a 30 s timeout.
	HTTPClient *http.Client
	// Clock defaults to the system clock.
	Clock Clock
}

// App is a GitHub App: it signs App JWTs, mints installation tokens and
// runs the OAuth flow. It is safe for concurrent use.
type App struct {
	appID        int64
	clientID     string
	clientSecret string
	redirectURL  string
	signer       ghinstallation.Signer
	apiBase      string
	webBase      string
	hc           *http.Client
	clock        Clock
}

// NewApp checks cfg and parses the private key.
func NewApp(cfg AppConfig) (*App, error) {
	if cfg.AppID <= 0 {
		return nil, errors.New("github: AppID must be positive")
	}
	if len(cfg.PrivateKeyPEM) == 0 {
		return nil, errors.New("github: PrivateKeyPEM is empty")
	}
	key, err := jwt.ParseRSAPrivateKeyFromPEM(cfg.PrivateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("github: parse App private key: %w", err)
	}

	a := &App{
		appID:        cfg.AppID,
		clientID:     cfg.ClientID,
		clientSecret: cfg.ClientSecret,
		redirectURL:  cfg.RedirectURL,
		signer:       newSigner(key),
		hc:           cfg.HTTPClient,
		clock:        cfg.Clock,
	}
	if a.hc == nil {
		a.hc = &http.Client{Timeout: defaultHTTPTimeout}
	}
	if a.clock == nil {
		a.clock = realClock{}
	}

	if a.apiBase, err = baseURL("APIBaseURL", cfg.APIBaseURL, DefaultAPIBaseURL); err != nil {
		return nil, err
	}
	if !strings.HasSuffix(a.apiBase, "/") {
		a.apiBase += "/"
	}
	if a.webBase, err = baseURL("WebBaseURL", cfg.WebBaseURL, DefaultWebBaseURL); err != nil {
		return nil, err
	}
	a.webBase = strings.TrimRight(a.webBase, "/")
	return a, nil
}

func newSigner(key *rsa.PrivateKey) ghinstallation.Signer {
	return ghinstallation.NewRSASigner(jwt.SigningMethodRS256, key)
}

func baseURL(name, v, def string) (string, error) {
	if v == "" {
		return def, nil
	}
	u, err := url.Parse(v)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.User != nil {
		return "", fmt.Errorf("github: %s %q is not an absolute http(s) URL", name, v)
	}
	return v, nil
}

// AppID returns the App's numeric ID.
func (a *App) AppID() int64 { return a.appID }

// issuer is the App JWT's iss claim.
func (a *App) issuer() string {
	if a.clientID != "" {
		return a.clientID
	}
	return strconv.FormatInt(a.appID, 10)
}

// appJWT signs a fresh App JWT: RS256, iat now-60s, exp now+540s.
func (a *App) appJWT() (string, error) {
	now := a.clock.Now()
	claims := &jwt.RegisteredClaims{
		Issuer:    a.issuer(),
		IssuedAt:  jwt.NewNumericDate(now.Add(-jwtBackdate)),
		ExpiresAt: jwt.NewNumericDate(now.Add(jwtLifetime)),
	}
	s, err := a.signer.Sign(claims)
	if err != nil {
		return "", fmt.Errorf("github: sign App JWT: %w", err)
	}
	return s, nil
}

// client returns a go-github client that sends token as a bearer
// credential, and only to the configured API origin.
func (a *App) client(token string) (*gh.Client, error) {
	c, err := gh.NewClient(
		gh.WithHTTPClient(a.hc),
		gh.WithURLs(&a.apiBase, &a.apiBase),
		gh.WithUserAgent(userAgent),
		gh.WithAuthToken(token),
	)
	if err != nil {
		return nil, fmt.Errorf("github: build client: %w", err)
	}
	return c, nil
}

// appClient returns a client authenticated as the App itself.
func (a *App) appClient() (*gh.Client, error) {
	t, err := a.appJWT()
	if err != nil {
		return nil, err
	}
	return a.client(t)
}

// wrapErr adds the operation to a go-github error and marks 404s with
// ErrNotFound.
func wrapErr(op string, resp *gh.Response, err error) error {
	if statusCode(resp) == http.StatusNotFound {
		return fmt.Errorf("github: %s: %w: %w", op, ErrNotFound, err)
	}
	return fmt.Errorf("github: %s: %w", op, err)
}

func statusCode(resp *gh.Response) int {
	if resp == nil || resp.Response == nil {
		return 0
	}
	return resp.StatusCode
}
