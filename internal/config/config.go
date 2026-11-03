// Package config loads the service configuration from environment
// variables.
//
// Load parses and checks the format of whatever is set; Validate then
// checks that a given service role has everything it needs, since the api
// and worker need different subsets. A typical main:
//
//	cfg, err := config.Load()
//	if err == nil {
//		err = cfg.Validate(config.RoleWorker)
//	}
//
// Config holds secrets. Its String method redacts them; never log the
// fields directly.
package config

import (
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// Role is a service that runs from the keith binary.
type Role string

const (
	RoleAPI    Role = "api"
	RoleWorker Role = "worker"
)

// LLM and sandbox provider IDs accepted in DEFAULT_LLM_PROVIDER and
// SANDBOX_PROVIDER. They match llm.Provider.Name and
// sandbox.Provider.Name.
const (
	LLMProviderAnthropic   = "anthropic"
	LLMProviderGemini      = "gemini"
	SandboxProviderDocker  = "docker"
	SandboxProviderDaytona = "daytona"
)

// Defaults for unset variables.
const (
	DefaultLLMProvider     = LLMProviderAnthropic
	DefaultAnthropicModel  = "claude-opus-5-5"
	DefaultSandboxProvider = SandboxProviderDocker
	DefaultListenAddr      = ":8080"
)

// SecretsKeySize is the length in bytes of the decoded SECRETS_KEY.
const SecretsKeySize = 32

// MinSessionSecretLen is the shortest SESSION_SECRET the api accepts.
const MinSessionSecretLen = 32

// Config is the configuration of the api and worker services.
type Config struct {
	// DatabaseURL is the Postgres connection string (DATABASE_URL).
	DatabaseURL string
	// RedisURL is the Redis connection string (REDIS_URL).
	RedisURL string

	// AnthropicAPIKey (ANTHROPIC_API_KEY) and GeminiAPIKey
	// (GEMINI_API_KEY) authenticate the LLM providers.
	AnthropicAPIKey string
	GeminiAPIKey    string
	// DefaultLLMProvider (DEFAULT_LLM_PROVIDER) is the provider new
	// sessions are pinned to: "anthropic" (default) or "gemini".
	DefaultLLMProvider string
	// DefaultModel (DEFAULT_MODEL) is the model new sessions use. It
	// defaults to claude-opus-5-5 for anthropic and must be set for
	// gemini.
	DefaultModel string

	// SandboxProvider (SANDBOX_PROVIDER) is "docker" (default, local dev)
	// or "daytona".
	SandboxProvider string
	// DaytonaAPIKey (DAYTONA_API_KEY) and DaytonaAPIURL (DAYTONA_API_URL,
	// optional) configure the Daytona provider.
	DaytonaAPIKey string
	DaytonaAPIURL string

	// GitHub App settings (GITHUB_APP_ID, GITHUB_APP_CLIENT_ID,
	// GITHUB_APP_CLIENT_SECRET, GITHUB_WEBHOOK_SECRET).
	GitHubAppID           int64
	GitHubAppClientID     string
	GitHubAppClientSecret string
	GitHubWebhookSecret   string
	// GitHubAppPrivateKey is the App's PEM private key, from
	// GITHUB_APP_PRIVATE_KEY (the PEM itself; literal \n sequences are
	// turned into newlines) or the file at GITHUB_APP_PRIVATE_KEY_PATH.
	GitHubAppPrivateKey []byte

	// PublicURL (PUBLIC_URL) is the api's external base URL, used for the
	// GitHub setup and OAuth callback URLs. No trailing slash.
	PublicURL string
	// WebOrigin (WEB_ORIGIN) is the web app's origin, for CORS, WebSocket
	// origin checks and redirects. No trailing slash.
	WebOrigin string
	// SessionSecret (SESSION_SECRET) signs user session cookies.
	SessionSecret string
	// SecretsKey is the decoded SECRETS_KEY (standard base64 of 32 random
	// bytes), the key-encryption key for org and repo secrets.
	SecretsKey []byte
	// ListenAddr (LISTEN_ADDR) is the api's listen address, default :8080.
	ListenAddr string
}

// Load reads the configuration from the process environment.
func Load() (*Config, error) {
	return LoadFrom(os.Getenv)
}

// LoadFrom reads the configuration through getenv, which returns "" for
// unset variables. It fails if a set variable is malformed; it does not
// check that required variables are present (see Validate).
func LoadFrom(getenv func(string) string) (*Config, error) {
	get := func(name string) string { return strings.TrimSpace(getenv(name)) }
	var errs []error

	c := &Config{
		DatabaseURL:           get("DATABASE_URL"),
		RedisURL:              get("REDIS_URL"),
		AnthropicAPIKey:       get("ANTHROPIC_API_KEY"),
		GeminiAPIKey:          get("GEMINI_API_KEY"),
		DefaultLLMProvider:    orDefault(get("DEFAULT_LLM_PROVIDER"), DefaultLLMProvider),
		DefaultModel:          get("DEFAULT_MODEL"),
		SandboxProvider:       orDefault(get("SANDBOX_PROVIDER"), DefaultSandboxProvider),
		DaytonaAPIKey:         get("DAYTONA_API_KEY"),
		DaytonaAPIURL:         get("DAYTONA_API_URL"),
		GitHubAppClientID:     get("GITHUB_APP_CLIENT_ID"),
		GitHubAppClientSecret: get("GITHUB_APP_CLIENT_SECRET"),
		GitHubWebhookSecret:   get("GITHUB_WEBHOOK_SECRET"),
		SessionSecret:         get("SESSION_SECRET"),
		ListenAddr:            orDefault(get("LISTEN_ADDR"), DefaultListenAddr),
	}

	switch c.DefaultLLMProvider {
	case LLMProviderAnthropic:
		c.DefaultModel = orDefault(c.DefaultModel, DefaultAnthropicModel)
	case LLMProviderGemini:
	default:
		errs = append(errs, fmt.Errorf("DEFAULT_LLM_PROVIDER: unknown provider %q (want %s or %s)",
			c.DefaultLLMProvider, LLMProviderAnthropic, LLMProviderGemini))
	}

	switch c.SandboxProvider {
	case SandboxProviderDocker, SandboxProviderDaytona:
	default:
		errs = append(errs, fmt.Errorf("SANDBOX_PROVIDER: unknown provider %q (want %s or %s)",
			c.SandboxProvider, SandboxProviderDocker, SandboxProviderDaytona))
	}

	if v := get("GITHUB_APP_ID"); v != "" {
		id, err := strconv.ParseInt(v, 10, 64)
		if err != nil || id <= 0 {
			errs = append(errs, fmt.Errorf("GITHUB_APP_ID: %q is not a positive integer", v))
		}
		c.GitHubAppID = id
	}

	key, err := loadPrivateKey(get("GITHUB_APP_PRIVATE_KEY"), get("GITHUB_APP_PRIVATE_KEY_PATH"))
	if err != nil {
		errs = append(errs, err)
	}
	c.GitHubAppPrivateKey = key

	if v := get("SECRETS_KEY"); v != "" {
		k, err := base64.StdEncoding.DecodeString(v)
		switch {
		case err != nil:
			errs = append(errs, errors.New("SECRETS_KEY: not valid standard base64"))
		case len(k) != SecretsKeySize:
			errs = append(errs, fmt.Errorf("SECRETS_KEY: decodes to %d bytes, want %d", len(k), SecretsKeySize))
		default:
			c.SecretsKey = k
		}
	}

	for _, u := range []struct {
		name string
		dst  *string
	}{{"PUBLIC_URL", &c.PublicURL}, {"WEB_ORIGIN", &c.WebOrigin}} {
		v, err := baseURL(u.name, get(u.name))
		if err != nil {
			errs = append(errs, err)
		}
		*u.dst = v
	}

	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	return c, nil
}

// Validate checks that every variable role needs is set.
//
// Both roles need DATABASE_URL, REDIS_URL, SECRETS_KEY, GITHUB_APP_ID and
// the App private key (the api mints tokens for setup and repo sync; the
// worker mints the per-session installation token and opens PRs).
// The api also needs the App's OAuth client, the webhook secret,
// PUBLIC_URL, WEB_ORIGIN and SESSION_SECRET. The worker also needs the
// default LLM provider's key and model, and Daytona credentials when
// SANDBOX_PROVIDER is daytona.
func (c *Config) Validate(role Role) error {
	var missing []string
	need := func(name string, ok bool) {
		if !ok {
			missing = append(missing, name)
		}
	}

	need("DATABASE_URL", c.DatabaseURL != "")
	need("REDIS_URL", c.RedisURL != "")
	need("SECRETS_KEY", len(c.SecretsKey) == SecretsKeySize)
	need("GITHUB_APP_ID", c.GitHubAppID > 0)
	need("GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_PATH", len(c.GitHubAppPrivateKey) > 0)

	var errs []error
	switch role {
	case RoleAPI:
		need("GITHUB_APP_CLIENT_ID", c.GitHubAppClientID != "")
		need("GITHUB_APP_CLIENT_SECRET", c.GitHubAppClientSecret != "")
		need("GITHUB_WEBHOOK_SECRET", c.GitHubWebhookSecret != "")
		need("PUBLIC_URL", c.PublicURL != "")
		need("WEB_ORIGIN", c.WebOrigin != "")
		need("SESSION_SECRET", c.SessionSecret != "")
		if c.SessionSecret != "" && len(c.SessionSecret) < MinSessionSecretLen {
			errs = append(errs, fmt.Errorf("SESSION_SECRET: must be at least %d characters", MinSessionSecretLen))
		}
	case RoleWorker:
		switch c.DefaultLLMProvider {
		case LLMProviderAnthropic:
			need("ANTHROPIC_API_KEY", c.AnthropicAPIKey != "")
		case LLMProviderGemini:
			need("GEMINI_API_KEY", c.GeminiAPIKey != "")
		}
		need("DEFAULT_MODEL", c.DefaultModel != "")
		if c.SandboxProvider == SandboxProviderDaytona {
			need("DAYTONA_API_KEY", c.DaytonaAPIKey != "")
		}
	default:
		return fmt.Errorf("config: unknown role %q", role)
	}

	if len(missing) > 0 {
		errs = append([]error{fmt.Errorf("missing %s", strings.Join(missing, ", "))}, errs...)
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("config for %s: %w", role, err)
	}
	return nil
}

// String describes the configuration with every secret replaced by "set"
// or "unset", so it is safe to log.
func (c *Config) String() string {
	set := func(ok bool) string {
		if ok {
			return "set"
		}
		return "unset"
	}
	return fmt.Sprintf("Config{DatabaseURL:%s RedisURL:%s AnthropicAPIKey:%s GeminiAPIKey:%s "+
		"DefaultLLMProvider:%s DefaultModel:%s SandboxProvider:%s DaytonaAPIKey:%s DaytonaAPIURL:%s "+
		"GitHubAppID:%d GitHubAppClientID:%s GitHubAppClientSecret:%s GitHubWebhookSecret:%s "+
		"GitHubAppPrivateKey:%s PublicURL:%s WebOrigin:%s SessionSecret:%s SecretsKey:%s ListenAddr:%s}",
		set(c.DatabaseURL != ""), set(c.RedisURL != ""), set(c.AnthropicAPIKey != ""), set(c.GeminiAPIKey != ""),
		c.DefaultLLMProvider, c.DefaultModel, c.SandboxProvider, set(c.DaytonaAPIKey != ""), c.DaytonaAPIURL,
		c.GitHubAppID, c.GitHubAppClientID, set(c.GitHubAppClientSecret != ""), set(c.GitHubWebhookSecret != ""),
		set(len(c.GitHubAppPrivateKey) > 0), c.PublicURL, c.WebOrigin, set(c.SessionSecret != ""),
		set(len(c.SecretsKey) > 0), c.ListenAddr)
}

// GoString keeps %#v from printing secrets.
func (c *Config) GoString() string { return c.String() }

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func loadPrivateKey(inline, path string) ([]byte, error) {
	var key []byte
	switch {
	case inline != "" && path != "":
		return nil, errors.New("GITHUB_APP_PRIVATE_KEY and GITHUB_APP_PRIVATE_KEY_PATH: set only one")
	case inline != "":
		if !strings.Contains(inline, "\n") {
			inline = strings.ReplaceAll(inline, `\n`, "\n")
		}
		key = []byte(inline)
	case path != "":
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("GITHUB_APP_PRIVATE_KEY_PATH: %w", err)
		}
		key = b
	default:
		return nil, nil
	}
	if block, _ := pem.Decode(key); block == nil {
		return nil, errors.New("GITHUB_APP_PRIVATE_KEY: not a PEM-encoded key")
	}
	return key, nil
}

func baseURL(name, v string) (string, error) {
	if v == "" {
		return "", nil
	}
	u, err := url.Parse(v)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("%s: %q is not an absolute http(s) URL", name, v)
	}
	return strings.TrimRight(v, "/"), nil
}
