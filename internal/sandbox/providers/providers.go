// Package providers builds the sandbox.Provider the configuration selects.
// It is the only place that imports every provider package, so the worker
// and CLI depend on one constructor rather than on each SDK.
package providers

import (
	"errors"
	"fmt"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/daytona"
	"github.com/Pranay0083/keith/internal/sandbox/docker"
)

// New returns the provider named by cfg.SandboxProvider:
//   - "docker": local containers from docker.DefaultImage
//     (sandbox/base.Dockerfile) unless a Spec names a template;
//   - "daytona": Daytona with DAYTONA_API_KEY and DAYTONA_API_URL.
//
// It does not contact the provider.
func New(cfg *config.Config) (sandbox.Provider, error) {
	if cfg == nil {
		return nil, errors.New("providers: nil config")
	}
	switch cfg.SandboxProvider {
	case config.SandboxProviderDocker, "":
		return docker.New(docker.Options{}), nil
	case config.SandboxProviderDaytona:
		if cfg.DaytonaAPIKey == "" {
			return nil, errors.New("providers: SANDBOX_PROVIDER=daytona needs DAYTONA_API_KEY")
		}
		p, err := daytona.New(daytona.Options{APIKey: cfg.DaytonaAPIKey, APIURL: cfg.DaytonaAPIURL})
		if err != nil {
			return nil, fmt.Errorf("providers: %w", err)
		}
		return p, nil
	default:
		return nil, fmt.Errorf("providers: unknown sandbox provider %q (want %s or %s)",
			cfg.SandboxProvider, config.SandboxProviderDocker, config.SandboxProviderDaytona)
	}
}
