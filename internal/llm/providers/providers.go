// Package providers builds an llm.Provider by name from the service
// configuration. It is the only place that knows every provider package,
// so the worker can resolve sessions.provider without importing them.
package providers

import (
	"errors"
	"fmt"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/llm/anthropic"
	"github.com/Pranay0083/keith/internal/llm/gemini"
)

// ErrUnknownProvider is returned for a provider name New doesn't know.
var ErrUnknownProvider = errors.New("unknown llm provider")

// ErrMissingKey is returned when the provider's API key is not configured.
var ErrMissingKey = errors.New("llm provider API key is not set")

// New returns the provider called name ("anthropic" or "gemini"),
// authenticated with its key from cfg. An empty name means
// cfg.DefaultLLMProvider.
func New(name string, cfg *config.Config) (llm.Provider, error) {
	if cfg == nil {
		return nil, errors.New("llm providers: nil config")
	}
	if name == "" {
		name = cfg.DefaultLLMProvider
	}
	switch name {
	case config.LLMProviderAnthropic:
		if cfg.AnthropicAPIKey == "" {
			return nil, fmt.Errorf("llm provider %s: %w: set ANTHROPIC_API_KEY", name, ErrMissingKey)
		}
		return anthropic.New(cfg.AnthropicAPIKey), nil
	case config.LLMProviderGemini:
		if cfg.GeminiAPIKey == "" {
			return nil, fmt.Errorf("llm provider %s: %w: set GEMINI_API_KEY", name, ErrMissingKey)
		}
		p, err := gemini.New(cfg.GeminiAPIKey)
		if err != nil {
			return nil, fmt.Errorf("llm provider %s: %w", name, err)
		}
		return p, nil
	default:
		return nil, fmt.Errorf("%w %q (want %s or %s)", ErrUnknownProvider, name,
			config.LLMProviderAnthropic, config.LLMProviderGemini)
	}
}
