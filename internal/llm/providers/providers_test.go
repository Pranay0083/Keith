package providers

import (
	"errors"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/llm/anthropic"
	"github.com/Pranay0083/keith/internal/llm/gemini"
)

func TestNew(t *testing.T) {
	cfg := &config.Config{
		AnthropicAPIKey:    "sk-ant-test",
		GeminiAPIKey:       "gm-test",
		DefaultLLMProvider: config.LLMProviderGemini,
	}

	p, err := New("anthropic", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.(*anthropic.Provider); !ok || p.Name() != config.LLMProviderAnthropic {
		t.Errorf("New(anthropic) = %T named %q", p, p.Name())
	}

	p, err = New("gemini", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.(*gemini.Provider); !ok || p.Name() != config.LLMProviderGemini {
		t.Errorf("New(gemini) = %T named %q", p, p.Name())
	}

	p, err = New("", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != config.LLMProviderGemini {
		t.Errorf("New(\"\") = %q, want the default provider", p.Name())
	}
}

func TestNewErrors(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		cfg      *config.Config
		is       error
		mention  string
	}{
		{"missing anthropic key", "anthropic", &config.Config{GeminiAPIKey: "g"}, ErrMissingKey, "ANTHROPIC_API_KEY"},
		{"missing gemini key", "gemini", &config.Config{AnthropicAPIKey: "a"}, ErrMissingKey, "GEMINI_API_KEY"},
		{"unknown", "openai", &config.Config{AnthropicAPIKey: "a"}, ErrUnknownProvider, `"openai"`},
		{"unknown default", "", &config.Config{}, ErrUnknownProvider, `""`},
		{"nil config", "anthropic", nil, nil, "nil config"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := New(tt.provider, tt.cfg)
			if err == nil {
				t.Fatalf("New = %v, want an error", p)
			}
			if tt.is != nil && !errors.Is(err, tt.is) {
				t.Errorf("err = %v, want errors.Is %v", err, tt.is)
			}
			if !strings.Contains(err.Error(), tt.mention) {
				t.Errorf("err = %v, want it to mention %s", err, tt.mention)
			}
		})
	}
}
