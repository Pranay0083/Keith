package models

import "github.com/Pranay0083/keith/internal/config"

type Model struct {
	ID   string // Model ID sent to the API
	Name string // Human-readable display name
}

// Antigravity models (discovered via fetchAvailableModels API endpoint)
var antigravityModels = map[string]Model{
	"gemini-3.1-pro-high": {ID: "gemini-3.1-pro-high", Name: "Gemini 3.1 Pro (High Thinking)"},
	"gemini-3.1-pro-low":  {ID: "gemini-3.1-pro-low", Name: "Gemini 3.1 Pro (Low Thinking)"},
	"gemini-3-flash":      {ID: "gemini-3-flash", Name: "Gemini 3 Flash"},
	"claude-sonnet-4-6":   {ID: "claude-sonnet-4-6", Name: "Claude Sonnet 4.6"},
	"claude-opus-4-6":     {ID: "claude-opus-4-6-thinking", Name: "Claude Opus 4.6 (Thinking)"},
	"gpt-oss-120b":        {ID: "gpt-oss-120b-medium", Name: "GPT-OSS 120B (Medium)"},
}

// Gemini API models (standard generativelanguage.googleapis.com)
var geminiModels = map[string]Model{
	"gemini-3.1-pro":       {ID: "gemini-3.1-pro", Name: "Gemini 3.1 Pro"},
	"gemini-3-pro":         {ID: "gemini-3-pro", Name: "Gemini 3 Pro"},
	"gemini-3-flash":       {ID: "gemini-3-flash", Name: "Gemini 3 Flash"},
	"gemini-2.5-flash":     {ID: "gemini-2.5-flash-preview-05-20", Name: "Gemini 2.5 Flash"},
	"gemini-2.5-pro":       {ID: "gemini-2.5-pro-preview-05-06", Name: "Gemini 2.5 Pro"},
	"gemini-2.0-flash":     {ID: "gemini-2.0-flash", Name: "Gemini 2.0 Flash"},
	"gemini-2.0-flash-exp": {ID: "gemini-2.0-flash-exp", Name: "Gemini 2.0 Flash (Experimental)"},
	"gemini-1.5-pro":       {ID: "gemini-1.5-pro", Name: "Gemini 1.5 Pro"},
	"gemini-1.5-flash":     {ID: "gemini-1.5-flash", Name: "Gemini 1.5 Flash"},
}

// Registry is the combined set of all models (used for backward compat).
var Registry = func() map[string]Model {
	all := make(map[string]Model)
	for k, v := range antigravityModels {
		all[k] = v
	}
	for k, v := range geminiModels {
		all[k] = v
	}
	return all
}()

// DefaultModel returns the default model ID for a given provider.
func DefaultModel(provider config.Provider) string {
	switch provider {
	case config.ProviderAntigravity:
		return "gemini-3-flash"
	case config.ProviderAPIKey:
		return "gemini-2.5-flash"
	default:
		return "gemini-3-flash"
	}
}

// GetModelsForProvider returns the model registry for a provider.
func GetModelsForProvider(provider config.Provider) map[string]Model {
	switch provider {
	case config.ProviderAntigravity:
		return antigravityModels
	case config.ProviderAPIKey:
		return geminiModels
	default:
		return Registry
	}
}

// GetModel looks up a model by its CLI-friendly key.
func GetModel(id string) Model {
	if m, ok := Registry[id]; ok {
		return m
	}
	// If not found in registry, pass through as-is (user might specify a raw model ID)
	return Model{ID: id, Name: id}
}
