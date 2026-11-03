package config

import (
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var testPEM = string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: []byte("test-key-bytes")}))

func env(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func fullEnv() map[string]string {
	return map[string]string{
		"DATABASE_URL":             "postgres://keith:keith@localhost:5432/keith?sslmode=disable",
		"REDIS_URL":                "redis://localhost:6379/0",
		"ANTHROPIC_API_KEY":        "sk-ant-test",
		"GITHUB_APP_ID":            "12345",
		"GITHUB_APP_CLIENT_ID":     "Iv1.abc",
		"GITHUB_APP_CLIENT_SECRET": "client-secret",
		"GITHUB_APP_PRIVATE_KEY":   testPEM,
		"GITHUB_WEBHOOK_SECRET":    "whsec",
		"PUBLIC_URL":               "https://api.example.com/",
		"WEB_ORIGIN":               "https://app.example.com",
		"SESSION_SECRET":           strings.Repeat("s", 32),
		"SECRETS_KEY":              base64.StdEncoding.EncodeToString(make([]byte, 32)),
	}
}

func TestDefaults(t *testing.T) {
	c, err := LoadFrom(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	if c.DefaultLLMProvider != "anthropic" || c.DefaultModel != "claude-opus-5-5" {
		t.Errorf("llm defaults = %q %q", c.DefaultLLMProvider, c.DefaultModel)
	}
	if c.SandboxProvider != "docker" || c.ListenAddr != ":8080" {
		t.Errorf("defaults = %q %q", c.SandboxProvider, c.ListenAddr)
	}
}

func TestFullConfigValidatesForBothRoles(t *testing.T) {
	c, err := LoadFrom(env(fullEnv()))
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range []Role{RoleAPI, RoleWorker} {
		if err := c.Validate(role); err != nil {
			t.Errorf("Validate(%s) = %v", role, err)
		}
	}
	if c.GitHubAppID != 12345 || c.PublicURL != "https://api.example.com" || len(c.SecretsKey) != 32 {
		t.Errorf("parsed = %d %q %d", c.GitHubAppID, c.PublicURL, len(c.SecretsKey))
	}
	if err := c.Validate("dispatcher"); err == nil {
		t.Error("unknown role accepted")
	}
}

func TestValidateReportsMissingPerRole(t *testing.T) {
	c, err := LoadFrom(env(map[string]string{"SANDBOX_PROVIDER": "daytona"}))
	if err != nil {
		t.Fatal(err)
	}

	err = c.Validate(RoleAPI)
	for _, name := range []string{"DATABASE_URL", "REDIS_URL", "SECRETS_KEY", "GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY",
		"GITHUB_APP_CLIENT_ID", "GITHUB_APP_CLIENT_SECRET", "GITHUB_WEBHOOK_SECRET", "PUBLIC_URL", "WEB_ORIGIN", "SESSION_SECRET"} {
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("api error %v does not name %s", err, name)
		}
	}
	if strings.Contains(err.Error(), "ANTHROPIC_API_KEY") || strings.Contains(err.Error(), "DAYTONA") {
		t.Errorf("api error names worker-only vars: %v", err)
	}

	err = c.Validate(RoleWorker)
	for _, name := range []string{"DATABASE_URL", "ANTHROPIC_API_KEY", "DAYTONA_API_KEY", "GITHUB_APP_ID"} {
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("worker error %v does not name %s", err, name)
		}
	}
	if strings.Contains(err.Error(), "SESSION_SECRET") || strings.Contains(err.Error(), "WEBHOOK") {
		t.Errorf("worker error names api-only vars: %v", err)
	}
}

func TestGeminiNeedsKeyAndModel(t *testing.T) {
	e := fullEnv()
	e["DEFAULT_LLM_PROVIDER"] = "gemini"
	c, err := LoadFrom(env(e))
	if err != nil {
		t.Fatal(err)
	}
	err = c.Validate(RoleWorker)
	if err == nil || !strings.Contains(err.Error(), "GEMINI_API_KEY") || !strings.Contains(err.Error(), "DEFAULT_MODEL") {
		t.Errorf("err = %v, want GEMINI_API_KEY and DEFAULT_MODEL missing", err)
	}
	e["GEMINI_API_KEY"] = "g"
	e["DEFAULT_MODEL"] = "gemini-model"
	c, _ = LoadFrom(env(e))
	if err := c.Validate(RoleWorker); err != nil {
		t.Errorf("Validate = %v", err)
	}
}

func TestMalformedValues(t *testing.T) {
	cases := map[string]map[string]string{
		"DEFAULT_LLM_PROVIDER":        {"DEFAULT_LLM_PROVIDER": "openai"},
		"SANDBOX_PROVIDER":            {"SANDBOX_PROVIDER": "e2b"},
		"GITHUB_APP_ID":               {"GITHUB_APP_ID": "abc"},
		"SECRETS_KEY: not valid":      {"SECRETS_KEY": "!!!"},
		"SECRETS_KEY: decodes to 16":  {"SECRETS_KEY": base64.StdEncoding.EncodeToString(make([]byte, 16))},
		"PUBLIC_URL":                  {"PUBLIC_URL": "api.example.com"},
		"WEB_ORIGIN":                  {"WEB_ORIGIN": "ftp://x"},
		"not a PEM":                   {"GITHUB_APP_PRIVATE_KEY": "hello"},
		"set only one":                {"GITHUB_APP_PRIVATE_KEY": testPEM, "GITHUB_APP_PRIVATE_KEY_PATH": "/x"},
		"GITHUB_APP_PRIVATE_KEY_PATH": {"GITHUB_APP_PRIVATE_KEY_PATH": "/nonexistent/key.pem"},
	}
	for want, e := range cases {
		_, err := LoadFrom(env(e))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%v: err = %v, want it to mention %q", e, err, want)
		}
	}
}

func TestSecretsKeyErrorDoesNotEchoValue(t *testing.T) {
	v := base64.StdEncoding.EncodeToString([]byte("sixteen byte key"))
	_, err := LoadFrom(env(map[string]string{"SECRETS_KEY": v}))
	if err == nil || strings.Contains(err.Error(), v) {
		t.Errorf("err = %v", err)
	}
}

func TestPrivateKeyFromPathAndEscapedNewlines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "app.pem")
	if err := os.WriteFile(path, []byte(testPEM), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := LoadFrom(env(map[string]string{"GITHUB_APP_PRIVATE_KEY_PATH": path}))
	if err != nil || string(c.GitHubAppPrivateKey) != testPEM {
		t.Fatalf("from path: err = %v", err)
	}

	escaped := strings.ReplaceAll(strings.TrimSpace(testPEM), "\n", `\n`)
	c, err = LoadFrom(env(map[string]string{"GITHUB_APP_PRIVATE_KEY": escaped}))
	if err != nil {
		t.Fatalf("escaped: %v", err)
	}
	if block, _ := pem.Decode(c.GitHubAppPrivateKey); block == nil || string(block.Bytes) != "test-key-bytes" {
		t.Errorf("escaped key not decoded: %q", c.GitHubAppPrivateKey)
	}
}

func TestShortSessionSecret(t *testing.T) {
	e := fullEnv()
	e["SESSION_SECRET"] = "short"
	c, _ := LoadFrom(env(e))
	if err := c.Validate(RoleAPI); err == nil || !strings.Contains(err.Error(), "SESSION_SECRET") {
		t.Errorf("err = %v", err)
	}
}

func TestStringRedactsSecrets(t *testing.T) {
	e := fullEnv()
	c, _ := LoadFrom(env(e))
	for _, out := range []string{c.String(), fmt.Sprintf("%v", c), fmt.Sprintf("%+v", c), fmt.Sprintf("%#v", c)} {
		for _, k := range []string{"DATABASE_URL", "REDIS_URL", "ANTHROPIC_API_KEY", "GITHUB_APP_CLIENT_SECRET",
			"GITHUB_WEBHOOK_SECRET", "SESSION_SECRET", "SECRETS_KEY"} {
			if strings.Contains(out, e[k]) {
				t.Errorf("output leaks %s: %s", k, out)
			}
		}
		if strings.Contains(out, "BEGIN") {
			t.Errorf("output leaks private key: %s", out)
		}
	}
}
