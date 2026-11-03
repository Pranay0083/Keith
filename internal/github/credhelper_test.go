package github

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// installHelper writes a helper reading tokenPath to a temp dir.
func installHelper(t *testing.T, tokenPath string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("needs a POSIX shell")
	}
	path := filepath.Join(t.TempDir(), "git-credential-keith")
	if err := os.WriteFile(path, []byte(credentialHelperScript(tokenPath, "github.com")), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func runHelper(t *testing.T, helper, action, input string) string {
	t.Helper()
	cmd := exec.Command(helper, action)
	cmd.Stdin = strings.NewReader(input)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("helper %s: %v", action, err)
	}
	return string(out)
}

func TestCredentialHelperScript(t *testing.T) {
	s := CredentialHelperScript()
	if !strings.HasPrefix(s, "#!/bin/sh\n") || !strings.Contains(s, "token_file='"+GitTokenPath+"'") ||
		!strings.Contains(s, "allowed_host='github.com'") {
		t.Errorf("script does not read %s for github.com:\n%s", GitTokenPath, s)
	}
}

func TestCredentialHelperAnswersFromFile(t *testing.T) {
	tokenPath := filepath.Join(t.TempDir(), "it's a dir", "git-token")
	if err := os.MkdirAll(filepath.Dir(tokenPath), 0o700); err != nil {
		t.Fatal(err)
	}
	helper := installHelper(t, tokenPath)
	github := "protocol=https\nhost=github.com\npath=acme/api.git\n\n"

	// No token file yet: no answer, no failure.
	if out := runHelper(t, helper, "get", github); out != "" {
		t.Errorf("without a token file, output = %q", out)
	}

	write := func(s string) {
		if err := os.WriteFile(tokenPath, []byte(s), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("ghs_first\n")
	if out, want := runHelper(t, helper, "get", github), "username=x-access-token\npassword=ghs_first\n"; out != want {
		t.Errorf("get = %q, want %q", out, want)
	}

	// Rotation: the next call reads the rewritten file. No trailing
	// newline is fine.
	write("ghs_second")
	if out := runHelper(t, helper, "get", github); !strings.Contains(out, "password=ghs_second\n") {
		t.Errorf("after rotation, get = %q", out)
	}

	for name, input := range map[string]string{
		"other host":     "protocol=https\nhost=evil.example\n\n",
		"lookalike host": "protocol=https\nhost=github.com.evil.example\n\n",
		"host with port": "protocol=https\nhost=github.com:8443\n\n",
		"plain http":     "protocol=http\nhost=github.com\n\n",
		"no protocol":    "host=github.com\n\n",
	} {
		if out := runHelper(t, helper, "get", input); out != "" {
			t.Errorf("%s: output = %q, want nothing", name, out)
		}
	}
	for _, action := range []string{"store", "erase"} {
		if out := runHelper(t, helper, action, github+"username=x\npassword=y\n"); out != "" {
			t.Errorf("%s: output = %q", action, out)
		}
	}

	write("ghs_x\nusername=attacker\n")
	if out := runHelper(t, helper, "get", github); out != "username=x-access-token\npassword=ghs_x\n" {
		t.Errorf("only the first line is used; got %q", out)
	}
	write("ghs bad token\n")
	if out := runHelper(t, helper, "get", github); out != "" {
		t.Errorf("malformed token: output = %q", out)
	}
}

func TestCredentialHelperWithGit(t *testing.T) {
	git, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git not installed")
	}
	tokenPath := filepath.Join(t.TempDir(), "git-token")
	if err := os.WriteFile(tokenPath, []byte("ghs_fromgit\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	helper := installHelper(t, tokenPath)

	cmd := exec.Command(git, "-c", "credential.helper=", "-c", "credential.https://github.com.helper="+helper, "credential", "fill")
	cmd.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "HOME="+t.TempDir(), "GIT_TERMINAL_PROMPT=0")
	cmd.Stdin = strings.NewReader("url=https://github.com/acme/api.git\n\n")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git credential fill: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "username=x-access-token\n") || !strings.Contains(string(out), "password=ghs_fromgit\n") {
		t.Errorf("git credential fill = %q", out)
	}
}
