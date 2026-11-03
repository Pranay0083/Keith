package api

import (
	"context"
	"testing"

	"github.com/Pranay0083/keith/internal/store"
)

func TestReservedSecretName(t *testing.T) {
	for _, name := range []string{
		"DOCKER_HOST", "docker_host", "Docker_Config", "DOCKER_TLS_VERIFY", "LD_PRELOAD", "ld_library_path",
		"GIT_SSH_COMMAND", "git_dir", "GIT_CONFIG_COUNT", "KEITH_ANYTHING", "PATH", "path", "HOME",
		"HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
	} {
		if !reservedSecretName(name) {
			t.Errorf("%s is not reserved", name)
		}
	}
	for _, name := range []string{
		"GITHUB_TOKEN", "GITLAB_TOKEN", "NPM_TOKEN", "DATABASE_URL", "PATHS", "MY_PATH", "HOMEBREW_NO_ANALYTICS",
		"DOCKERHUB_TOKEN", "LDAP_URL", "KEITHLY", "PROXY", "SOCKS_PROXY_URL",
	} {
		if reservedSecretName(name) {
			t.Errorf("%s is reserved", name)
		}
	}
}

// Secrets become environment variables of the sandbox, and of the
// docker provider's own docker CLI, so names that would redirect or
// reconfigure them are refused.
func TestSecretReservedNamesIT(t *testing.T) {
	w := newWorld(t)
	base := "/api/orgs/" + w.org.ID + "/secrets/"
	for _, tc := range []struct{ name, repoID string }{
		{"DOCKER_HOST", ""}, {"http_proxy", ""}, {"LD_PRELOAD", ""}, {"PATH", ""},
		{"GIT_ASKPASS", w.repo.ID}, {"KEITH_TOKEN", w.repo.ID},
	} {
		body := map[string]any{"value": "tcp://evil.test:2375"}
		if tc.repoID != "" {
			body["repo_id"] = tc.repoID
		}
		e := wantError(t, w.do("PUT", base+tc.name, body, w.as(w.admin.ID)), 422, codeValidationFailed)
		if e.Fields["name"] != reservedSecretMsg {
			t.Errorf("%s: fields %+v", tc.name, e.Fields)
		}
	}
	if list, _ := w.st.ListOrgSecrets(context.Background(), w.org.ID); len(list) != 0 {
		t.Fatalf("reserved secrets stored: %+v", list)
	}
	decode[secretJSON](t, w.do("PUT", base+"GITHUB_TOKEN", map[string]any{"value": "ghp_x"}, w.as(w.admin.ID)), 200)

	// A reserved name saved before the rule can still be deleted.
	if _, err := w.st.PutSecret(context.Background(), store.PutSecretParams{
		OrgID: w.org.ID, Name: "DOCKER_HOST", Ciphertext: []byte("sealed"),
	}); err != nil {
		t.Fatal(err)
	}
	if r := w.do("DELETE", base+"DOCKER_HOST", nil, w.as(w.admin.ID)); r.Code != 204 {
		t.Fatalf("delete reserved secret: %d %s", r.Code, r.Body)
	}
}
