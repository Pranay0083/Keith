package api

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/store"
)

func TestReposAndProfilesIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()

	// List: all repos including inaccessible ones, by name, profile null.
	list := decode[[]repoJSON](t, w.do("GET", "/api/orgs/"+w.org.ID+"/repos", nil, w.as(w.member.ID)), 200)
	var names []string
	for _, r := range list {
		names = append(names, r.FullName)
		if r.Profile != nil {
			t.Errorf("%s has a profile", r.FullName)
		}
	}
	if strings.Join(names, ",") != "acme/api,acme/gone,acme/web" || list[1].Accessible || !list[0].Accessible || !list[0].Private {
		t.Fatalf("repos %+v", list)
	}
	wantError(t, w.do("GET", "/api/orgs/"+w.org.ID+"/repos", nil, w.as(w.outsider.ID)), 403, codeNotAMember)

	// GET profile before one exists: empty strings, updated_at null.
	path := "/api/repos/" + w.repo.ID + "/profile"
	resp := w.do("GET", path, nil, w.as(w.member.ID))
	if resp.Code != 200 || strings.TrimSpace(resp.Body.String()) !=
		`{"repo_id":"`+w.repo.ID+`","install_cmd":"","build_cmd":"","test_cmd":"","run_cmd":"","knowledge":"","updated_at":null}` {
		t.Fatalf("empty profile %d %s", resp.Code, resp.Body)
	}
	// Not a member, unknown repo: 404. Malformed: 400.
	wantError(t, w.do("GET", path, nil, w.as(w.outsider.ID)), 404, codeNotFound)
	wantError(t, w.do("GET", "/api/repos/00000000-0000-4000-8000-000000000000/profile", nil, w.as(w.owner.ID)), 404, codeNotFound)
	wantError(t, w.do("GET", "/api/repos/xyz/profile", nil, w.as(w.owner.ID)), 400, codeBadRequest)

	// PUT: all five required, length limits.
	full := map[string]string{"install_cmd": "npm ci", "build_cmd": "npm run build", "test_cmd": "npm test", "run_cmd": "npm run dev", "knowledge": "pnpm"}
	missing := map[string]string{"install_cmd": "npm ci"}
	e := wantError(t, w.do("PUT", path, missing, w.as(w.member.ID)), 422, codeValidationFailed)
	if len(e.Fields) != 4 || e.Fields["knowledge"] != "is required" {
		t.Errorf("fields %+v", e.Fields)
	}
	long := map[string]string{}
	for k, v := range full {
		long[k] = v
	}
	long["test_cmd"] = strings.Repeat("é", maxCmdLen+1)
	long["knowledge"] = strings.Repeat("k", maxKnowledgeLen)
	e = wantError(t, w.do("PUT", path, long, w.as(w.member.ID)), 422, codeValidationFailed)
	if len(e.Fields) != 1 || e.Fields["test_cmd"] == "" {
		t.Errorf("fields %+v", e.Fields)
	}
	wantError(t, w.do("PUT", path, `{"install_cmd":1}`, w.as(w.member.ID)), 422, codeValidationFailed)
	wantError(t, w.do("PUT", path, full, w.as(w.outsider.ID)), 404, codeNotFound)

	got := decode[repoProfileJSON](t, w.do("PUT", path, full, w.as(w.member.ID)), 200)
	if got.RepoID != w.repo.ID || got.InstallCmd != "npm ci" || got.Knowledge != "pnpm" || got.UpdatedAt == nil {
		t.Fatalf("saved %+v", got)
	}
	// A snapshot survives the same install_cmd and is cleared by a new one.
	snap := store.Snapshot{ID: "snap-1", Provider: "docker", LockHash: "h"}
	if err := w.st.SetRepoSnapshot(ctx, w.repo.ID, snap); err != nil {
		t.Fatal(err)
	}
	full["test_cmd"] = "npm run test:ci"
	resp = w.do("PUT", path, full, w.as(w.member.ID))
	if resp.Code != 200 || strings.Contains(resp.Body.String(), "snapshot") {
		t.Fatalf("put %d %s", resp.Code, resp.Body)
	}
	if p, _ := w.st.GetRepoProfile(ctx, w.repo.ID); p.Snapshot != snap {
		t.Fatalf("snapshot cleared on unchanged install_cmd: %+v", p.Snapshot)
	}
	full["install_cmd"] = "pnpm i"
	w.do("PUT", path, full, w.as(w.member.ID))
	if p, _ := w.st.GetRepoProfile(ctx, w.repo.ID); p.Snapshot != (store.Snapshot{}) {
		t.Fatalf("snapshot kept after install_cmd change: %+v", p.Snapshot)
	}

	// The list now carries the profile.
	list = decode[[]repoJSON](t, w.do("GET", "/api/orgs/"+w.org.ID+"/repos", nil, w.as(w.member.ID)), 200)
	for _, r := range list {
		if (r.ID == w.repo.ID) != (r.Profile != nil) {
			t.Errorf("%s profile %+v", r.FullName, r.Profile)
		}
	}
}

func TestSecretsIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	var logs bytes.Buffer
	w.srv.logger = slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	base := "/api/orgs/" + w.org.ID + "/secrets"
	const value = "sk_test_super_secret_value"

	// Roles: members may list, only owner/admin may write.
	wantError(t, w.do("PUT", base+"/API_KEY", map[string]any{"value": value}, w.as(w.member.ID)), 403, codeForbiddenRole)
	wantError(t, w.do("PUT", base+"/API_KEY", map[string]any{"value": value}, w.as(w.outsider.ID)), 403, codeNotAMember)
	wantError(t, w.do("DELETE", base+"/API_KEY", nil, w.as(w.member.ID)), 403, codeForbiddenRole)
	wantError(t, w.do("GET", base, nil, w.as(w.outsider.ID)), 403, codeNotAMember)

	// Validation.
	for name, tc := range map[string]struct {
		path  string
		body  any
		field string
	}{
		"bad name":     {base + "/1BAD", map[string]any{"value": value}, "name"},
		"long name":    {base + "/" + strings.Repeat("A", 129), map[string]any{"value": value}, "name"},
		"no value":     {base + "/OK", map[string]any{}, "value"},
		"empty value":  {base + "/OK", map[string]any{"value": ""}, "value"},
		"big value":    {base + "/OK", map[string]any{"value": strings.Repeat("x", maxSecretValueLen+1)}, "value"},
		"foreign repo": {base + "/OK", map[string]any{"value": value, "repo_id": w.otherOrgsRepo.ID}, "repo_id"},
		"bad repo id":  {base + "/OK", map[string]any{"value": value, "repo_id": "nope"}, "repo_id"},
		"repo id type": {base + "/OK", map[string]any{"value": value, "repo_id": 5}, "repo_id"},
	} {
		e := wantError(t, w.do("PUT", tc.path, tc.body, w.as(w.admin.ID)), 422, codeValidationFailed)
		if e.Fields[tc.field] == "" {
			t.Errorf("%s: fields %+v", name, e.Fields)
		}
	}

	// Create an org secret, a repo secret of the same name, and another.
	put := func(name string, body map[string]any) secretJSON {
		t.Helper()
		return decode[secretJSON](t, w.do("PUT", base+"/"+name, body, w.as(w.admin.ID)), 200)
	}
	org := put("DATABASE_URL", map[string]any{"value": value, "repo_id": nil})
	if org.Name != "DATABASE_URL" || org.RepoID != nil || org.RepoFullName != nil {
		t.Fatalf("org secret %+v", org)
	}
	rs := put("DATABASE_URL", map[string]any{"value": "repo-" + value, "repo_id": w.repo.ID})
	if rs.RepoID == nil || *rs.RepoID != w.repo.ID || rs.RepoFullName == nil || *rs.RepoFullName != "acme/web" {
		t.Fatalf("repo secret %+v", rs)
	}
	put("API_KEY", map[string]any{"value": value})
	// Replacing keeps one row.
	put("API_KEY", map[string]any{"value": value + "2"})

	resp := w.do("GET", base, nil, w.as(w.member.ID))
	list := decode[[]secretJSON](t, resp, 200)
	if len(list) != 3 || list[0].Name != "API_KEY" || list[1].Name != "DATABASE_URL" || list[1].RepoID != nil ||
		list[2].RepoID == nil {
		t.Fatalf("list %s", resp.Body)
	}
	if strings.Contains(resp.Body.String(), value) {
		t.Fatal("secret value in list response")
	}

	// The stored value opens with the keyring under the row's AAD.
	cts, err := w.st.LoadSecretCiphertexts(ctx, w.org.ID, w.repo.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cts {
		pt, err := w.srv.keyring.Open(c.Ciphertext, c.AAD())
		if err != nil {
			t.Fatalf("open %s: %v", c.Name, err)
		}
		if c.Name == "DATABASE_URL" && c.RepoID == w.repo.ID && string(pt) != "repo-"+value {
			t.Errorf("repo secret = %q", pt)
		}
	}

	// Delete: org secret by default, repo secret with repo_id; 404 after.
	if r := w.do("DELETE", base+"/DATABASE_URL?repo_id="+w.repo.ID, nil, w.as(w.owner.ID)); r.Code != 204 {
		t.Fatalf("delete repo secret %d %s", r.Code, r.Body)
	}
	wantError(t, w.do("DELETE", base+"/DATABASE_URL?repo_id="+w.repo.ID, nil, w.as(w.owner.ID)), 404, codeNotFound)
	if r := w.do("DELETE", base+"/DATABASE_URL", nil, w.as(w.owner.ID)); r.Code != 204 {
		t.Fatalf("delete org secret %d", r.Code)
	}
	wantError(t, w.do("DELETE", base+"/DATABASE_URL", nil, w.as(w.owner.ID)), 404, codeNotFound)
	wantError(t, w.do("DELETE", base+"/API_KEY?repo_id=bad", nil, w.as(w.owner.ID)), 400, codeBadRequest)

	if strings.Contains(logs.String(), value) {
		t.Fatal("secret value logged")
	}
}
