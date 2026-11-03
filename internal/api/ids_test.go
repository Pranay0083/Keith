package api

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// IDs are UUIDs, which have several spellings: upper case, braces, a
// urn:uuid: prefix, no hyphens. The API canonicalizes every ID where it
// enters, so a non-canonical one finds the same row, never seals a
// secret under an AAD its row cannot reproduce, and is never a 500.
func TestNonCanonicalIDsIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	spellings := func(id string) []string {
		return []string{strings.ToUpper(id), "urn:uuid:" + id, "{" + id + "}", strings.ReplaceAll(id, "-", "")}
	}

	// Secrets sealed through upper-case IDs open with the row's AAD, as
	// the worker opens them.
	upperOrg := strings.ToUpper(w.org.ID)
	decode[secretJSON](t, w.do("PUT", "/api/orgs/"+upperOrg+"/secrets/NPM_TOKEN", map[string]any{"value": "s3cret"}, w.as(w.admin.ID)), 200)
	rs := decode[secretJSON](t, w.do("PUT", "/api/orgs/"+upperOrg+"/secrets/DB_URL",
		map[string]any{"value": "pg://x", "repo_id": strings.ToUpper(w.repo.ID)}, w.as(w.admin.ID)), 200)
	if rs.RepoID == nil || *rs.RepoID != w.repo.ID {
		t.Fatalf("repo secret %+v", rs)
	}
	cts, err := w.st.LoadSecretCiphertexts(ctx, w.org.ID, w.repo.ID)
	if err != nil || len(cts) != 2 {
		t.Fatalf("ciphertexts %+v %v", cts, err)
	}
	for _, ct := range cts {
		if _, err := w.srv.keyring.Open(ct.Ciphertext, ct.AAD()); err != nil {
			t.Errorf("worker cannot open %s (org %s, repo %q): %v", ct.Name, ct.OrgID, ct.RepoID, err)
		}
	}
	if r := w.do("DELETE", "/api/orgs/"+w.org.ID+"/secrets/DB_URL?repo_id="+escapeBraces(strings.ToUpper(w.repo.ID)), nil, w.as(w.admin.ID)); r.Code != 204 {
		t.Fatalf("delete with an upper-case repo_id: %d %s", r.Code, r.Body)
	}

	// Path IDs in every spelling find the row; unknown ones are 404.
	x := w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{strings.ToUpper(w.repo.ID)}})
	if len(x.Repos) != 1 || x.Repos[0].RepoID != w.repo.ID {
		t.Fatalf("session repos %+v", x.Repos)
	}
	for _, id := range spellings(x.ID) {
		got := decode[sessionJSON](t, w.do("GET", "/api/sessions/"+escapeBraces(id), nil, w.as(w.member.ID)), 200)
		if got.ID != x.ID || got.OrgID != w.org.ID {
			t.Errorf("GET session %s: %+v", id, got)
		}
	}
	for _, id := range spellings(w.org.ID) {
		if r := w.do("GET", "/api/orgs/"+escapeBraces(id)+"/repos", nil, w.as(w.member.ID)); r.Code != 200 {
			t.Errorf("GET org %s repos: %d %s", id, r.Code, r.Body)
		}
	}
	for _, id := range spellings(w.repo.ID) {
		if r := w.do("GET", "/api/repos/"+escapeBraces(id)+"/profile", nil, w.as(w.member.ID)); r.Code != 200 {
			t.Errorf("GET repo %s profile: %d %s", id, r.Code, r.Body)
		}
	}
	const unknown = "00000000-0000-4000-8000-0000000000ff"
	wantError(t, w.do("GET", "/api/sessions/urn:uuid:"+unknown, nil, w.as(w.member.ID)), http.StatusNotFound, codeNotFound)
	wantError(t, w.do("GET", "/api/sessions/urn:uuid:nope", nil, w.as(w.member.ID)), http.StatusBadRequest, codeBadRequest)

	// The same repo in two spellings is one repo, so not distinct.
	e := wantError(t, w.do("POST", "/api/orgs/"+w.org.ID+"/sessions",
		map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID, strings.ToUpper(w.repo.ID)}}, w.as(w.member.ID)), 422, codeValidationFailed)
	if e.Fields["repo_ids"] != "must be distinct" {
		t.Errorf("fields %+v", e.Fields)
	}
}

// escapeBraces percent-encodes the braces of a UUID spelling for a URL.
func escapeBraces(s string) string {
	return strings.NewReplacer("{", "%7B", "}", "%7D").Replace(s)
}
