package store

import (
	"bytes"
	"strings"
	"testing"
)

func TestValidIDForms(t *testing.T) {
	const id = "3c4b2a10-8d1e-4f5a-9b6c-7d8e9f0a1b2c"
	for _, v := range []string{id, strings.ToUpper(id)} {
		if !validID(v) {
			t.Errorf("validID(%q) = false", v)
		}
	}
	// Forms uuid.Validate accepts but Postgres does not all take.
	for _, v := range []string{"urn:uuid:" + id, "{" + id + "}", strings.ReplaceAll(id, "-", ""), "", "nope"} {
		if validID(v) {
			t.Errorf("validID(%q) = true", v)
		}
	}
}

func TestSecretAADCanonicalizesIDs(t *testing.T) {
	const org, repo = "3c4b2a10-8d1e-4f5a-9b6c-7d8e9f0a1b2c", "5a2c0f00-1111-4222-8333-444455556666"
	want := []byte("keith/secret/v1\x00org=" + org + "\x00repo=" + repo + "\x00name=TOKEN")
	if got := SecretAAD(org, repo, "TOKEN"); !bytes.Equal(got, want) {
		t.Fatalf("canonical AAD changed: %q", got)
	}
	for _, ids := range [][2]string{
		{strings.ToUpper(org), strings.ToUpper(repo)},
		{"urn:uuid:" + org, "{" + repo + "}"},
	} {
		if got := SecretAAD(ids[0], ids[1], "TOKEN"); !bytes.Equal(got, want) {
			t.Errorf("SecretAAD(%q, %q) = %q, want the canonical AAD", ids[0], ids[1], got)
		}
	}
	if got := SecretAAD(org, "", "TOKEN"); !bytes.Equal(got, []byte("keith/secret/v1\x00org="+org+"\x00repo=\x00name=TOKEN")) {
		t.Errorf("org secret AAD %q", got)
	}
	// The name is case-sensitive, as environment variables are.
	if bytes.Equal(SecretAAD(org, repo, "token"), want) {
		t.Error("name folded")
	}
}
