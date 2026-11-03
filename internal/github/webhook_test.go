package github

import (
	"errors"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// GitHub's published test vector, from "Validating webhook deliveries".
const (
	vectorSecret    = "It's a Secret to Everybody"
	vectorPayload   = "Hello, World!"
	vectorSignature = "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17"
)

func TestVerifySignatureKnownVector(t *testing.T) {
	if err := VerifySignature([]byte(vectorSecret), []byte(vectorPayload), vectorSignature); err != nil {
		t.Fatalf("known vector: %v", err)
	}
}

func TestVerifySignatureRejects(t *testing.T) {
	secret, body := []byte(vectorSecret), []byte(vectorPayload)
	hexSig := strings.TrimPrefix(vectorSignature, "sha256=")
	cases := []struct {
		name   string
		secret []byte
		body   []byte
		header string
		want   error
	}{
		{"tampered body", secret, []byte("Hello, World?"), vectorSignature, ErrBadSignature},
		{"wrong secret", []byte("another secret"), body, vectorSignature, ErrBadSignature},
		{"sha1 prefix", secret, body, "sha1=" + hexSig, ErrBadSignature},
		{"no prefix", secret, body, hexSig, ErrBadSignature},
		{"uppercase prefix", secret, body, "SHA256=" + hexSig, ErrBadSignature},
		{"truncated digest", secret, body, vectorSignature[:len(vectorSignature)-2], ErrBadSignature},
		{"not hex", secret, body, "sha256=" + strings.Repeat("zz", 32), ErrBadSignature},
		{"flipped digit", secret, body, vectorSignature[:len(vectorSignature)-1] + "8", ErrBadSignature},
		{"missing header", secret, body, "", ErrMissingSignature},
	}
	for _, c := range cases {
		if err := VerifySignature(c.secret, c.body, c.header); !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
	}
	if err := VerifySignature(nil, body, vectorSignature); err == nil {
		t.Error("empty secret: no error")
	}
}

func TestDeliveryIDAndEventType(t *testing.T) {
	r := httptest.NewRequest("POST", "/hooks/github", nil)
	r.Header.Set("X-GitHub-Delivery", "72d3162e-cc78-11e3-81ab-4c9367dc0958")
	r.Header.Set("X-GitHub-Event", "installation")
	if got := DeliveryID(r); got != "72d3162e-cc78-11e3-81ab-4c9367dc0958" {
		t.Errorf("DeliveryID = %q", got)
	}
	if got := EventType(r); got != EventInstallation {
		t.Errorf("EventType = %q", got)
	}
}

func TestParseInstallationEvent(t *testing.T) {
	body := `{
	  "action": "created",
	  "installation": {
	    "id": 1234, "app_id": 4242, "repository_selection": "selected",
	    "account": {"id": 9, "login": "acme", "type": "Organization"}
	  },
	  "repositories": [
	    {"id": 1, "name": "api", "full_name": "acme/api", "private": true},
	    {"id": 2, "name": "web", "full_name": "acme/web", "private": false}
	  ],
	  "sender": {"login": "octo"}
	}`
	got, err := ParseInstallationEvent([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	want := InstallationEvent{
		Action:              "created",
		InstallationID:      1234,
		Account:             Account{ID: 9, Login: "acme", Type: "Organization"},
		RepositorySelection: "selected",
		Repos:               []RepoRef{{ID: 1, FullName: "acme/api", Private: true}, {ID: 2, FullName: "acme/web"}},
		Sender:              "octo",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("event = %+v\nwant    %+v", got, want)
	}

	deleted, err := ParseInstallationEvent([]byte(`{"action":"deleted","installation":{"id":1234,"account":{"login":"acme"}}}`))
	if err != nil || deleted.Action != "deleted" || deleted.InstallationID != 1234 || deleted.Repos != nil {
		t.Errorf("deleted event = %+v, %v", deleted, err)
	}

	for _, bad := range []string{`not json`, `{"action":"created"}`, `{"installation":{"id":1}}`} {
		if _, err := ParseInstallationEvent([]byte(bad)); err == nil {
			t.Errorf("ParseInstallationEvent(%s) succeeded", bad)
		}
	}
}

func TestParseInstallationRepositoriesEvent(t *testing.T) {
	body := `{
	  "action": "removed",
	  "installation": {"id": 77, "account": {"id": 3, "login": "octo", "type": "User"}},
	  "repository_selection": "selected",
	  "repositories_added": [],
	  "repositories_removed": [{"id": 5, "full_name": "octo/old", "private": true}],
	  "sender": {"login": "octo"}
	}`
	got, err := ParseInstallationRepositoriesEvent([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	want := InstallationRepositoriesEvent{
		Action:              "removed",
		InstallationID:      77,
		Account:             Account{ID: 3, Login: "octo", Type: "User"},
		RepositorySelection: "selected",
		Removed:             []RepoRef{{ID: 5, FullName: "octo/old", Private: true}},
		Sender:              "octo",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("event = %+v\nwant    %+v", got, want)
	}

	added, err := ParseInstallationRepositoriesEvent([]byte(`{"action":"added","installation":{"id":77,"repository_selection":"all"},
	  "repositories_added":[{"id":6,"full_name":"octo/new"}]}`))
	if err != nil || len(added.Added) != 1 || added.Added[0].FullName != "octo/new" || added.RepositorySelection != "all" {
		t.Errorf("added event = %+v, %v", added, err)
	}

	if _, err := ParseInstallationRepositoriesEvent([]byte(`{"action":"added"}`)); err == nil {
		t.Error("event without installation parsed")
	}
}
