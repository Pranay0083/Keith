package store

import (
	"bytes"
	"errors"
	"testing"
	"time"
)

func TestRoleAtLeast(t *testing.T) {
	cases := []struct {
		r, min Role
		want   bool
	}{
		{RoleOwner, RoleOwner, true},
		{RoleOwner, RoleAdmin, true},
		{RoleOwner, RoleMember, true},
		{RoleAdmin, RoleOwner, false},
		{RoleAdmin, RoleAdmin, true},
		{RoleAdmin, RoleMember, true},
		{RoleMember, RoleAdmin, false},
		{RoleMember, RoleMember, true},
		{Role("guest"), RoleMember, false},
		{Role(""), Role(""), false},
	}
	for _, c := range cases {
		if got := c.r.AtLeast(c.min); got != c.want {
			t.Errorf("%q.AtLeast(%q) = %v, want %v", c.r, c.min, got, c.want)
		}
	}
}

func TestCursorRoundTrip(t *testing.T) {
	ts := time.Date(2026, 11, 3, 10, 15, 3, 123456000, time.UTC)
	id := "6f1c2d4e-8a9b-4c3d-9e0f-112233445566"
	gotTS, gotID, err := decodeCursor(encodeCursor(ts, id))
	if err != nil {
		t.Fatal(err)
	}
	if !gotTS.Equal(ts) || gotID != id {
		t.Errorf("round trip = %v %q, want %v %q", gotTS, gotID, ts, id)
	}
	for _, bad := range []string{"!!", "bm9waXBl", encodeCursor(ts, "not-a-uuid")} {
		if _, _, err := decodeCursor(bad); !errors.Is(err, ErrInvalidCursor) {
			t.Errorf("decodeCursor(%q) err = %v, want ErrInvalidCursor", bad, err)
		}
	}
}

func TestSecretAADDistinguishesRows(t *testing.T) {
	a := SecretAAD("o1", "", "TOKEN")
	for _, b := range [][]byte{
		SecretAAD("o2", "", "TOKEN"),
		SecretAAD("o1", "r1", "TOKEN"),
		SecretAAD("o1", "", "TOKEN2"),
		// A name cannot smuggle a repo, since fields are NUL-separated and
		// names are env var names.
		SecretAAD("o1", "", "repo=r1"),
	} {
		if bytes.Equal(a, b) {
			t.Errorf("SecretAAD collision: %q", b)
		}
	}
}

func TestValidSecretName(t *testing.T) {
	for name, want := range map[string]bool{
		"TOKEN": true, "_x": true, "a1_B2": true,
		"": false, "1A": false, "A-B": false, "A B": false, "A=B": false,
	} {
		if got := ValidSecretName(name); got != want {
			t.Errorf("ValidSecretName(%q) = %v, want %v", name, got, want)
		}
	}
}
