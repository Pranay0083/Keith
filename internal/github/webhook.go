package github

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// Webhook request headers.
const (
	SignatureHeader = "X-Hub-Signature-256"
	EventHeader     = "X-GitHub-Event"
	DeliveryHeader  = "X-GitHub-Delivery"
)

// MaxWebhookBody is GitHub's cap on a webhook payload (25 MB). Read
// bodies through http.MaxBytesReader with this limit.
const MaxWebhookBody = 25 << 20

// Webhook event names (X-GitHub-Event) this package parses.
const (
	EventInstallation             = "installation"
	EventInstallationRepositories = "installation_repositories"
)

var (
	// ErrMissingSignature means the X-Hub-Signature-256 header is absent.
	ErrMissingSignature = errors.New("github: webhook signature missing")
	// ErrBadSignature means the signature is malformed or does not match.
	ErrBadSignature = errors.New("github: webhook signature invalid")
)

// VerifySignature checks header, the X-Hub-Signature-256 value
// ("sha256=" and a hex HMAC-SHA256 of body under secret), in constant
// time. Verify before parsing or storing anything from the body. An
// empty secret is an error, never a pass.
func VerifySignature(secret, body []byte, header string) error {
	if len(secret) == 0 {
		return errors.New("github: webhook secret is empty")
	}
	if header == "" {
		return ErrMissingSignature
	}
	hexSig, ok := strings.CutPrefix(header, "sha256=")
	if !ok {
		return fmt.Errorf("%w: want a sha256= prefix", ErrBadSignature)
	}
	got, err := hex.DecodeString(hexSig)
	if err != nil || len(got) != sha256.Size {
		return fmt.Errorf("%w: not a hex SHA-256 digest", ErrBadSignature)
	}
	mac := hmac.New(sha256.New, secret)
	mac.Write(body)
	if !hmac.Equal(got, mac.Sum(nil)) {
		return ErrBadSignature
	}
	return nil
}

// DeliveryID returns the delivery's unique ID (X-GitHub-Delivery), the
// dedupe key for webhook_deliveries. A redelivery keeps the same ID.
func DeliveryID(r *http.Request) string { return r.Header.Get(DeliveryHeader) }

// EventType returns the event name (X-GitHub-Event).
func EventType(r *http.Request) string { return r.Header.Get(EventHeader) }

// Account is the user or org an installation belongs to.
type Account struct {
	ID    int64
	Login string
	// Type is "Organization" or "User".
	Type string
}

// RepoRef is a repo as webhook payloads list it.
type RepoRef struct {
	ID       int64
	FullName string
	Private  bool
}

// InstallationEvent is an installation webhook.
type InstallationEvent struct {
	// Action is created, deleted, suspend, unsuspend or
	// new_permissions_accepted.
	Action              string
	InstallationID      int64
	Account             Account
	RepositorySelection string
	// Repos are the repos the installation can access; GitHub sends
	// them on created.
	Repos []RepoRef
	// Sender is the login of the user who acted.
	Sender string
}

// InstallationRepositoriesEvent is an installation_repositories webhook:
// repos were added to or removed from an installation.
type InstallationRepositoriesEvent struct {
	// Action is added or removed.
	Action              string
	InstallationID      int64
	Account             Account
	RepositorySelection string
	Added               []RepoRef
	Removed             []RepoRef
	Sender              string
}

type wireInstallation struct {
	ID      int64 `json:"id"`
	Account struct {
		ID    int64  `json:"id"`
		Login string `json:"login"`
		Type  string `json:"type"`
	} `json:"account"`
	RepositorySelection string `json:"repository_selection"`
}

type wireRepo struct {
	ID       int64  `json:"id"`
	FullName string `json:"full_name"`
	Private  bool   `json:"private"`
}

type wireSender struct {
	Login string `json:"login"`
}

func repoRefs(in []wireRepo) []RepoRef {
	if len(in) == 0 {
		return nil
	}
	out := make([]RepoRef, len(in))
	for i, r := range in {
		out[i] = RepoRef(r)
	}
	return out
}

func (w wireInstallation) account() Account {
	return Account{ID: w.Account.ID, Login: w.Account.Login, Type: w.Account.Type}
}

// ParseInstallationEvent parses an installation webhook body. Verify the
// signature first.
func ParseInstallationEvent(body []byte) (InstallationEvent, error) {
	var p struct {
		Action       string           `json:"action"`
		Installation wireInstallation `json:"installation"`
		Repositories []wireRepo       `json:"repositories"`
		Sender       wireSender       `json:"sender"`
	}
	if err := json.Unmarshal(body, &p); err != nil {
		return InstallationEvent{}, fmt.Errorf("github: parse installation event: %w", err)
	}
	if p.Action == "" || p.Installation.ID <= 0 {
		return InstallationEvent{}, errors.New("github: parse installation event: missing action or installation id")
	}
	return InstallationEvent{
		Action:              p.Action,
		InstallationID:      p.Installation.ID,
		Account:             p.Installation.account(),
		RepositorySelection: p.Installation.RepositorySelection,
		Repos:               repoRefs(p.Repositories),
		Sender:              p.Sender.Login,
	}, nil
}

// ParseInstallationRepositoriesEvent parses an installation_repositories
// webhook body. Verify the signature first.
func ParseInstallationRepositoriesEvent(body []byte) (InstallationRepositoriesEvent, error) {
	var p struct {
		Action              string           `json:"action"`
		Installation        wireInstallation `json:"installation"`
		RepositorySelection string           `json:"repository_selection"`
		Added               []wireRepo       `json:"repositories_added"`
		Removed             []wireRepo       `json:"repositories_removed"`
		Sender              wireSender       `json:"sender"`
	}
	if err := json.Unmarshal(body, &p); err != nil {
		return InstallationRepositoriesEvent{}, fmt.Errorf("github: parse installation_repositories event: %w", err)
	}
	if p.Action == "" || p.Installation.ID <= 0 {
		return InstallationRepositoriesEvent{}, errors.New("github: parse installation_repositories event: missing action or installation id")
	}
	sel := p.RepositorySelection
	if sel == "" {
		sel = p.Installation.RepositorySelection
	}
	return InstallationRepositoriesEvent{
		Action:              p.Action,
		InstallationID:      p.Installation.ID,
		Account:             p.Installation.account(),
		RepositorySelection: sel,
		Added:               repoRefs(p.Added),
		Removed:             repoRefs(p.Removed),
		Sender:              p.Sender.Login,
	}, nil
}
