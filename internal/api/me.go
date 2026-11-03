package api

import (
	"errors"
	"net/http"
	"time"

	"github.com/google/uuid"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/store"
)

// canonicalID returns v as a canonical UUID string (lower case, with
// hyphens), or false if v is not a UUID. uuid.Parse also takes upper
// case, braces, a urn:uuid: prefix and no hyphens; every ID that enters
// the API goes through here, because IDs are also compared as strings (a
// secret is sealed with its org and repo IDs) and Postgres rejects the
// urn form.
func canonicalID(v string) (string, bool) {
	u, err := uuid.Parse(v)
	if err != nil {
		return "", false
	}
	return u.String(), true
}

// pathID returns the path wildcard name, which must be a UUID (else 400),
// in canonical form.
func pathID(r *http.Request, name string) (string, error) {
	id, ok := canonicalID(r.PathValue(name))
	if !ok {
		return "", errBadRequest("%s is not a valid ID.", name)
	}
	return id, nil
}

// orgAccess checks that the signed-in user is a member of the path's
// org_id with at least role min. Non-members (and unknown orgs) get 403
// not_a_member; members below min get 403 forbidden_role.
func (s *Server) orgAccess(r *http.Request, min store.Role) (orgID string, role store.Role, err error) {
	orgID, err = pathID(r, "org_id")
	if err != nil {
		return "", "", err
	}
	role, err = s.store.GetOrgRole(r.Context(), orgID, currentUser(r).ID)
	if errors.Is(err, store.ErrNotFound) {
		return "", "", errNotAMember
	}
	if err != nil {
		return "", "", err
	}
	if !role.AtLeast(min) {
		return "", "", errForbiddenRole
	}
	return orgID, role, nil
}

// memberOf reports whether the signed-in user belongs to orgID.
func (s *Server) memberOf(r *http.Request, orgID string) (bool, error) {
	return s.store.HasOrgRole(r.Context(), orgID, currentUser(r).ID, store.RoleMember)
}

// ts formats a timestamp as RFC 3339 UTC, as every response does.
type ts time.Time

func (t ts) MarshalJSON() ([]byte, error) {
	return []byte(`"` + time.Time(t).UTC().Format(time.RFC3339) + `"`), nil
}

// UnmarshalJSON reads an RFC 3339 string (for tests and clients in Go).
func (t *ts) UnmarshalJSON(b []byte) error {
	var v time.Time
	if err := v.UnmarshalJSON(b); err != nil {
		return err
	}
	*t = ts(v)
	return nil
}

// tsPtr maps nil to JSON null.
func tsPtr(t *time.Time) *ts {
	if t == nil {
		return nil
	}
	v := ts(*t)
	return &v
}

type meUser struct {
	ID        string `json:"id"`
	Login     string `json:"login"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
}

type meOrg struct {
	ID   string     `json:"id"`
	Name string     `json:"name"`
	Role store.Role `json:"role"`
}

type meResponse struct {
	User meUser  `json:"user"`
	Orgs []meOrg `json:"orgs"`
}

// handleMe is GET /api/me.
func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) error {
	u := currentUser(r)
	ms, err := s.store.ListOrgsForUser(r.Context(), u.ID)
	if err != nil {
		return err
	}
	resp := meResponse{
		User: meUser{ID: u.ID, Login: u.GitHubLogin, Name: u.Name, AvatarURL: u.AvatarURL},
		Orgs: make([]meOrg, 0, len(ms)),
	}
	for _, m := range ms {
		resp.Orgs = append(resp.Orgs, meOrg{ID: m.Org.ID, Name: m.Org.Name, Role: m.Role})
	}
	writeJSON(w, http.StatusOK, resp)
	return nil
}

// knownProviders are the LLM providers in display order.
var knownProviders = []string{config.LLMProviderAnthropic, config.LLMProviderGemini}

type providerInfo struct {
	ID string `json:"id"`
	// DefaultModel is "" when the server has no default for the provider
	// (gemini without DEFAULT_GEMINI_MODEL); sessions must then name one.
	DefaultModel string `json:"default_model"`
}

type providersResponse struct {
	DefaultProvider string         `json:"default_provider"`
	Providers       []providerInfo `json:"providers"`
}

// handleProviders is GET /api/providers: the providers with keys.
func (s *Server) handleProviders(w http.ResponseWriter, _ *http.Request) error {
	resp := providersResponse{DefaultProvider: s.cfg.DefaultLLMProvider, Providers: []providerInfo{}}
	for _, p := range knownProviders {
		if s.cfg.LLMProviderConfigured(p) {
			resp.Providers = append(resp.Providers, providerInfo{ID: p, DefaultModel: s.cfg.DefaultModelFor(p)})
		}
	}
	// Point the default at a usable provider when the server default has
	// no key.
	if !s.cfg.LLMProviderConfigured(resp.DefaultProvider) && len(resp.Providers) > 0 {
		resp.DefaultProvider = resp.Providers[0].ID
	}
	writeJSON(w, http.StatusOK, resp)
	return nil
}
