package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/Pranay0083/keith/internal/store"
)

// Secret limits.
const (
	maxSecretNameLen  = 128
	maxSecretValueLen = 65536
)

// Secret values are write-only. No handler returns one, and none is ever
// logged: the request body is decoded into a local and sealed at once.

type secretJSON struct {
	Name         string  `json:"name"`
	RepoID       *string `json:"repo_id"`
	RepoFullName *string `json:"repo_full_name"`
	CreatedAt    ts      `json:"created_at"`
	UpdatedAt    ts      `json:"updated_at"`
}

func strOrNull(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func toSecretJSON(m store.SecretMeta, repoFullName string) secretJSON {
	return secretJSON{m.Name, strOrNull(m.RepoID), strOrNull(repoFullName), ts(m.CreatedAt), ts(m.UpdatedAt)}
}

// handleListSecrets is GET /api/orgs/{org_id}/secrets.
func (s *Server) handleListSecrets(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleMember)
	if err != nil {
		return err
	}
	list, err := s.store.ListOrgSecrets(r.Context(), orgID)
	if err != nil {
		return err
	}
	out := make([]secretJSON, 0, len(list))
	for _, x := range list {
		out = append(out, toSecretJSON(x.SecretMeta, x.RepoFullName))
	}
	writeJSON(w, http.StatusOK, out)
	return nil
}

type putSecretRequest struct {
	Value *string `json:"value"`
	// RepoID is absent, null or a repo ID. It is decoded as raw JSON so a
	// wrong type is a field error, not a 400.
	RepoID json.RawMessage `json:"repo_id"`
}

func validSecretName(name string) bool {
	return len(name) <= maxSecretNameLen && store.ValidSecretName(name)
}

// handlePutSecret is PUT /api/orgs/{org_id}/secrets/{name} (owner or
// admin).
func (s *Server) handlePutSecret(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleAdmin)
	if err != nil {
		return err
	}
	var req putSecretRequest
	if err := decodeJSON(r, &req); err != nil {
		return err
	}
	name := r.PathValue("name")
	fields := map[string]string{}
	if !validSecretName(name) {
		fields["name"] = "must match ^[A-Za-z_][A-Za-z0-9_]*$ and be at most 128 characters"
	}
	switch {
	case req.Value == nil || *req.Value == "":
		fields["value"] = "is required"
	case len(*req.Value) > maxSecretValueLen:
		fields["value"] = "must be at most 65536 bytes"
	}
	repoID := ""
	repoFullName := ""
	if len(req.RepoID) > 0 && string(req.RepoID) != "null" {
		var ok bool
		if json.Unmarshal(req.RepoID, &repoID) == nil {
			repoID, ok = canonicalID(repoID)
		}
		if !ok {
			repoID = ""
			fields["repo_id"] = "must be a repo ID or null"
		} else {
			repo, err := s.store.GetRepo(r.Context(), repoID)
			switch {
			case errors.Is(err, store.ErrNotFound) || (err == nil && repo.OrgID != orgID):
				fields["repo_id"] = "is not a repo of this organization"
			case err != nil:
				return err
			default:
				repoFullName = repo.FullName
			}
		}
	}
	if err := validationFields(fields); err != nil {
		return err
	}

	ct, err := s.keyring.Seal([]byte(*req.Value), store.SecretAAD(orgID, repoID, name))
	if err != nil {
		return err
	}
	meta, err := s.store.PutSecret(r.Context(), store.PutSecretParams{
		OrgID: orgID, RepoID: repoID, Name: name, Ciphertext: ct, CreatedBy: currentUser(r).ID,
	})
	if errors.Is(err, store.ErrNotFound) { // the repo moved out of the org meanwhile
		return validation("repo_id", "is not a repo of this organization")
	}
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, toSecretJSON(meta, repoFullName))
	return nil
}

// handleDeleteSecret is DELETE /api/orgs/{org_id}/secrets/{name}
// [?repo_id=] (owner or admin).
func (s *Server) handleDeleteSecret(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleAdmin)
	if err != nil {
		return err
	}
	repoID := r.URL.Query().Get("repo_id")
	if repoID != "" {
		var ok bool
		if repoID, ok = canonicalID(repoID); !ok {
			return errBadRequest("repo_id is not a valid ID.")
		}
	}
	name := r.PathValue("name")
	if !validSecretName(name) {
		return errNotFound
	}
	err = s.store.DeleteSecret(r.Context(), orgID, repoID, name)
	if errors.Is(err, store.ErrNotFound) {
		return errNotFound
	}
	if err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}
