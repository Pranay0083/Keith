package api

import (
	"errors"
	"net/http"
	"net/url"
	"strconv"

	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/hooks"
	"github.com/Pranay0083/keith/internal/store"
)

type installationJSON struct {
	ID             string                   `json:"id"`
	InstallationID int64                    `json:"installation_id"`
	AccountLogin   string                   `json:"account_login"`
	Status         store.InstallationStatus `json:"status"`
	CreatedAt      ts                       `json:"created_at"`
}

// handleListInstallations is GET /api/orgs/{org_id}/github/installations.
func (s *Server) handleListInstallations(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleMember)
	if err != nil {
		return err
	}
	insts, err := s.store.ListInstallations(r.Context(), orgID)
	if err != nil {
		return err
	}
	out := make([]installationJSON, 0, len(insts))
	for _, in := range insts {
		out = append(out, installationJSON{in.ID, in.InstallationID, in.AccountLogin, in.Status, ts(in.CreatedAt)})
	}
	writeJSON(w, http.StatusOK, out)
	return nil
}

// handleInstallURL is GET /api/orgs/{org_id}/github/install-url (owner or
// admin).
func (s *Server) handleInstallURL(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleAdmin)
	if err != nil {
		return err
	}
	slug, err := s.appSlug(r.Context())
	if err != nil {
		return err
	}
	state, err := s.signer.signSetupState(orgID, currentUser(r).ID, s.now().Add(setupStateTTL))
	if err != nil {
		return err
	}
	u, err := s.gh.InstallURL(slug, state)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, map[string]string{"url": u})
	return nil
}

// Setup failure codes (docs/api.md, GET /github/setup).
const (
	setupErrState       = "setup_state"
	setupErrNotVerified = "installation_not_verified"
	setupErrOtherOrg    = "installation_other_org"
	setupErrFailed      = "setup_failed"
)

// handleSetup is GET /github/setup, the App's Setup URL. It links an
// installation to the org bound in state, but only after proving the
// signed-in user owns the account the installation is on: anyone can
// call this URL with someone else's installation_id, and anyone with
// read access to one of its repos can see an installation.
func (s *Server) handleSetup(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	fail := func(code string, err error) {
		if err != nil {
			s.log(r).Warn("github setup failed", "code", code, "err", err)
		} else {
			s.log(r).Warn("github setup refused", "code", code)
		}
		http.Redirect(w, r, s.webURLFor("/connect?error="+url.QueryEscape(code)), http.StatusFound)
	}

	user, _, err := s.sessionUser(r)
	if errors.Is(err, errUnauthenticated) {
		http.Redirect(w, r, s.webURLFor("/login?next="+url.QueryEscape("/connect")), http.StatusFound)
		return
	}
	if err != nil {
		fail(setupErrFailed, err)
		return
	}

	// 1. The state is ours, fresh, for this user, who still administers
	// its org.
	q := r.URL.Query()
	st, err := s.signer.verifySetupState(q.Get("state"), s.now())
	if err != nil || st.UserID != user.ID {
		fail(setupErrState, nil)
		return
	}
	ok, err := s.store.HasOrgRole(ctx, st.OrgID, user.ID, store.RoleAdmin)
	if err != nil {
		fail(setupErrFailed, err)
		return
	}
	if !ok {
		fail(setupErrState, nil)
		return
	}

	// 2. A request for a GitHub org owner's approval: nothing to link yet.
	if q.Get("setup_action") == "request" {
		http.Redirect(w, r, s.webURLFor("/connect?pending=1"), http.StatusFound)
		return
	}

	// 3. Prove the user owns the installation: exchange the code, check
	// the token is this user's, look for the installation in
	// GET /user/installations, then require the user to be its account
	// (a user installation) or an active admin of its organization.
	instID, err := strconv.ParseInt(q.Get("installation_id"), 10, 64)
	if err != nil || instID <= 0 {
		fail(setupErrFailed, nil)
		return
	}
	code := q.Get("code")
	if code == "" {
		fail(setupErrNotVerified, nil)
		return
	}
	tok, err := s.setupGH.Exchange(ctx, code)
	if err != nil {
		fail(setupErrFailed, err)
		return
	}
	ghUser, err := s.setupGH.GetUser(ctx, tok.AccessToken)
	if err != nil {
		fail(setupErrFailed, err)
		return
	}
	if ghUser.ID != user.GitHubUserID {
		fail(setupErrNotVerified, nil)
		return
	}
	canAccess, err := s.setupGH.UserCanAccessInstallation(ctx, tok.AccessToken, instID)
	if err != nil {
		fail(setupErrFailed, err)
		return
	}
	if !canAccess {
		fail(setupErrNotVerified, nil)
		return
	}

	inst, err := s.gh.GetInstallation(ctx, instID)
	if err != nil {
		fail(setupErrFailed, err)
		return
	}
	switch inst.AccountType {
	case github.AccountTypeUser:
		if inst.AccountID != user.GitHubUserID {
			fail(setupErrNotVerified, nil)
			return
		}
	case github.AccountTypeOrganization:
		admin, err := s.setupGH.UserIsOrgAdmin(ctx, tok.AccessToken, inst.AccountLogin)
		if err != nil {
			fail(setupErrFailed, err)
			return
		}
		if !admin {
			fail(setupErrNotVerified, nil)
			return
		}
	default:
		fail(setupErrNotVerified, nil)
		return
	}

	// 4. Link the installation to the org.
	status := store.InstallationActive
	if inst.Suspended {
		status = store.InstallationSuspended
	}
	if _, err := s.store.UpsertInstallation(ctx, store.UpsertInstallationParams{
		OrgID: st.OrgID, InstallationID: instID, AccountLogin: inst.AccountLogin, Status: status,
	}); errors.Is(err, store.ErrConflict) {
		fail(setupErrOtherOrg, nil)
		return
	} else if err != nil {
		fail(setupErrFailed, err)
		return
	}

	// 5. Sync its repos.
	if _, err := hooks.SyncInstallationRepos(ctx, s.store, s.gh, instID); err != nil {
		fail(setupErrFailed, err)
		return
	}
	s.log(r).Info("github installation linked", "org_id", st.OrgID, "installation_id", instID)
	http.Redirect(w, r, s.webURLFor("/repos?installed=1"), http.StatusFound)
}
