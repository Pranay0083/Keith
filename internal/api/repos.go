package api

import (
	"errors"
	"net/http"
	"unicode/utf8"

	"github.com/Pranay0083/keith/internal/store"
)

// Profile field limits, in characters.
const (
	maxCmdLen       = 2000
	maxKnowledgeLen = 20000
)

type profileJSON struct {
	InstallCmd string `json:"install_cmd"`
	BuildCmd   string `json:"build_cmd"`
	TestCmd    string `json:"test_cmd"`
	RunCmd     string `json:"run_cmd"`
	Knowledge  string `json:"knowledge"`
	UpdatedAt  *ts    `json:"updated_at"`
}

func toProfileJSON(p store.RepoProfile) profileJSON {
	t := ts(p.UpdatedAt)
	return profileJSON{p.InstallCmd, p.BuildCmd, p.TestCmd, p.RunCmd, p.Knowledge, &t}
}

type repoJSON struct {
	ID            string       `json:"id"`
	FullName      string       `json:"full_name"`
	DefaultBranch string       `json:"default_branch"`
	Private       bool         `json:"private"`
	Accessible    bool         `json:"accessible"`
	Profile       *profileJSON `json:"profile"`
}

// handleListRepos is GET /api/orgs/{org_id}/repos: every repo, including
// inaccessible ones, by full name.
func (s *Server) handleListRepos(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleMember)
	if err != nil {
		return err
	}
	repos, err := s.store.ListRepos(r.Context(), orgID, true)
	if err != nil {
		return err
	}
	profiles, err := s.store.ListRepoProfiles(r.Context(), orgID)
	if err != nil {
		return err
	}
	out := make([]repoJSON, 0, len(repos))
	for _, rp := range repos {
		j := repoJSON{ID: rp.ID, FullName: rp.FullName, DefaultBranch: rp.DefaultBranch, Private: rp.Private, Accessible: rp.Accessible}
		if p, ok := profiles[rp.ID]; ok {
			pj := toProfileJSON(p)
			j.Profile = &pj
		}
		out = append(out, j)
	}
	writeJSON(w, http.StatusOK, out)
	return nil
}

// repoAccess loads the path's repo_id for a member of its org. Unknown
// repos and repos in other orgs are both 404.
func (s *Server) repoAccess(r *http.Request) (store.Repo, error) {
	id, err := pathID(r, "repo_id")
	if err != nil {
		return store.Repo{}, err
	}
	repo, err := s.store.GetRepo(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		return store.Repo{}, errNotFound
	}
	if err != nil {
		return store.Repo{}, err
	}
	ok, err := s.memberOf(r, repo.OrgID)
	if err != nil {
		return store.Repo{}, err
	}
	if !ok {
		return store.Repo{}, errNotFound
	}
	return repo, nil
}

type repoProfileJSON struct {
	RepoID string `json:"repo_id"`
	profileJSON
}

func (s *Server) writeProfile(w http.ResponseWriter, repoID string, p *store.RepoProfile) {
	out := repoProfileJSON{RepoID: repoID}
	if p != nil {
		out.profileJSON = toProfileJSON(*p)
	}
	writeJSON(w, http.StatusOK, out)
}

// handleGetProfile is GET /api/repos/{repo_id}/profile.
func (s *Server) handleGetProfile(w http.ResponseWriter, r *http.Request) error {
	repo, err := s.repoAccess(r)
	if err != nil {
		return err
	}
	p, err := s.store.GetRepoProfile(r.Context(), repo.ID)
	if errors.Is(err, store.ErrNotFound) {
		s.writeProfile(w, repo.ID, nil)
		return nil
	}
	if err != nil {
		return err
	}
	s.writeProfile(w, repo.ID, &p)
	return nil
}

type putProfileRequest struct {
	InstallCmd *string `json:"install_cmd"`
	BuildCmd   *string `json:"build_cmd"`
	TestCmd    *string `json:"test_cmd"`
	RunCmd     *string `json:"run_cmd"`
	Knowledge  *string `json:"knowledge"`
}

// handlePutProfile is PUT /api/repos/{repo_id}/profile: replace the whole
// profile. A changed install_cmd clears the snapshot (in the store).
func (s *Server) handlePutProfile(w http.ResponseWriter, r *http.Request) error {
	repo, err := s.repoAccess(r)
	if err != nil {
		return err
	}
	var req putProfileRequest
	if err := decodeJSON(r, &req); err != nil {
		return err
	}
	fields := map[string]string{}
	check := func(name string, v *string, max int) string {
		switch {
		case v == nil:
			fields[name] = "is required"
		case utf8.RuneCountInString(*v) > max:
			fields[name] = "must be at most " + itoa(max) + " characters"
		default:
			return *v
		}
		return ""
	}
	p := store.UpsertRepoProfileParams{
		RepoID:     repo.ID,
		InstallCmd: check("install_cmd", req.InstallCmd, maxCmdLen),
		BuildCmd:   check("build_cmd", req.BuildCmd, maxCmdLen),
		TestCmd:    check("test_cmd", req.TestCmd, maxCmdLen),
		RunCmd:     check("run_cmd", req.RunCmd, maxCmdLen),
		Knowledge:  check("knowledge", req.Knowledge, maxKnowledgeLen),
	}
	if err := validationFields(fields); err != nil {
		return err
	}
	saved, err := s.store.UpsertRepoProfile(r.Context(), p)
	if errors.Is(err, store.ErrNotFound) {
		return errNotFound
	}
	if err != nil {
		return err
	}
	s.writeProfile(w, repo.ID, &saved)
	return nil
}
