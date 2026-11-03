package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/hooks"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
)

// Session limits (docs/api.md).
const (
	maxPromptLen   = 32000
	maxReposPerRun = 10
	maxModelLen    = 100
	maxTextLen     = 16000
	defaultPage    = 25
	maxPage        = 100
)

// webTrigger is sessions.trigger for a session started from the web app.
type webTrigger struct {
	Kind     string `json:"kind"`
	IssueURL string `json:"issue_url,omitempty"`
}

type sessionRepoJSON struct {
	RepoID   string  `json:"repo_id"`
	FullName string  `json:"full_name"`
	Branch   string  `json:"branch"`
	PRNumber *int    `json:"pr_number"`
	PRURL    *string `json:"pr_url"`
}

type createdByJSON struct {
	ID        string `json:"id"`
	Login     string `json:"login"`
	AvatarURL string `json:"avatar_url"`
}

type sessionJSON struct {
	ID              string               `json:"id"`
	OrgID           string               `json:"org_id"`
	Status          events.SessionStatus `json:"status"`
	Prompt          string               `json:"prompt"`
	IssueURL        *string              `json:"issue_url"`
	Provider        string               `json:"provider"`
	Model           string               `json:"model"`
	CreatedBy       *createdByJSON       `json:"created_by"`
	CancelRequested bool                 `json:"cancel_requested"`
	CreatedAt       ts                   `json:"created_at"`
	FinishedAt      *ts                  `json:"finished_at"`
	Repos           []sessionRepoJSON    `json:"repos"`
}

// issueURLOf reads the issue URL from a web trigger.
func issueURLOf(trigger json.RawMessage) *string {
	if len(trigger) == 0 {
		return nil
	}
	var t webTrigger
	if json.Unmarshal(trigger, &t) != nil || t.Kind != "web" || t.IssueURL == "" {
		return nil
	}
	return &t.IssueURL
}

func toSessionJSON(x store.Session, users map[string]store.User) sessionJSON {
	out := sessionJSON{
		ID: x.ID, OrgID: x.OrgID, Status: x.Status, Prompt: x.Prompt, IssueURL: issueURLOf(x.Trigger),
		Provider: x.Provider, Model: x.Model, CancelRequested: x.CancelRequestedAt != nil,
		CreatedAt: ts(x.CreatedAt), FinishedAt: tsPtr(x.FinishedAt), Repos: make([]sessionRepoJSON, 0, len(x.Repos)),
	}
	if u, ok := users[x.CreatedBy]; ok {
		out.CreatedBy = &createdByJSON{u.ID, u.GitHubLogin, u.AvatarURL}
	}
	for _, r := range x.Repos {
		j := sessionRepoJSON{RepoID: r.RepoID, FullName: r.FullName, Branch: r.Branch, PRURL: strOrNull(r.PRURL)}
		if r.PRNumber != 0 {
			n := r.PRNumber
			j.PRNumber = &n
		}
		out.Repos = append(out.Repos, j)
	}
	return out
}

// sessionsJSON converts sessions, looking up their authors in one query.
func (s *Server) sessionsJSON(r *http.Request, xs []store.Session) ([]sessionJSON, error) {
	var ids []string
	for _, x := range xs {
		if x.CreatedBy != "" {
			ids = append(ids, x.CreatedBy)
		}
	}
	users, err := s.store.GetUsers(r.Context(), ids)
	if err != nil {
		return nil, err
	}
	out := make([]sessionJSON, 0, len(xs))
	for _, x := range xs {
		out = append(out, toSessionJSON(x, users))
	}
	return out, nil
}

// sessionAccess loads the path's session for a member of its org.
// Unknown sessions and sessions of other orgs are both 404.
func (s *Server) sessionAccess(r *http.Request) (store.Session, error) {
	id, err := pathID(r, "id")
	if err != nil {
		return store.Session{}, err
	}
	x, err := s.store.GetSession(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		return store.Session{}, errNotFound
	}
	if err != nil {
		return store.Session{}, err
	}
	ok, err := s.memberOf(r, x.OrgID)
	if err != nil {
		return store.Session{}, err
	}
	if !ok {
		return store.Session{}, errNotFound
	}
	return x, nil
}

type createSessionRequest struct {
	Prompt   *string  `json:"prompt"`
	RepoIDs  []string `json:"repo_ids"`
	IssueURL *string  `json:"issue_url"`
	Provider *string  `json:"provider"`
	Model    *string  `json:"model"`
}

// handleCreateSession is POST /api/orgs/{org_id}/sessions.
func (s *Server) handleCreateSession(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	orgID, _, err := s.orgAccess(r, store.RoleMember)
	if err != nil {
		return err
	}
	var req createSessionRequest
	if err := decodeJSON(r, &req); err != nil {
		return err
	}
	fields := map[string]string{}

	// The org's accessible repos, to check repo_ids and the issue's repo.
	repos, err := s.store.ListRepos(ctx, orgID, false)
	if err != nil {
		return err
	}
	accessible := make(map[string]store.Repo, len(repos))
	byName := make(map[string]store.Repo, len(repos))
	for _, rp := range repos {
		accessible[rp.ID] = rp
		byName[strings.ToLower(rp.FullName)] = rp
	}

	// repo_ids: distinct, accessible repos of this org, in order.
	repoIDs := make([]string, 0, len(req.RepoIDs))
	for _, id := range req.RepoIDs {
		switch {
		case uuid.Validate(id) != nil:
			fields["repo_ids"] = "must be repo IDs"
		case containsString(repoIDs, id):
			fields["repo_ids"] = "must be distinct"
		default:
			if _, ok := accessible[id]; !ok {
				fields["repo_ids"] = "must be accessible repos of this organization"
			}
		}
		repoIDs = append(repoIDs, id)
	}

	// issue_url: a GitHub issue in an accessible repo of the org, which
	// is tagged automatically if it is not already.
	issueURL := ""
	if req.IssueURL != nil {
		issueURL = strings.TrimSpace(*req.IssueURL)
	}
	if issueURL != "" {
		name, _, err := github.ParseIssueURL(issueURL)
		if err != nil {
			fields["issue_url"] = "must be https://github.com/{owner}/{repo}/issues/{number}"
		} else if rp, ok := byName[strings.ToLower(name)]; !ok {
			fields["issue_url"] = "issue_repo_not_tagged"
		} else if !containsString(repoIDs, rp.ID) && fields["repo_ids"] == "" {
			repoIDs = append(repoIDs, rp.ID)
		}
	}
	if fields["repo_ids"] == "" {
		switch {
		case len(repoIDs) == 0:
			fields["repo_ids"] = "at least one repo is required"
		case len(repoIDs) > maxReposPerRun:
			fields["repo_ids"] = "at most 10 repos"
		}
	}

	// prompt: may be "" only with an issue.
	prompt := ""
	if req.Prompt != nil {
		prompt = *req.Prompt
	}
	switch {
	case utf8.RuneCountInString(prompt) > maxPromptLen:
		fields["prompt"] = "must be at most 32000 characters"
	case strings.TrimSpace(prompt) == "" && issueURL == "":
		fields["prompt"] = "is required without an issue_url"
	}

	// provider: the request's, else the org's default, else the server's.
	provider := ""
	if req.Provider != nil {
		provider = *req.Provider
	}
	if provider != "" && !isKnownProvider(provider) {
		fields["provider"] = "must be anthropic or gemini"
	}
	model := ""
	if req.Model != nil {
		model = strings.TrimSpace(*req.Model)
	}
	if utf8.RuneCountInString(model) > maxModelLen {
		fields["model"] = "must be at most 100 characters"
	}
	if err := validationFields(fields); err != nil {
		return err
	}
	if provider == "" {
		org, err := s.store.GetOrg(ctx, orgID)
		if err != nil {
			return err
		}
		provider = org.DefaultLLMProvider
	}
	if provider == "" {
		provider = s.cfg.DefaultLLMProvider
	}
	if !s.cfg.LLMProviderConfigured(provider) {
		return &apiError{
			Status: http.StatusUnprocessableEntity, Code: codeProviderUnavailable,
			Message: "The " + provider + " provider is not configured on this server.",
			Fields:  map[string]string{"provider": "is not configured on this server"},
		}
	}
	// model: the request's, else the provider's default.
	if model == "" {
		model = s.cfg.DefaultModelFor(provider)
	}
	if model == "" {
		return validation("model", "is required: the server has no default model for "+provider)
	}

	trigger, err := json.Marshal(webTrigger{Kind: "web", IssueURL: issueURL})
	if err != nil {
		return err
	}

	// Check the cap and create the session (with its pinned provider and
	// model) in one transaction; the org row lock makes the cap exact.
	var sess store.Session
	err = s.store.WithTx(ctx, func(tx *store.Store) error {
		if err := tx.LockOrg(ctx, orgID); err != nil {
			return err
		}
		if err := s.checkCap(ctx, tx, orgID); err != nil {
			return err
		}
		var err error
		sess, err = tx.CreateSession(ctx, store.CreateSessionParams{
			OrgID: orgID, CreatedBy: currentUser(r).ID, Prompt: prompt,
			Provider: provider, Model: model, RepoIDs: repoIDs, Trigger: trigger,
		})
		return err
	})
	if errors.Is(err, store.ErrNotFound) { // a repo became inaccessible meanwhile
		return validation("repo_ids", "must be accessible repos of this organization")
	}
	if err != nil {
		return err
	}

	// Enqueue only after commit, so the worker always finds the row.
	if err := s.queue.EnqueueRunSession(ctx, queue.RunSessionPayload{SessionID: sess.ID}); err != nil {
		s.log(r).Error("enqueue run_session failed", "session_id", sess.ID, "err", err)
		if serr := s.store.SetSessionStatus(ctx, sess.ID, events.StatusFailed, "could not queue the session"); serr != nil {
			s.log(r).Error("mark unqueued session failed", "session_id", sess.ID, "err", serr)
		}
		return errInternal
	}
	writeJSON(w, http.StatusCreated, map[string]string{"id": sess.ID})
	return nil
}

// checkCap answers 429 rate_limited when the org already has
// MaxActiveSessionsPerOrg active sessions, for a request that would add
// one: a new session, or a resume of a finished or waiting one. Call it
// inside WithTx after tx.LockOrg, which makes the count exact against
// other API requests.
//
// A resumed session keeps its finished or waiting status until a worker
// picks it up, so resumes queued close together can each see room. The
// worker must re-check the cap before it runs a resumed session.
//
// Lock order: the org row, then a session row (GetSessionForUpdate).
func (s *Server) checkCap(ctx context.Context, tx *store.Store, orgID string) error {
	n, err := tx.CountActiveSessions(ctx, orgID)
	if err != nil {
		return err
	}
	if n >= s.cfg.MaxActiveSessionsPerOrg {
		return &apiError{
			Status: http.StatusTooManyRequests, Code: codeRateLimited, RetryAfter: rateLimitRetryAfter,
			Message: fmt.Sprintf("This organization already has %d active sessions. Wait for one to finish.", n),
		}
	}
	return nil
}

func isKnownProvider(p string) bool {
	return p == config.LLMProviderAnthropic || p == config.LLMProviderGemini
}

func containsString(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

type sessionPageJSON struct {
	Sessions   []sessionJSON `json:"sessions"`
	NextCursor *string       `json:"next_cursor"`
}

// handleListSessions is GET /api/orgs/{org_id}/sessions.
func (s *Server) handleListSessions(w http.ResponseWriter, r *http.Request) error {
	orgID, _, err := s.orgAccess(r, store.RoleMember)
	if err != nil {
		return err
	}
	q := r.URL.Query()
	limit := defaultPage
	if v := q.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > maxPage {
			return errBadRequest("limit must be an integer from 1 to 100.")
		}
		limit = n
	}
	page, err := s.store.ListSessions(r.Context(), store.ListSessionsParams{OrgID: orgID, Limit: limit, Cursor: q.Get("cursor")})
	if errors.Is(err, store.ErrInvalidCursor) {
		return errBadRequest("cursor is not valid.")
	}
	if err != nil {
		return err
	}
	list, err := s.sessionsJSON(r, page.Sessions)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, sessionPageJSON{Sessions: list, NextCursor: strOrNull(page.NextCursor)})
	return nil
}

// handleGetSession is GET /api/sessions/{id}.
func (s *Server) handleGetSession(w http.ResponseWriter, r *http.Request) error {
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	list, err := s.sessionsJSON(r, []store.Session{x})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, list[0])
	return nil
}

// inputText checks a message or answer text: 1 to 16,000 characters
// after trimming. It returns the trimmed text.
func inputText(text *string) (string, error) {
	if text == nil {
		return "", validation("text", "is required")
	}
	t := strings.TrimSpace(*text)
	switch n := utf8.RuneCountInString(t); {
	case n == 0:
		return "", validation("text", "is required")
	case n > maxTextLen:
		return "", validation("text", "must be at most 16000 characters")
	}
	return t, nil
}

type messageRequest struct {
	Text *string `json:"text"`
}

type messageResponse struct {
	MessageID string `json:"message_id"`
}

// handlePostMessage is POST /api/sessions/{id}/messages. It stores the
// message for the worker; for a finished session it also enqueues
// resume_session. It never writes an event.
func (s *Server) handlePostMessage(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	var req messageRequest
	if err := decodeJSON(r, &req); err != nil {
		return err
	}
	text, err := inputText(req.Text)
	if err != nil {
		return err
	}
	var in store.Input
	var status events.SessionStatus
	err = s.store.WithTx(ctx, func(tx *store.Store) error {
		// The org lock comes first (see checkCap). The session row lock
		// orders this insert against the worker's status writes (see the
		// worker contract in package queue).
		if err := tx.LockOrg(ctx, x.OrgID); err != nil {
			return err
		}
		cur, err := tx.GetSessionForUpdate(ctx, x.ID)
		if err != nil {
			return err
		}
		status = cur.Status
		if status == events.StatusWaitingForUser {
			return errorf(http.StatusConflict, codeAnswerRequired, "The session is waiting for an answer. Use /answers.")
		}
		if status.Terminal() { // this message resumes the session
			if err := checkReposUsable(ctx, tx, x.ID); err != nil {
				return err
			}
			if err := s.checkCap(ctx, tx, x.OrgID); err != nil {
				return err
			}
		}
		in, err = tx.CreateInput(ctx, store.CreateInputParams{
			SessionID: x.ID, Kind: store.InputMessage, UserID: currentUser(r).ID, Text: text,
		})
		return err
	})
	if errors.Is(err, store.ErrNotFound) {
		return errNotFound
	}
	if err != nil {
		return err
	}
	if status.Terminal() {
		if err := s.queue.EnqueueResumeSession(ctx, queue.ResumeSessionPayload{
			SessionID: x.ID, Reason: queue.ResumeMessage, InputID: in.ID,
		}); err != nil {
			s.undoInput(r, in, "")
			return fmt.Errorf("enqueue resume_session: %w", err)
		}
	}
	writeJSON(w, http.StatusAccepted, messageResponse{in.ID})
	return nil
}

// checkReposUsable refuses, with 409 repo_access_lost, to resume a
// session when its org can no longer use one of its repos: the repo was
// removed from the installation, the App was uninstalled, or the repo
// moved to another org's installation. The worker would otherwise
// provision it with no access, or with the other org's.
func checkReposUsable(ctx context.Context, tx *store.Store, sessionID string) error {
	repos, err := tx.ListSessionRepos(ctx, sessionID)
	if err != nil {
		return err
	}
	for _, r := range repos {
		if r.Inaccessible {
			return errorf(http.StatusConflict, codeRepoAccessLost,
				"%s is no longer available to this organization, so this session cannot continue. Start a new session.", r.FullName)
		}
	}
	return nil
}

// undoInput deletes an input whose resume_session could not be enqueued
// and, for an answer, reopens its question, so the user can retry.
func (s *Server) undoInput(r *http.Request, in store.Input, questionID string) {
	ctx := r.Context()
	if err := s.store.DeleteInput(ctx, in.SessionID, in.ID); err != nil {
		s.log(r).Error("undo input", "session_id", in.SessionID, "input_id", in.ID, "err", err)
	}
	if questionID != "" {
		if _, err := s.store.ReopenQuestion(ctx, in.SessionID, questionID); err != nil {
			s.log(r).Error("reopen question", "session_id", in.SessionID, "err", err)
		}
	}
}

type answerRequest struct {
	QuestionID *string `json:"question_id"`
	Text       *string `json:"text"`
}

// handlePostAnswer is POST /api/sessions/{id}/answers: answer the open
// question.asked, then enqueue resume_session.
func (s *Server) handlePostAnswer(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	var req answerRequest
	if err := decodeJSON(r, &req); err != nil {
		return err
	}
	if req.QuestionID == nil || *req.QuestionID == "" {
		return validation("question_id", "is required")
	}
	qid := *req.QuestionID
	text, err := inputText(req.Text)
	if err != nil {
		return err
	}
	var in store.Input
	err = s.store.WithTx(ctx, func(tx *store.Store) error {
		if err := tx.LockOrg(ctx, x.OrgID); err != nil { // before the session row (see checkCap)
			return err
		}
		cur, err := tx.GetSessionForUpdate(ctx, x.ID)
		if err != nil {
			return err
		}
		if cur.Status != events.StatusWaitingForUser {
			return errorf(http.StatusConflict, codeNotWaiting, "The session is not waiting for an answer.")
		}
		if cur.OpenQuestionID == "" || cur.OpenQuestionID != qid {
			return errorf(http.StatusConflict, codeQuestionMismatch, "That question is not the open question.")
		}
		if err := checkReposUsable(ctx, tx, x.ID); err != nil {
			return err
		}
		// waiting_for_user is not an active status: the answer makes the
		// session active again.
		if err := s.checkCap(ctx, tx, x.OrgID); err != nil {
			return err
		}
		in, err = tx.CreateInput(ctx, store.CreateInputParams{
			SessionID: x.ID, Kind: store.InputAnswer, QuestionID: qid, UserID: currentUser(r).ID, Text: text,
		})
		if err != nil {
			return err
		}
		// Answered: a second answer to it is a question_mismatch.
		return tx.ClearOpenQuestion(ctx, x.ID)
	})
	if errors.Is(err, store.ErrNotFound) {
		return errNotFound
	}
	if err != nil {
		return err
	}
	if err := s.queue.EnqueueResumeSession(ctx, queue.ResumeSessionPayload{
		SessionID: x.ID, Reason: queue.ResumeAnswer, InputID: in.ID,
	}); err != nil {
		s.undoInput(r, in, qid)
		return fmt.Errorf("enqueue resume_session: %w", err)
	}
	writeJSON(w, http.StatusAccepted, messageResponse{in.ID})
	return nil
}

// handleStop is POST /api/sessions/{id}/stop.
func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) error {
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	err = hooks.StopSession(r.Context(), s.hooksDeps(), x.ID)
	if errors.Is(err, hooks.ErrSessionFinished) {
		return errorf(http.StatusConflict, codeNotRunning, "The session has already finished.")
	}
	if errors.Is(err, store.ErrNotFound) {
		return errNotFound
	}
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusAccepted, struct{}{})
	return nil
}

func (s *Server) hooksDeps() hooks.Deps {
	return hooks.Deps{Store: s.store, GitHub: s.gh, Queue: s.queue, Control: s.control, Logger: s.logger}
}

type diffRepoJSON struct {
	FullName  string `json:"full_name"`
	Diff      string `json:"diff"`
	Truncated bool   `json:"truncated"`
	UpdatedAt *ts    `json:"updated_at"`
}

// handleDiff is GET /api/sessions/{id}/diff.
func (s *Server) handleDiff(w http.ResponseWriter, r *http.Request) error {
	x, err := s.sessionAccess(r)
	if err != nil {
		return err
	}
	diffs, err := s.store.ListSessionDiffs(r.Context(), x.ID)
	if err != nil {
		return err
	}
	out := struct {
		Repos []diffRepoJSON `json:"repos"`
	}{Repos: make([]diffRepoJSON, 0, len(diffs))}
	for _, d := range diffs {
		out.Repos = append(out.Repos, diffRepoJSON{d.FullName, d.Diff, d.Truncated, tsPtr(d.UpdatedAt)})
	}
	writeJSON(w, http.StatusOK, out)
	return nil
}
