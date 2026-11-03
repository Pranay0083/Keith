package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/control"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/store"
)

// create posts a session as the member and returns the stored session.
func (w *world) create(t *testing.T, body map[string]any) store.Session {
	t.Helper()
	resp := decode[map[string]string](t, w.do("POST", "/api/orgs/"+w.org.ID+"/sessions", body, w.as(w.member.ID)), http.StatusCreated)
	x, err := w.st.GetSession(context.Background(), resp["id"])
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func (w *world) setStatus(t *testing.T, id string, st events.SessionStatus) {
	t.Helper()
	if err := w.st.SetSessionStatus(context.Background(), id, st, ""); err != nil {
		t.Fatal(err)
	}
}

func TestCreateSessionIT(t *testing.T) {
	w := newWorld(t)
	path := "/api/orgs/" + w.org.ID + "/sessions"

	x := w.create(t, map[string]any{"prompt": "Fix the flaky test", "repo_ids": []string{w.repo2.ID, w.repo.ID}})
	if x.Status != events.StatusQueued || x.Prompt != "Fix the flaky test" || x.Provider != "anthropic" ||
		x.Model != "claude-opus-5-5" || x.CreatedBy != w.member.ID {
		t.Fatalf("session %+v", x)
	}
	if len(x.Repos) != 2 || x.Repos[0].RepoID != w.repo2.ID || x.Repos[1].RepoID != w.repo.ID || x.Repos[0].Branch != "agent/"+x.ID {
		t.Fatalf("repos %+v", x.Repos)
	}
	if runs := w.q.Runs(); len(runs) != 1 || runs[0].SessionID != x.ID {
		t.Fatalf("runs %+v", runs)
	}
	// The API wrote no event.
	if n := len(w.evs.All(x.ID)); n != 0 {
		t.Fatalf("API wrote %d events", n)
	}

	// The session object.
	got := decode[sessionJSON](t, w.do("GET", "/api/sessions/"+x.ID, nil, w.as(w.owner.ID)), 200)
	if got.ID != x.ID || got.OrgID != w.org.ID || got.IssueURL != nil || got.CreatedBy == nil || got.CreatedBy.Login != "member" ||
		got.CancelRequested || got.FinishedAt != nil || len(got.Repos) != 2 || got.Repos[0].FullName != "acme/api" ||
		got.Repos[0].PRNumber != nil || got.Repos[0].PRURL != nil {
		t.Fatalf("session json %+v", got)
	}
	raw := w.do("GET", "/api/sessions/"+x.ID, nil, w.as(w.owner.ID)).Body.String()
	for _, k := range []string{`"issue_url":null`, `"finished_at":null`, `"pr_number":null`, `"pr_url":null`, `"cancel_requested":false`} {
		if !strings.Contains(raw, k) {
			t.Errorf("session json lacks %s: %s", k, raw)
		}
	}

	// Validation.
	eleven := make([]string, 11)
	for i := range eleven {
		r, err := w.st.UpsertRepo(context.Background(), store.UpsertRepoParams{InstallationID: w.inst.InstallationID, GitHubRepoID: randID(), FullName: "acme/r" + itoa(i)})
		if err != nil {
			t.Fatal(err)
		}
		eleven[i] = r.ID
	}
	for name, tc := range map[string]struct {
		body  map[string]any
		field string
	}{
		"no prompt":      {map[string]any{"repo_ids": []string{w.repo.ID}}, "prompt"},
		"blank prompt":   {map[string]any{"prompt": "  ", "repo_ids": []string{w.repo.ID}}, "prompt"},
		"long prompt":    {map[string]any{"prompt": strings.Repeat("p", maxPromptLen+1), "repo_ids": []string{w.repo.ID}}, "prompt"},
		"no repos":       {map[string]any{"prompt": "p", "repo_ids": []string{}}, "repo_ids"},
		"eleven repos":   {map[string]any{"prompt": "p", "repo_ids": eleven}, "repo_ids"},
		"duplicate":      {map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID, w.repo.ID}}, "repo_ids"},
		"inaccessible":   {map[string]any{"prompt": "p", "repo_ids": []string{w.gone.ID}}, "repo_ids"},
		"other org repo": {map[string]any{"prompt": "p", "repo_ids": []string{w.otherOrgsRepo.ID}}, "repo_ids"},
		"bad repo id":    {map[string]any{"prompt": "p", "repo_ids": []string{"x"}}, "repo_ids"},
		"repo_ids type":  {map[string]any{"prompt": "p", "repo_ids": "x"}, "repo_ids"},
		"bad issue":      {map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}, "issue_url": "https://gitlab.com/a/b/issues/1"}, "issue_url"},
		"bad provider":   {map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}, "provider": "openai"}, "provider"},
		"long model":     {map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}, "model": strings.Repeat("m", 101)}, "model"},
	} {
		e := wantError(t, w.do("POST", path, tc.body, w.as(w.member.ID)), 422, codeValidationFailed)
		if e.Fields[tc.field] == "" {
			t.Errorf("%s: fields %+v", name, e.Fields)
		}
	}
	// An issue in a repo the org cannot use: issue_repo_not_tagged.
	for _, u := range []string{"https://github.com/zeta/app/issues/3", "https://github.com/acme/gone/issues/3"} {
		e := wantError(t, w.do("POST", path, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}, "issue_url": u}, w.as(w.member.ID)), 422, codeValidationFailed)
		if e.Fields["issue_url"] != "issue_repo_not_tagged" {
			t.Errorf("%s: fields %+v", u, e.Fields)
		}
	}
	// Access.
	wantError(t, w.do("POST", path, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}, w.as(w.outsider.ID)), 403, codeNotAMember)
	wantError(t, w.do("POST", path, `{"prompt":`, w.as(w.member.ID)), 400, codeBadRequest)
	if n := len(w.q.Runs()); n != 1 {
		t.Fatalf("rejected creates enqueued: %d runs", n)
	}
}

func TestCreateSessionIssueURLIT(t *testing.T) {
	w := newWorld(t)
	// The issue's repo is added after the tagged ones.
	x := w.create(t, map[string]any{"prompt": "see issue", "repo_ids": []string{w.repo2.ID}, "issue_url": "https://github.com/ACME/web/issues/42"})
	if len(x.Repos) != 2 || x.Repos[0].RepoID != w.repo2.ID || x.Repos[1].RepoID != w.repo.ID {
		t.Fatalf("repos %+v", x.Repos)
	}
	got := decode[sessionJSON](t, w.do("GET", "/api/sessions/"+x.ID, nil, w.as(w.member.ID)), 200)
	if got.IssueURL == nil || *got.IssueURL != "https://github.com/ACME/web/issues/42" {
		t.Fatalf("issue_url %v", got.IssueURL)
	}
	// Already tagged: not duplicated. Issue only: empty prompt and no
	// repo_ids are fine.
	x = w.create(t, map[string]any{"repo_ids": []string{w.repo.ID}, "issue_url": "https://github.com/acme/web/issues/1"})
	if len(x.Repos) != 1 {
		t.Fatalf("repos %+v", x.Repos)
	}
	x = w.create(t, map[string]any{"prompt": "", "issue_url": "https://github.com/acme/api/pull/7"})
	if len(x.Repos) != 1 || x.Repos[0].RepoID != w.repo2.ID || x.Prompt != "" {
		t.Fatalf("issue-only session %+v", x)
	}
}

func TestCreateSessionProviderIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	path := "/api/orgs/" + w.org.ID + "/sessions"
	base := func(extra map[string]any) map[string]any {
		b := map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}
		for k, v := range extra {
			b[k] = v
		}
		return b
	}

	// Gemini has no key: provider_unavailable.
	wantError(t, w.do("POST", path, base(map[string]any{"provider": "gemini"}), w.as(w.member.ID)), 422, codeProviderUnavailable)
	// With a key but no default model, a model is required.
	w.cfg.GeminiAPIKey = "g"
	e := wantError(t, w.do("POST", path, base(map[string]any{"provider": "gemini"}), w.as(w.member.ID)), 422, codeValidationFailed)
	if e.Fields["model"] == "" {
		t.Fatalf("fields %+v", e.Fields)
	}
	x := w.create(t, base(map[string]any{"provider": "gemini", "model": "gemini-x"}))
	if x.Provider != "gemini" || x.Model != "gemini-x" {
		t.Fatalf("explicit model %+v", x)
	}
	w.cfg.DefaultGeminiModel = "gemini-3-pro"
	x = w.create(t, base(map[string]any{"provider": "gemini"}))
	if x.Model != "gemini-3-pro" {
		t.Fatalf("gemini default model %q", x.Model)
	}
	// The org default applies when the request names no provider.
	if err := w.st.SetOrgDefaultLLMProvider(ctx, w.org.ID, "gemini"); err != nil {
		t.Fatal(err)
	}
	x = w.create(t, base(nil))
	if x.Provider != "gemini" || x.Model != "gemini-3-pro" {
		t.Fatalf("org default %+v", x)
	}
	// The request's provider wins over the org's.
	x = w.create(t, base(map[string]any{"provider": "anthropic", "model": "claude-custom"}))
	if x.Provider != "anthropic" || x.Model != "claude-custom" {
		t.Fatalf("request provider %+v", x)
	}
	// An org default without a key is unavailable.
	w.cfg.GeminiAPIKey = ""
	wantError(t, w.do("POST", path, base(nil), w.as(w.member.ID)), 422, codeProviderUnavailable)
}

func TestCreateSessionCapIT(t *testing.T) {
	w := newWorld(t)
	w.cfg.MaxActiveSessionsPerOrg = 2
	body := map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}
	a := w.create(t, body)
	b := w.create(t, body)
	resp := w.do("POST", "/api/orgs/"+w.org.ID+"/sessions", body, w.as(w.member.ID))
	wantError(t, resp, http.StatusTooManyRequests, codeRateLimited)
	if resp.Header().Get("Retry-After") != "30" {
		t.Errorf("Retry-After %q", resp.Header().Get("Retry-After"))
	}
	// Waiting and finished sessions do not count.
	w.setStatus(t, a.ID, events.StatusWaitingForUser)
	w.create(t, body)
	w.setStatus(t, b.ID, events.StatusCompleted)
	w.create(t, body)
	wantError(t, w.do("POST", "/api/orgs/"+w.org.ID+"/sessions", body, w.as(w.member.ID)), 429, codeRateLimited)

	// Concurrent creates never exceed the cap.
	w2 := newWorld(t)
	w2.cfg.MaxActiveSessionsPerOrg = 3
	codes := make(chan int, 10)
	for range 10 {
		go func() {
			codes <- w2.do("POST", "/api/orgs/"+w2.org.ID+"/sessions", map[string]any{"prompt": "p", "repo_ids": []string{w2.repo.ID}}, w2.as(w2.member.ID)).Code
		}()
	}
	created := 0
	for range 10 {
		if <-codes == http.StatusCreated {
			created++
		}
	}
	if created != 3 {
		t.Fatalf("created %d sessions concurrently, cap 3", created)
	}
}

// In V1 a session's repos must share one installation: the worker mints
// one token for all of them, and fails a session that spans two. The API
// refuses it up front instead of accepting it and counting it against
// the cap.
func TestCreateSessionOneInstallationIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	inst2, err := w.st.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: w.org.ID, InstallationID: randID(), AccountLogin: "acme-two"})
	if err != nil {
		t.Fatal(err)
	}
	other, err := w.st.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: inst2.InstallationID, GitHubRepoID: randID(), FullName: "acme-two/x"})
	if err != nil {
		t.Fatal(err)
	}
	base := "/api/orgs/" + w.org.ID + "/sessions"
	for name, body := range map[string]map[string]any{
		"two installations": {"prompt": "p", "repo_ids": []string{w.repo.ID, other.ID}},
		"issue on another":  {"prompt": "p", "repo_ids": []string{w.repo.ID}, "issue_url": "https://github.com/acme-two/x/issues/3"},
	} {
		e := wantError(t, w.do("POST", base, body, w.as(w.member.ID)), 422, codeValidationFailed)
		if !strings.Contains(e.Fields["repo_ids"], "one GitHub App installation") {
			t.Errorf("%s: fields %+v", name, e.Fields)
		}
	}
	if page, _ := w.st.ListSessions(ctx, store.ListSessionsParams{OrgID: w.org.ID}); len(page.Sessions) != 0 {
		t.Fatalf("sessions created: %+v", page.Sessions)
	}
	if n := len(w.q.Runs()); n != 0 {
		t.Fatalf("%d run_session enqueued", n)
	}
	// Several repos of one installation, and one repo of the other, are fine.
	w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID, w.repo2.ID}})
	w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{other.ID}, "issue_url": "https://github.com/acme-two/x/issues/3"})
}

// The per-org cap also holds for resumes: a follow-up to a finished
// session, or an answer to a waiting one, makes the session active again.
func TestResumeCapIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	w.cfg.MaxActiveSessionsPerOrg = 1
	body := map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}
	var done []store.Session
	for range 3 {
		x := w.create(t, body)
		w.setStatus(t, x.ID, events.StatusCancelled) // e.g. stopped at once
		done = append(done, x)
	}
	waiting := w.create(t, body)
	if err := w.st.SetOpenQuestion(ctx, waiting.ID, "toolu_1"); err != nil {
		t.Fatal(err)
	}
	w.setStatus(t, waiting.ID, events.StatusWaitingForUser)
	active := w.create(t, body) // the org is now at its cap
	wantError(t, w.do("POST", "/api/orgs/"+w.org.ID+"/sessions", body, w.as(w.member.ID)), 429, codeRateLimited)

	for _, x := range done {
		resp := w.do("POST", "/api/sessions/"+x.ID+"/messages", map[string]string{"text": "go"}, w.as(w.member.ID))
		wantError(t, resp, http.StatusTooManyRequests, codeRateLimited)
		if resp.Header().Get("Retry-After") != "30" {
			t.Errorf("Retry-After %q", resp.Header().Get("Retry-After"))
		}
		if p, _ := w.st.ListPendingInputs(ctx, x.ID); len(p) != 0 {
			t.Fatalf("refused message stored: %+v", p)
		}
	}
	answer := map[string]string{"question_id": "toolu_1", "text": "yes"}
	wantError(t, w.do("POST", "/api/sessions/"+waiting.ID+"/answers", answer, w.as(w.member.ID)), 429, codeRateLimited)
	if got, _ := w.st.GetSession(ctx, waiting.ID); got.OpenQuestionID != "toolu_1" {
		t.Fatalf("question closed by a refused answer: %q", got.OpenQuestionID)
	}
	if n := len(w.q.Resumes()); n != 0 {
		t.Fatalf("%d resumes enqueued while the org is at its cap", n)
	}
	// Steering the active session is not a resume, so it is not capped.
	decode[messageResponse](t, w.do("POST", "/api/sessions/"+active.ID+"/messages", map[string]string{"text": "also"}, w.as(w.member.ID)), 202)

	// Once the active session finishes, a resume fits again.
	w.setStatus(t, active.ID, events.StatusCompleted)
	decode[messageResponse](t, w.do("POST", "/api/sessions/"+waiting.ID+"/answers", answer, w.as(w.member.ID)), 202)
	if rs := w.q.Resumes(); len(rs) != 1 || rs[0].SessionID != waiting.ID {
		t.Fatalf("resumes %+v", rs)
	}
}

func TestCreateSessionEnqueueFailureIT(t *testing.T) {
	w := newWorld(t)
	w.q.SetErr(errors.New("redis down"))
	wantError(t, w.do("POST", "/api/orgs/"+w.org.ID+"/sessions", map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}, w.as(w.member.ID)),
		500, codeInternal)
	page, err := w.st.ListSessions(context.Background(), store.ListSessionsParams{OrgID: w.org.ID})
	if err != nil || len(page.Sessions) != 1 || page.Sessions[0].Status != events.StatusFailed {
		t.Fatalf("unqueued session %+v %v", page.Sessions, err)
	}
}

func TestListAndGetSessionsIT(t *testing.T) {
	w := newWorld(t)
	path := "/api/orgs/" + w.org.ID + "/sessions"
	var ids []string
	for i := range 3 {
		ids = append(ids, w.create(t, map[string]any{"prompt": "p" + itoa(i), "repo_ids": []string{w.repo.ID}}).ID)
	}
	page := decode[sessionPageJSON](t, w.do("GET", path+"?limit=2", nil, w.as(w.member.ID)), 200)
	if len(page.Sessions) != 2 || page.Sessions[0].ID != ids[2] || page.Sessions[1].ID != ids[1] || page.NextCursor == nil {
		t.Fatalf("page 1 %+v", page)
	}
	page = decode[sessionPageJSON](t, w.do("GET", path+"?limit=2&cursor="+*page.NextCursor, nil, w.as(w.member.ID)), 200)
	if len(page.Sessions) != 1 || page.Sessions[0].ID != ids[0] || page.NextCursor != nil {
		t.Fatalf("page 2 %+v", page)
	}
	if raw := w.do("GET", path+"?limit=5", nil, w.as(w.member.ID)).Body.String(); !strings.Contains(raw, `"next_cursor":null`) {
		t.Errorf("last page %s", raw)
	}
	for _, q := range []string{"?limit=0", "?limit=101", "?limit=x", "?cursor=bogus!"} {
		wantError(t, w.do("GET", path+q, nil, w.as(w.member.ID)), 400, codeBadRequest)
	}
	wantError(t, w.do("GET", path, nil, w.as(w.outsider.ID)), 403, codeNotAMember)
	empty := decode[sessionPageJSON](t, w.do("GET", "/api/orgs/"+w.otherOrg.ID+"/sessions", nil, w.as(w.outsider.ID)), 200)
	if empty.Sessions == nil || len(empty.Sessions) != 0 {
		t.Fatalf("empty list %+v", empty)
	}

	// By ID: other orgs and unknown IDs are 404, malformed 400.
	wantError(t, w.do("GET", "/api/sessions/"+ids[0], nil, w.as(w.outsider.ID)), 404, codeNotFound)
	wantError(t, w.do("GET", "/api/sessions/00000000-0000-4000-8000-000000000000", nil, w.as(w.member.ID)), 404, codeNotFound)
	wantError(t, w.do("GET", "/api/sessions/nope", nil, w.as(w.member.ID)), 400, codeBadRequest)

	// PRs and terminal status show up.
	ctx := context.Background()
	if err := w.st.SetSessionRepoPR(ctx, ids[0], w.repo.ID, 7, "https://github.com/acme/web/pull/7"); err != nil {
		t.Fatal(err)
	}
	w.setStatus(t, ids[0], events.StatusCompleted)
	got := decode[sessionJSON](t, w.do("GET", "/api/sessions/"+ids[0], nil, w.as(w.member.ID)), 200)
	if got.Status != events.StatusCompleted || got.FinishedAt == nil || got.Repos[0].PRNumber == nil || *got.Repos[0].PRNumber != 7 {
		t.Fatalf("finished session %+v", got)
	}
}

func TestMessagesIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	x := w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}})
	path := "/api/sessions/" + x.ID + "/messages"

	// Running statuses: stored, no resume.
	for _, st := range []events.SessionStatus{events.StatusQueued, events.StatusRunning} {
		w.setStatus(t, x.ID, st)
		resp := decode[messageResponse](t, w.do("POST", path, map[string]string{"text": "  use dayjs  "}, w.as(w.member.ID)), 202)
		in, err := w.st.GetInput(ctx, x.ID, resp.MessageID)
		if err != nil || !strings.HasPrefix(resp.MessageID, "m_") || in.Text != "use dayjs" || in.Kind != store.InputMessage || in.UserID != w.member.ID {
			t.Fatalf("%s: input %+v %v", st, in, err)
		}
	}
	if n := len(w.q.Resumes()); n != 0 {
		t.Fatalf("resume enqueued for a running session: %d", n)
	}

	// waiting_for_user: answer_required.
	w.setStatus(t, x.ID, events.StatusWaitingForUser)
	wantError(t, w.do("POST", path, map[string]string{"text": "hi"}, w.as(w.member.ID)), 409, codeAnswerRequired)

	// Terminal: stored and resume_session enqueued.
	w.setStatus(t, x.ID, events.StatusCompleted)
	resp := decode[messageResponse](t, w.do("POST", path, map[string]string{"text": "one more thing"}, w.as(w.member.ID)), 202)
	rs := w.q.Resumes()
	if len(rs) != 1 || rs[0] != (queue.ResumeSessionPayload{SessionID: x.ID, Reason: queue.ResumeMessage, InputID: resp.MessageID}) {
		t.Fatalf("resumes %+v", rs)
	}

	// The enqueue fails: 500 and the message is gone, so a retry is clean.
	w.q.SetErr(errors.New("down"))
	wantError(t, w.do("POST", path, map[string]string{"text": "lost?"}, w.as(w.member.ID)), 500, codeInternal)
	w.q.SetErr(nil)
	pending, _ := w.st.ListPendingInputs(ctx, x.ID)
	for _, p := range pending {
		if p.Text == "lost?" {
			t.Fatal("undone message still pending")
		}
	}

	// Validation and access.
	for _, body := range []any{map[string]string{"text": "   "}, map[string]string{}, map[string]string{"text": strings.Repeat("x", maxTextLen+1)}} {
		e := wantError(t, w.do("POST", path, body, w.as(w.member.ID)), 422, codeValidationFailed)
		if e.Fields["text"] == "" {
			t.Errorf("fields %+v", e.Fields)
		}
	}
	wantError(t, w.do("POST", path, map[string]string{"text": "hi"}, w.as(w.outsider.ID)), 404, codeNotFound)
	wantError(t, w.do("POST", path, map[string]string{"text": "hi"}, withOrigin("https://evil.test"), w.as(w.member.ID)), 403, codeBadOrigin)
	if n := len(w.evs.All(x.ID)); n != 0 {
		t.Fatalf("API wrote %d events", n)
	}
}

func TestAnswersIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	x := w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}})
	path := "/api/sessions/" + x.ID + "/answers"
	answer := map[string]string{"question_id": "toolu_1", "text": "Use the sandbox key"}

	wantError(t, w.do("POST", path, answer, w.as(w.member.ID)), 409, codeNotWaiting)
	if err := w.st.SetOpenQuestion(ctx, x.ID, "toolu_1"); err != nil {
		t.Fatal(err)
	}
	w.setStatus(t, x.ID, events.StatusRunning)
	wantError(t, w.do("POST", path, answer, w.as(w.member.ID)), 409, codeNotWaiting)
	w.setStatus(t, x.ID, events.StatusWaitingForUser)

	wantError(t, w.do("POST", path, map[string]string{"question_id": "toolu_0", "text": "x"}, w.as(w.member.ID)), 409, codeQuestionMismatch)
	e := wantError(t, w.do("POST", path, map[string]string{"text": "x"}, w.as(w.member.ID)), 422, codeValidationFailed)
	if e.Fields["question_id"] == "" {
		t.Errorf("fields %+v", e.Fields)
	}
	wantError(t, w.do("POST", path, map[string]string{"question_id": "toolu_1", "text": " "}, w.as(w.member.ID)), 422, codeValidationFailed)
	wantError(t, w.do("POST", path, answer, w.as(w.outsider.ID)), 404, codeNotFound)

	// The enqueue fails: 500, the answer is undone and the question
	// reopened.
	w.q.SetErr(errors.New("down"))
	wantError(t, w.do("POST", path, answer, w.as(w.member.ID)), 500, codeInternal)
	w.q.SetErr(nil)
	if got, _ := w.st.GetSession(ctx, x.ID); got.OpenQuestionID != "toolu_1" {
		t.Fatalf("question not reopened: %q", got.OpenQuestionID)
	}
	if p, _ := w.st.ListPendingInputs(ctx, x.ID); len(p) != 0 {
		t.Fatalf("undone answer pending: %+v", p)
	}

	resp := decode[messageResponse](t, w.do("POST", path, answer, w.as(w.member.ID)), 202)
	in, err := w.st.GetInput(ctx, x.ID, resp.MessageID)
	if err != nil || in.Kind != store.InputAnswer || in.QuestionID != "toolu_1" || in.Text != "Use the sandbox key" {
		t.Fatalf("answer %+v %v", in, err)
	}
	rs := w.q.Resumes()
	if len(rs) != 1 || rs[0] != (queue.ResumeSessionPayload{SessionID: x.ID, Reason: queue.ResumeAnswer, InputID: resp.MessageID}) {
		t.Fatalf("resumes %+v", rs)
	}
	// Answered: the question is no longer open.
	wantError(t, w.do("POST", path, answer, w.as(w.member.ID)), 409, codeQuestionMismatch)
	if n := len(w.evs.All(x.ID)); n != 0 {
		t.Fatalf("API wrote %d events", n)
	}
}

// A message or answer that would resume a session is refused once the
// org can no longer use one of the session's repos, most importantly
// when the repo moved to another org's installation: the worker would
// otherwise run org A's session on org B's installation.
func TestResumeAfterRepoAccessLostIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	x := w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID, w.repo2.ID}})
	msg := func() *httptest.ResponseRecorder {
		return w.do("POST", "/api/sessions/"+x.ID+"/messages", map[string]string{"text": "push to main please"}, w.as(w.member.ID))
	}
	answer := func() *httptest.ResponseRecorder {
		return w.do("POST", "/api/sessions/"+x.ID+"/answers", map[string]string{"question_id": "toolu_1", "text": "yes"}, w.as(w.member.ID))
	}
	// fatal takes a store call's (value, error) and fails t on the error.
	fatal := func(_ any, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	nothingQueued := func(what string) {
		t.Helper()
		if n := len(w.q.Resumes()); n != 0 {
			t.Fatalf("%s: %d resumes enqueued", what, n)
		}
		if p, _ := w.st.ListPendingInputs(ctx, x.ID); len(p) != 0 {
			t.Fatalf("%s: input stored: %+v", what, p)
		}
	}

	// A running session still takes steering messages: no resume.
	w.setStatus(t, x.ID, events.StatusRunning)
	fatal(w.st.SetReposInaccessible(ctx, w.inst.InstallationID, []int64{w.repo2.GitHubRepoID}))
	decode[messageResponse](t, msg(), 202)
	w.setStatus(t, x.ID, events.StatusCompleted)
	pending, err := w.st.ListPendingInputs(ctx, x.ID)
	fatal(pending, err)
	fatal(w.st.MarkInputsConsumed(ctx, x.ID, []string{pending[0].ID}))

	// One repo removed from the installation: no follow-up.
	wantError(t, msg(), 409, codeRepoAccessLost)
	nothingQueued("removed repo")
	fatal(w.st.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: w.inst.InstallationID, GitHubRepoID: w.repo2.GitHubRepoID, FullName: "acme/api"}))

	// The App is uninstalled from org A and reinstalled into org B: the
	// repo rows move to org B's installation.
	fatal(nil, w.st.SetInstallationStatus(ctx, w.inst.InstallationID, store.InstallationDeleted))
	fatal(w.st.SetInstallationReposInaccessible(ctx, w.inst.InstallationID))
	newInst := randID()
	fatal(w.st.UpsertInstallation(ctx, store.UpsertInstallationParams{OrgID: w.otherOrg.ID, InstallationID: newInst, AccountLogin: "acme"}))
	fatal(w.st.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: newInst, GitHubRepoID: w.repo.GitHubRepoID, FullName: "acme/web"}))
	fatal(w.st.UpsertRepo(ctx, store.UpsertRepoParams{InstallationID: newInst, GitHubRepoID: w.repo2.GitHubRepoID, FullName: "acme/api"}))
	wantError(t, msg(), 409, codeRepoAccessLost)
	nothingQueued("moved repos, message")

	// An open question cannot be answered either, and stays open.
	fatal(nil, w.st.SetOpenQuestion(ctx, x.ID, "toolu_1"))
	w.setStatus(t, x.ID, events.StatusWaitingForUser)
	wantError(t, answer(), 409, codeRepoAccessLost)
	nothingQueued("moved repos, answer")
	if got, _ := w.st.GetSession(ctx, x.ID); got.OpenQuestionID != "toolu_1" {
		t.Fatalf("question closed by a refused answer: %q", got.OpenQuestionID)
	}
	// Org B's members cannot see org A's session at all.
	wantError(t, w.do("POST", "/api/sessions/"+x.ID+"/messages", map[string]string{"text": "hi"}, w.as(w.outsider.ID)), 404, codeNotFound)
}

func TestStopIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	stop := func(id string) int { return w.do("POST", "/api/sessions/"+id+"/stop", nil, w.as(w.member.ID)).Code }
	body := map[string]any{"prompt": "p", "repo_ids": []string{w.repo.ID}}

	// queued: flag, signal, and a resume so a worker records it.
	q := w.create(t, body)
	resp := w.do("POST", "/api/sessions/"+q.ID+"/stop", nil, w.as(w.member.ID))
	if resp.Code != 202 || strings.TrimSpace(resp.Body.String()) != "{}" {
		t.Fatalf("stop %d %s", resp.Code, resp.Body)
	}
	if got, _ := w.st.GetSession(ctx, q.ID); got.CancelRequestedAt == nil {
		t.Fatal("cancel flag not set")
	}
	if p := w.ctl.Published(); len(p) != 1 || p[0] != (control.Published{SessionID: q.ID, Signal: control.SignalCancel}) {
		t.Fatalf("published %+v", p)
	}
	if rs := w.q.Resumes(); len(rs) != 1 || rs[0] != (queue.ResumeSessionPayload{SessionID: q.ID, Reason: queue.ResumeCancel}) {
		t.Fatalf("resumes %+v", rs)
	}
	got := decode[sessionJSON](t, w.do("GET", "/api/sessions/"+q.ID, nil, w.as(w.member.ID)), 200)
	if !got.CancelRequested {
		t.Fatal("cancel_requested false")
	}
	// Idempotent.
	if c := stop(q.ID); c != 202 {
		t.Fatalf("second stop %d", c)
	}

	// running: flag and signal only; a worker holds it.
	r := w.create(t, body)
	w.setStatus(t, r.ID, events.StatusRunning)
	before := len(w.q.Resumes())
	if c := stop(r.ID); c != 202 || len(w.q.Resumes()) != before {
		t.Fatalf("running stop %d, resumes %d -> %d", c, before, len(w.q.Resumes()))
	}
	// waiting_for_user: resume.
	wt := w.create(t, body)
	w.setStatus(t, wt.ID, events.StatusWaitingForUser)
	if c := stop(wt.ID); c != 202 || len(w.q.Resumes()) != before+1 {
		t.Fatalf("waiting stop %d", c)
	}
	// Terminal: not_running.
	done := w.create(t, body)
	w.setStatus(t, done.ID, events.StatusCancelled)
	wantError(t, w.do("POST", "/api/sessions/"+done.ID+"/stop", nil, w.as(w.member.ID)), 409, codeNotRunning)
	if got, _ := w.st.GetSession(ctx, done.ID); got.CancelRequestedAt != nil {
		t.Fatal("terminal session flagged")
	}
	// Access.
	wantError(t, w.do("POST", "/api/sessions/"+r.ID+"/stop", nil, w.as(w.outsider.ID)), 404, codeNotFound)
	wantError(t, w.do("POST", "/api/sessions/"+r.ID+"/stop", nil, withOrigin(""), w.as(w.member.ID)), 403, codeBadOrigin)
}

func TestDiffIT(t *testing.T) {
	w := newWorld(t)
	ctx := context.Background()
	x := w.create(t, map[string]any{"prompt": "p", "repo_ids": []string{w.repo2.ID, w.repo.ID}})
	raw := w.do("GET", "/api/sessions/"+x.ID+"/diff", nil, w.as(w.member.ID)).Body.String()
	want := `{"repos":[{"full_name":"acme/api","diff":"","truncated":false,"updated_at":null},{"full_name":"acme/web","diff":"","truncated":false,"updated_at":null}]}`
	if strings.TrimSpace(raw) != want {
		t.Fatalf("empty diff %s", raw)
	}
	if err := w.st.SetSessionRepoDiff(ctx, x.ID, w.repo.ID, "diff --git a/x b/x", true); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Repos []diffRepoJSON `json:"repos"`
	}
	got = decode[struct {
		Repos []diffRepoJSON `json:"repos"`
	}](t, w.do("GET", "/api/sessions/"+x.ID+"/diff", nil, w.as(w.member.ID)), 200)
	if got.Repos[1].Diff != "diff --git a/x b/x" || !got.Repos[1].Truncated || got.Repos[1].UpdatedAt == nil || got.Repos[0].UpdatedAt != nil {
		t.Fatalf("diff %+v", got)
	}
	wantError(t, w.do("GET", "/api/sessions/"+x.ID+"/diff", nil, w.as(w.outsider.ID)), 404, codeNotFound)
}
