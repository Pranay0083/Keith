package github

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
)

func newTestHost(t *testing.T, f *fakeGitHub) *Host {
	t.Helper()
	ts, err := NewTokenSource(f.app(t), 77, []string{"acme/api"}, WorkerPermissions)
	if err != nil {
		t.Fatal(err)
	}
	return NewHost(ts)
}

// serveIssue serves issue acme/api#7 with n comments, per to a page.
func serveIssue(f *fakeGitHub, n, per int) *atomic.Int32 {
	pages := new(atomic.Int32)
	f.mux.HandleFunc("GET /repos/acme/api/issues/{num}", func(w http.ResponseWriter, r *http.Request) {
		if !f.authorized(w, r) {
			return
		}
		if r.PathValue("num") != "7" {
			writeJSON(w, http.StatusNotFound, map[string]string{"message": "Not Found"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"number": 7, "title": "Login fails", "body": "Steps to reproduce...", "state": "open",
			"user":     map[string]any{"login": "reporter"},
			"labels":   []map[string]any{{"name": "bug"}, {"name": "p1"}},
			"html_url": "https://github.com/acme/api/issues/7",
			"comments": n,
		})
	})
	base := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	f.mux.HandleFunc("GET /repos/acme/api/issues/7/comments", func(w http.ResponseWriter, r *http.Request) {
		if !f.authorized(w, r) {
			return
		}
		pages.Add(1)
		page := pageParam(r)
		last := max(1, (n+per-1)/per)
		var out []map[string]any
		for i := (page - 1) * per; i < min(page*per, n); i++ {
			out = append(out, map[string]any{
				"id": i + 1, "body": fmt.Sprintf("comment %d", i+1),
				"user":       map[string]any{"login": fmt.Sprintf("user%d", i+1)},
				"created_at": base.Add(time.Duration(i) * time.Hour).Format(time.RFC3339),
			})
		}
		linkHeader(w, r, page, last)
		writeJSON(w, http.StatusOK, out)
	})
	return pages
}

func TestHostReadIssueWithComments(t *testing.T) {
	f := newFakeGitHub(t)
	pages := serveIssue(f, 5, 2)
	h := newTestHost(t, f)

	is, err := h.ReadIssue(context.Background(), "acme/api", 7)
	if err != nil {
		t.Fatal(err)
	}
	if is.Repo != "acme/api" || is.Number != 7 || is.Title != "Login fails" || is.Body != "Steps to reproduce..." ||
		is.State != "open" || is.Author != "reporter" || is.URL != "https://github.com/acme/api/issues/7" ||
		!reflect.DeepEqual(is.Labels, []string{"bug", "p1"}) {
		t.Errorf("issue = %+v", is)
	}
	if len(is.Comments) != 5 || pages.Load() != 3 {
		t.Fatalf("got %d comments over %d pages, want 5 over 3", len(is.Comments), pages.Load())
	}
	want := agent.IssueComment{Author: "user4", Body: "comment 4", CreatedAt: time.Date(2026, 10, 1, 3, 0, 0, 0, time.UTC)}
	if got := is.Comments[3]; got.Author != want.Author || got.Body != want.Body || !got.CreatedAt.Equal(want.CreatedAt) {
		t.Errorf("comment 4 = %+v, want %+v", got, want)
	}

	// The worker token was minted with the worker permissions.
	if got := f.lastMint(t).Body["permissions"]; !reflect.DeepEqual(got, map[string]any{
		"contents": "read", "pull_requests": "write", "issues": "read", "metadata": "read",
	}) {
		t.Errorf("worker token permissions = %v", got)
	}
}

func TestHostReadIssueCapsComments(t *testing.T) {
	f := newFakeGitHub(t)
	pages := serveIssue(f, 130, 50)
	h := newTestHost(t, f)
	is, err := h.ReadIssue(context.Background(), "acme/api", 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(is.Comments) != MaxIssueComments {
		t.Fatalf("got %d comments, want the cap %d", len(is.Comments), MaxIssueComments)
	}
	if is.Comments[0].Body != "comment 1" || is.Comments[99].Body != "comment 100" {
		t.Errorf("kept comments %q..%q, want the oldest 100", is.Comments[0].Body, is.Comments[99].Body)
	}
	if n := pages.Load(); n != 2 {
		t.Errorf("fetched %d comment pages, want 2", n)
	}
}

func TestHostReadIssueErrors(t *testing.T) {
	f := newFakeGitHub(t)
	serveIssue(f, 0, 10)
	h := newTestHost(t, f)
	ctx := context.Background()

	if _, err := h.ReadIssue(ctx, "acme/api", 8); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing issue: err = %v, want ErrNotFound", err)
	}
	if _, err := h.ReadIssue(ctx, "acme/other", 7); !errors.Is(err, errOutOfScope) {
		t.Errorf("repo outside the token scope: err = %v", err)
	}
	if _, err := h.ReadIssue(ctx, "evil/api", 7); !errors.Is(err, errOutOfScope) {
		t.Errorf("same name, other owner: err = %v", err)
	}
	for _, bad := range []string{"api", "acme/api/x", ""} {
		if _, err := h.ReadIssue(ctx, bad, 7); err == nil {
			t.Errorf("ReadIssue(%q) succeeded", bad)
		}
	}
	if _, err := h.ReadIssue(ctx, "acme/api", 0); err == nil {
		t.Error("issue 0 succeeded")
	}
	if is, err := h.ReadIssue(ctx, "acme/api", 7); err != nil || is.Comments != nil {
		t.Errorf("issue without comments = %+v, %v", is, err)
	}
}

// fakePulls serves list and create for acme/api pulls.
type fakePulls struct {
	mu        sync.Mutex
	prs       []map[string]any
	creates   []map[string]any
	lists     []string
	hideFirst bool // the first list misses PRs created by someone else
	conflict  bool // create answers 422 "already exists" and records the PR
}

func (p *fakePulls) pr(num int, head, base string) map[string]any {
	return map[string]any{
		"number": num, "html_url": fmt.Sprintf("https://github.com/acme/api/pull/%d", num),
		"head": map[string]any{"ref": head, "repo": map[string]any{"full_name": "acme/api"}},
		"base": map[string]any{"ref": base},
	}
}

// snapshot returns the create bodies and list queries seen so far.
func (p *fakePulls) snapshot() ([]map[string]any, []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]map[string]any(nil), p.creates...), append([]string(nil), p.lists...)
}

func (p *fakePulls) serve(f *fakeGitHub) {
	f.mux.HandleFunc("GET /repos/acme/api/pulls", func(w http.ResponseWriter, r *http.Request) {
		if !f.authorized(w, r) {
			return
		}
		p.mu.Lock()
		defer p.mu.Unlock()
		q := r.URL.Query()
		p.lists = append(p.lists, q.Encode())
		out := []map[string]any{}
		if p.hideFirst && len(p.lists) == 1 {
			writeJSON(w, http.StatusOK, out)
			return
		}
		for _, pr := range p.prs {
			if "acme:"+pr["head"].(map[string]any)["ref"].(string) == q.Get("head") &&
				pr["base"].(map[string]any)["ref"] == q.Get("base") && q.Get("state") == "open" {
				out = append(out, pr)
			}
		}
		writeJSON(w, http.StatusOK, out)
	})
	f.mux.HandleFunc("POST /repos/acme/api/pulls", func(w http.ResponseWriter, r *http.Request) {
		if !f.authorized(w, r) {
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		p.mu.Lock()
		defer p.mu.Unlock()
		p.creates = append(p.creates, body)
		if p.conflict {
			writeJSON(w, http.StatusUnprocessableEntity, map[string]any{
				"message": "Validation Failed",
				"errors": []map[string]any{{
					"resource": "PullRequest", "code": "custom",
					"message": "A pull request already exists for acme:" + body["head"].(string) + ".",
				}},
			})
			return
		}
		pr := p.pr(100+len(p.prs), body["head"].(string), body["base"].(string))
		p.prs = append(p.prs, pr)
		writeJSON(w, http.StatusCreated, pr)
	})
}

func TestHostOpenPRCreates(t *testing.T) {
	f := newFakeGitHub(t)
	pulls := &fakePulls{}
	pulls.serve(f)
	h := newTestHost(t, f)

	req := agent.PRRequest{Repo: "acme/api", Head: "agent/s1", Base: "main", Title: "Fix login", Body: "Closes #7", Draft: true}
	pr, err := h.OpenPR(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	want := agent.PR{Repo: "acme/api", Number: 100, URL: "https://github.com/acme/api/pull/100", Head: "agent/s1", Base: "main"}
	if pr != want {
		t.Errorf("PR = %+v, want %+v", pr, want)
	}
	wantBody := map[string]any{"title": "Fix login", "head": "agent/s1", "base": "main", "body": "Closes #7", "draft": true}
	creates, lists := pulls.snapshot()
	if len(creates) != 1 || !reflect.DeepEqual(creates[0], wantBody) {
		t.Errorf("create bodies = %v, want [%v]", creates, wantBody)
	}
	if len(lists) != 1 || lists[0] != "base=main&head=acme%3Aagent%2Fs1&per_page=10&state=open" {
		t.Errorf("list queries = %v", lists)
	}

	// A retry finds the open PR and creates nothing.
	again, err := h.OpenPR(context.Background(), req)
	if err != nil || again != want {
		t.Errorf("retry = %+v, %v; want %+v", again, err, want)
	}
	if creates, _ := pulls.snapshot(); len(creates) != 1 {
		t.Errorf("retry created another PR: %d creates", len(creates))
	}
}

func TestHostOpenPRRecoversFromAlreadyExists(t *testing.T) {
	f := newFakeGitHub(t)
	pulls := &fakePulls{hideFirst: true, conflict: true}
	pulls.prs = append(pulls.prs, pulls.pr(42, "agent/s1", "main"))
	pulls.serve(f)
	h := newTestHost(t, f)

	pr, err := h.OpenPR(context.Background(), agent.PRRequest{Repo: "acme/api", Head: "agent/s1", Base: "main", Title: "Fix"})
	if err != nil {
		t.Fatal(err)
	}
	if pr.Number != 42 {
		t.Errorf("PR = %+v, want the existing #42", pr)
	}
	if creates, lists := pulls.snapshot(); len(creates) != 1 || len(lists) != 2 {
		t.Errorf("creates = %d, lists = %d; want 1 and 2", len(creates), len(lists))
	}
}

func TestHostOpenPRRejects(t *testing.T) {
	f := newFakeGitHub(t)
	pulls := &fakePulls{}
	pulls.serve(f)
	h := newTestHost(t, f)
	ok := agent.PRRequest{Repo: "acme/api", Head: "agent/s1", Base: "main", Title: "Fix"}
	cases := map[string]func(*agent.PRRequest){
		"head main":         func(r *agent.PRRequest) { r.Head = "main" },
		"head feature":      func(r *agent.PRRequest) { r.Head = "feature/x" },
		"head agent only":   func(r *agent.PRRequest) { r.Head = "agent/" },
		"no base":           func(r *agent.PRRequest) { r.Base = "" },
		"base is head":      func(r *agent.PRRequest) { r.Base = r.Head },
		"blank title":       func(r *agent.PRRequest) { r.Title = "  " },
		"repo out of scope": func(r *agent.PRRequest) { r.Repo = "acme/web" },
	}
	for name, mod := range cases {
		req := ok
		mod(&req)
		if _, err := h.OpenPR(context.Background(), req); err == nil {
			t.Errorf("%s: OpenPR succeeded", name)
		}
	}
	if creates, lists := pulls.snapshot(); len(creates)+len(lists) != 0 {
		t.Errorf("invalid requests reached GitHub: %d creates, %d lists", len(creates), len(lists))
	}
}
