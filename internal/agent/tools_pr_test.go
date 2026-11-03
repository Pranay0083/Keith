package agent_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/agent/agenttest"
	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/sandbox/memsandbox"
)

// gitRepo scripts the git commands open_pr runs.
type gitRepo struct {
	head, status, ahead string
	pushExit            int
	commitExit          int
}

func (g gitRepo) handler() memsandbox.Handler {
	return memsandbox.Route(
		memsandbox.On(memsandbox.Prefix("git rev-parse --abbrev-ref HEAD"), memsandbox.OK(g.head+"\n")),
		memsandbox.On(memsandbox.Prefix("git status --porcelain"), memsandbox.OK(g.status)),
		memsandbox.On(memsandbox.Prefix("git rev-list --count"), memsandbox.OK(g.ahead+"\n")),
		memsandbox.On(memsandbox.Prefix("git add -A"), memsandbox.OK("")),
		memsandbox.On(memsandbox.Contains(" commit "), memsandbox.Exit(g.commitExit, "", "hook failed: lint")),
		memsandbox.On(memsandbox.Prefix("git push"), memsandbox.Exit(g.pushExit, "", "! [rejected] non-fast-forward")),
	)
}

func (h *harness) gitCmds() []string {
	var out []string
	for _, r := range h.sb.ExecRequests() {
		if r.Dir != "/workspace/api" {
			h.t.Errorf("git ran in %q", r.Dir)
		}
		if r.Env["GIT_TERMINAL_PROMPT"] != "0" {
			h.t.Errorf("git may prompt: %+v", r.Env)
		}
		out = append(out, r.Cmd)
	}
	return out
}

var prInput = map[string]any{"title": "Fix login redirect", "body": "Fixes #7.\n\nTested with go test."}

func TestOpenPRNoChanges(t *testing.T) {
	h := newToolHarness(t)
	h.sb.SetHandler(gitRepo{head: "agent/s1", status: "", ahead: "0"}.handler())
	r := h.tool("p1", "open_pr", prInput)
	if r.IsError || !strings.Contains(text(r), "nothing to open a pull request for") {
		t.Fatalf("result = %q", text(r))
	}
	if got := h.gitCmds(); !slices.Equal(got, []string{
		"git rev-parse --abbrev-ref HEAD", "git status --porcelain", "git rev-list --count 'origin/main'..HEAD",
	}) {
		t.Fatalf("git = %q", got)
	}
	if len(h.host.PRRequests()) != 0 || len(agenttest.Payloads[events.PROpenedPayload](h.events, events.TypePROpened)) != 0 {
		t.Fatal("opened a PR without changes")
	}
}

func TestOpenPRBranchValidation(t *testing.T) {
	h := newToolHarness(t)
	h.sb.SetHandler(gitRepo{head: "main", status: " M a.go\n", ahead: "0"}.handler())
	wantErrorResult(t, h.tool("p1", "open_pr", prInput), `on branch "main", not the session branch "agent/s1"`)

	for _, bad := range []string{"main", "agent/", "agent/../main", "feature/x", "agent/a b"} {
		h.run.Repos[0].Branch = bad
		wantErrorResult(t, h.tool("p-"+bad, "open_pr", prInput), "Only agent/* branches may be pushed")
	}
	for _, c := range h.gitCmds() {
		if strings.HasPrefix(c, "git push") {
			t.Fatalf("pushed: %q", c)
		}
	}
	if len(h.host.PRRequests()) != 0 {
		t.Fatal("opened a PR")
	}
}

func TestOpenPRHappyPath(t *testing.T) {
	h := newToolHarness(t)
	h.run.GitAuthor = agent.GitIdentity{Name: "keith[bot]", Email: "1+keith[bot]@users.noreply.github.com"}
	var recorded []agent.PR
	h.run.OnPROpened = func(ctx context.Context, rc agent.RepoContext, pr agent.PR) error {
		if n := len(agenttest.Payloads[events.PROpenedPayload](h.events, events.TypePROpened)); n != 0 {
			t.Error("pr.opened emitted before the PR was recorded")
		}
		recorded = append(recorded, pr)
		return nil
	}
	h.sb.SetHandler(gitRepo{head: "agent/s1", status: " M main.go\n?? new.go\n", ahead: "0"}.handler())
	r := h.tool("p1", "open_pr", map[string]any{"title": "Fix login redirect", "body": "Fixes #7.\n\nTested with go test.", "draft": true})
	if r.IsError || text(r) != "Opened draft pull request #100: https://github.com/acme/api/pull/100 (agent/s1 into main)." {
		t.Fatalf("result = %q", text(r))
	}
	want := []string{
		"git rev-parse --abbrev-ref HEAD",
		"git status --porcelain",
		"git rev-list --count 'origin/main'..HEAD",
		"git add -A",
		"git -c user.name='keith[bot]' -c user.email='1+keith[bot]@users.noreply.github.com' commit -q -m 'Fix login redirect' -m 'Fixes #7.\n\nTested with go test.'",
		"git push -u origin 'agent/s1'",
	}
	if got := h.gitCmds(); !slices.Equal(got, want) {
		t.Fatalf("git =\n%q\nwant\n%q", got, want)
	}
	for _, c := range h.gitCmds() {
		if strings.Contains(strings.ToLower(c), "token") || strings.Contains(c, "https://") {
			t.Fatalf("credential-looking command: %q", c)
		}
	}
	reqs := h.host.PRRequests()
	if len(reqs) != 1 || reqs[0] != (agent.PRRequest{Repo: "acme/api", Head: "agent/s1", Base: "main", Title: "Fix login redirect", Body: "Fixes #7.\n\nTested with go test.", Draft: true}) {
		t.Fatalf("PR request = %+v", reqs)
	}
	po := agenttest.Payloads[events.PROpenedPayload](h.events, events.TypePROpened)
	if len(po) != 1 || po[0] != (events.PROpenedPayload{Repo: "acme/api", Number: 100, URL: "https://github.com/acme/api/pull/100", Title: "Fix login redirect", Branch: "agent/s1"}) {
		t.Fatalf("pr.opened = %+v", po)
	}
	if len(recorded) != 1 || recorded[0].Number != 100 {
		t.Fatalf("recorded = %+v", recorded)
	}

	// More work later: commit and push, same PR, no second pr.opened.
	h.reload()
	h.run.OnPROpened = func(context.Context, agent.RepoContext, agent.PR) error { t.Error("recorded twice"); return nil }
	h.sb.ResetCalls()
	r = h.tool("p2", "open_pr", prInput)
	if r.IsError || !strings.Contains(text(r), "existing pull request #100") {
		t.Fatalf("second = %q", text(r))
	}
	if got := h.gitCmds(); !slices.Contains(got, "git push -u origin 'agent/s1'") || !strings.Contains(got[4], "user.name='Keith'") {
		t.Fatalf("second git = %q", got)
	}
	if len(h.host.PRRequests()) != 1 || len(agenttest.Payloads[events.PROpenedPayload](h.events, events.TypePROpened)) != 1 {
		t.Fatal("opened a second PR")
	}
}

func TestOpenPRCommittedButClean(t *testing.T) {
	h := newToolHarness(t)
	h.sb.SetHandler(gitRepo{head: "agent/s1", status: "", ahead: "2"}.handler())
	r := h.tool("p1", "open_pr", prInput)
	if r.IsError || !strings.Contains(text(r), "#100") {
		t.Fatalf("result = %q", text(r))
	}
	for _, c := range h.gitCmds() {
		if strings.Contains(c, "commit") || strings.Contains(c, "add -A") {
			t.Fatalf("committed a clean tree: %q", c)
		}
	}
}

func TestOpenPRFailures(t *testing.T) {
	h := newToolHarness(t)
	h.sb.SetHandler(gitRepo{head: "agent/s1", status: " M a\n", ahead: "0", pushExit: 1}.handler())
	wantErrorResult(t, h.tool("p1", "open_pr", prInput), "git push failed (exit 1):\n! [rejected]")

	h.sb.SetHandler(gitRepo{head: "agent/s1", status: " M a\n", ahead: "0", commitExit: 1}.handler())
	wantErrorResult(t, h.tool("p2", "open_pr", prInput), "hook failed: lint")

	h.sb.SetHandler(gitRepo{head: "agent/s1", status: " M a\n", ahead: "0"}.handler())
	h.host.PRErr = errors.New("422 validation failed")
	wantErrorResult(t, h.tool("p3", "open_pr", prInput), "opening the pull request failed: 422")
	h.host.PRErr = nil
	if len(agenttest.Payloads[events.PROpenedPayload](h.events, events.TypePROpened)) != 0 {
		t.Fatal("pr.opened after a failure")
	}

	wantErrorResult(t, h.tool("p4", "open_pr", map[string]any{"title": " ", "body": ""}), "title is empty")
	wantErrorResult(t, h.tool("p5", "open_pr", map[string]any{"repo": "acme/other", "title": "x", "body": ""}), "not one of this session's repos")
	h.run.Repos = append(h.run.Repos, agent.RepoContext{FullName: "acme/web", Path: "/workspace/web", Branch: "agent/s1", DefaultBranch: "main"})
	wantErrorResult(t, h.tool("p6", "open_pr", prInput), "repo is required")
	h.run.Host = nil
	wantErrorResult(t, h.tool("p7", "open_pr", prInput), "not available")
}
