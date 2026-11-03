package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
	"github.com/Pranay0083/keith/internal/sandbox"
)

// AgentBranchPrefix is the only branch namespace the agent pushes to.
const AgentBranchPrefix = "agent/"

const (
	gitTimeout  = 2 * time.Minute
	pushTimeout = 5 * time.Minute
	maxPRTitle  = 256
)

// ValidateAgentBranch checks that name is agent/<something> and a safe git
// ref name (the same rule as internal/github.ValidateAgentBranch).
func ValidateAgentBranch(name string) error {
	rest, ok := strings.CutPrefix(name, AgentBranchPrefix)
	if !ok || rest == "" {
		return fmt.Errorf("branch %q is not under %s", name, AgentBranchPrefix)
	}
	if len(name) > 255 {
		return fmt.Errorf("branch name is %d bytes, max 255", len(name))
	}
	for _, r := range rest {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' || r == '/') {
			return fmt.Errorf("branch %q has a disallowed character %q", name, r)
		}
	}
	if strings.Contains(rest, "..") || strings.HasSuffix(rest, ".") {
		return fmt.Errorf("branch %q is not a valid ref name", name)
	}
	for c := range strings.SplitSeq(rest, "/") {
		if c == "" || strings.HasPrefix(c, ".") || strings.HasSuffix(c, ".lock") {
			return fmt.Errorf("branch %q is not a valid ref name", name)
		}
	}
	return nil
}

// pickRepo resolves an optional owner/name argument to a session repo.
func pickRepo(run *Run, name string) (RepoContext, *Result) {
	name = strings.TrimSpace(name)
	if name == "" {
		if len(run.Repos) == 1 {
			return run.Repos[0], nil
		}
		r := errorResult("repo is required: give it as owner/name" + repoChoices(run))
		return RepoContext{}, &r
	}
	for _, rc := range run.Repos {
		if strings.EqualFold(rc.FullName, name) {
			return rc, nil
		}
	}
	r := errorResult(fmt.Sprintf("%q is not one of this session's repos%s", name, repoChoices(run)))
	return RepoContext{}, &r
}

type openPRTool struct{}

func (openPRTool) Spec() llm.ToolSpec {
	return spec(toolOpenPR,
		"Commit all changes in a repo, push the session's agent/* branch and open a pull request against the default branch. Run the tests first. If nothing changed, no pull request is opened. Calling it again after more changes pushes them to the same pull request.",
		map[string]any{
			"repo":  strProp("The repository as owner/name. Optional when the session has a single repo."),
			"title": strProp("The pull request title, also used as the commit subject: short, imperative, specific (for example \"Fix pagination off-by-one in /users\")."),
			"body":  strProp("The pull request description in Markdown: what changed and why, how it was tested, and anything the reviewer should check. Reference the issue (for example \"Fixes #12\") when there is one."),
			"draft": boolProp("Open the pull request as a draft, for work that is incomplete or needs discussion."),
		},
		"title", "body")
}

func (openPRTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Repo  string `json:"repo"`
		Title string `json:"title"`
		Body  string `json:"body"`
		Draft bool   `json:"draft"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	if run.Host == nil {
		return errorResult("Opening pull requests is not available in this session."), nil
	}
	rc, bad := pickRepo(run, in.Repo)
	if bad != nil {
		return *bad, nil
	}
	title := oneLine(in.Title, maxPRTitle)
	if title == "" {
		return errorResult("title is empty."), nil
	}
	if err := ValidateAgentBranch(rc.Branch); err != nil {
		return errorResult(fmt.Sprintf("Refusing to push %s: %v. Only agent/* branches may be pushed.", rc.FullName, err)), nil
	}

	git := func(cmd string, timeout time.Duration) (*sandbox.ExecResult, error) {
		return run.Sandbox.Exec(ctx, sandbox.ExecRequest{
			Cmd: cmd, Dir: rc.Path, Timeout: timeout,
			Env: map[string]string{"GIT_TERMINAL_PROMPT": "0"},
		}, nil)
	}
	failed := func(step string, res *sandbox.ExecResult, err error) (Result, error) {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		if err != nil {
			return errorResult(fmt.Sprintf("%s failed: %v", step, err)), nil
		}
		out := strings.TrimSpace(res.Stdout + "\n" + res.Stderr)
		return errorResult(fmt.Sprintf("%s failed (exit %d):\n%s", step, res.ExitCode, tailString(validUTF8(out), 4000))), nil
	}

	res, err := git("git rev-parse --abbrev-ref HEAD", gitTimeout)
	if err != nil || res.ExitCode != 0 {
		return failed("git rev-parse", res, err)
	}
	if head := strings.TrimSpace(res.Stdout); head != rc.Branch {
		return errorResult(fmt.Sprintf("%s is on branch %q, not the session branch %q. Check out %s (keeping your commits on it) and try again.",
			rc.FullName, head, rc.Branch, rc.Branch)), nil
	}
	status, err := git("git status --porcelain", gitTimeout)
	if err != nil || status.ExitCode != 0 {
		return failed("git status", status, err)
	}
	dirty := strings.TrimSpace(status.Stdout) != ""
	ahead := -1
	if rc.DefaultBranch != "" {
		if res, err := git("git rev-list --count "+shellQuote("origin/"+rc.DefaultBranch)+"..HEAD", gitTimeout); err == nil && res.ExitCode == 0 {
			if n, perr := strconv.Atoi(strings.TrimSpace(res.Stdout)); perr == nil {
				ahead = n
			}
		} else if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
	}
	if !dirty && ahead == 0 {
		return textResult("no changes in "+rc.FullName, fmt.Sprintf(
			"There is nothing to open a pull request for in %s: the working tree is clean and %s has no commits beyond origin/%s. No pull request was opened.",
			rc.FullName, rc.Branch, rc.DefaultBranch)), nil
	}

	if dirty {
		if res, err := git("git add -A", gitTimeout); err != nil || res.ExitCode != 0 {
			return failed("git add", res, err)
		}
		author := run.GitAuthor
		if author.Name == "" || author.Email == "" {
			author = DefaultGitAuthor
		}
		cmd := fmt.Sprintf("git -c user.name=%s -c user.email=%s commit -q -m %s",
			shellQuote(author.Name), shellQuote(author.Email), shellQuote(title))
		if body := strings.TrimSpace(in.Body); body != "" {
			cmd += " -m " + shellQuote(body)
		}
		if res, err := git(cmd, gitTimeout); err != nil || res.ExitCode != 0 {
			return failed("git commit", res, err)
		}
	}
	if res, err := git("git push -u origin "+shellQuote(rc.Branch), pushTimeout); err != nil || res.ExitCode != 0 {
		return failed("git push", res, err)
	}

	if run.state != nil {
		if pr, ok := run.state.Aux.PRs[rc.FullName]; ok {
			return textResult(fmt.Sprintf("pushed to PR #%d", pr.Number), fmt.Sprintf(
				"Pushed the new commits to %s; the existing pull request #%d (%s) is updated.", rc.Branch, pr.Number, pr.URL)), nil
		}
	}
	pr, err := run.Host.OpenPR(ctx, PRRequest{
		Repo: rc.FullName, Head: rc.Branch, Base: rc.DefaultBranch,
		Title: title, Body: in.Body, Draft: in.Draft,
	})
	if err != nil {
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return errorResult(fmt.Sprintf("Pushed %s, but opening the pull request failed: %v", rc.Branch, err)), nil
	}
	if run.OnPROpened != nil {
		if err := run.OnPROpened(ctx, rc, pr); err != nil {
			return Result{}, fmt.Errorf("agent: OnPROpened: %w", err)
		}
	}
	if run.state != nil {
		if run.state.Aux.PRs == nil {
			run.state.Aux.PRs = map[string]PR{}
		}
		run.state.Aux.PRs[rc.FullName] = pr
	}
	if err := run.emit(ctx, events.TypePROpened, events.PROpenedPayload{
		Repo: rc.FullName, Number: pr.Number, URL: pr.URL, Title: title, Branch: rc.Branch,
	}); err != nil {
		return Result{}, err
	}
	kind := "pull request"
	if in.Draft {
		kind = "draft pull request"
	}
	return textResult(fmt.Sprintf("opened PR #%d", pr.Number), fmt.Sprintf(
		"Opened %s #%d: %s (%s into %s).", kind, pr.Number, pr.URL, rc.Branch, pr.Base)), nil
}
