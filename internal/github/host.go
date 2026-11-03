package github

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	gh "github.com/google/go-github/v93/github"

	"github.com/Pranay0083/keith/internal/agent"
)

// MaxIssueComments caps the comments ReadIssue returns, oldest first.
const MaxIssueComments = 100

var errOutOfScope = errors.New("repo is not in this session's token scope")

// Host is the agent.RepoHost the worker gives the agent. It reaches
// GitHub with a worker TokenSource (WorkerPermissions), from outside the
// sandbox, and only for the repos that source is scoped to.
type Host struct {
	token       func(ctx context.Context) (string, error)
	newClient   func(token string) (*gh.Client, error)
	repos       []string
	maxComments int
	perPage     int
}

var _ agent.RepoHost = (*Host)(nil)

// NewHost returns a Host over tokens, which should carry
// WorkerPermissions and the session's repos. Never give it the sandbox
// TokenSource.
func NewHost(tokens *TokenSource) *Host {
	return &Host{
		token: func(ctx context.Context) (string, error) {
			t, err := tokens.Token(ctx)
			return t.Value, err
		},
		newClient:   tokens.app.client,
		repos:       tokens.repos,
		maxComments: MaxIssueComments,
		perPage:     100,
	}
}

// PATHostOptions configures NewPATHost.
type PATHostOptions struct {
	// APIBaseURL is the REST API base, default DefaultAPIBaseURL.
	APIBaseURL string
	// HTTPClient makes every request. The default has a 30 s timeout.
	HTTPClient *http.Client
}

// NewPATHost returns a Host that authenticates with a personal access
// token instead of an App installation token, for `keith run --local`
// (V0, single user). Like NewHost it serves only the given repos
// (owner/name). The token's own scopes are whatever its owner gave it,
// so prefer a fine-grained token limited to those repos.
func NewPATHost(token string, repos []string, opts PATHostOptions) (*Host, error) {
	if strings.TrimSpace(token) == "" {
		return nil, errors.New("github: empty personal access token")
	}
	if len(repos) == 0 {
		return nil, ErrNoRepos
	}
	for _, r := range repos {
		if _, _, err := splitRepo(r); err != nil {
			return nil, err
		}
	}
	api, err := baseURL("APIBaseURL", opts.APIBaseURL, DefaultAPIBaseURL)
	if err != nil {
		return nil, err
	}
	if !strings.HasSuffix(api, "/") {
		api += "/"
	}
	hc := opts.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: defaultHTTPTimeout}
	}
	return &Host{
		token:       func(context.Context) (string, error) { return token, nil },
		newClient:   func(t string) (*gh.Client, error) { return restClient(hc, api, t) },
		repos:       append([]string(nil), repos...),
		maxComments: MaxIssueComments,
		perPage:     100,
	}, nil
}

func (h *Host) client(ctx context.Context) (*gh.Client, error) {
	t, err := h.token(ctx)
	if err != nil {
		return nil, err
	}
	return h.newClient(t)
}

// scoped splits repo and checks that the token source covers it.
func (h *Host) scoped(repo string) (owner, name string, err error) {
	owner, name, err = splitRepo(repo)
	if err != nil {
		return "", "", err
	}
	for _, r := range h.repos {
		o, n, hasOwner := strings.Cut(r, "/")
		if !hasOwner {
			o, n = "", r
		}
		if strings.EqualFold(n, name) && (o == "" || strings.EqualFold(o, owner)) {
			return owner, name, nil
		}
	}
	return "", "", fmt.Errorf("github: %s: %w", repo, errOutOfScope)
}

// ReadIssue fetches an issue or pull request and up to MaxIssueComments
// of its comments, oldest first.
func (h *Host) ReadIssue(ctx context.Context, repo string, number int) (agent.Issue, error) {
	owner, name, err := h.scoped(repo)
	if err != nil {
		return agent.Issue{}, err
	}
	if number <= 0 {
		return agent.Issue{}, fmt.Errorf("github: invalid issue number %d", number)
	}
	c, err := h.client(ctx)
	if err != nil {
		return agent.Issue{}, err
	}
	is, resp, err := c.Issues.Get(ctx, owner, name, number)
	if err != nil {
		return agent.Issue{}, wrapErr(fmt.Sprintf("get issue %s#%d", repo, number), resp, err)
	}
	out := agent.Issue{
		Repo:   repo,
		Number: number,
		Title:  is.GetTitle(),
		Body:   is.GetBody(),
		State:  is.GetState(),
		Author: is.GetUser().GetLogin(),
		URL:    is.GetHTMLURL(),
	}
	for _, l := range is.Labels {
		out.Labels = append(out.Labels, l.GetName())
	}

	opts := &gh.IssueListCommentsOptions{ListOptions: gh.ListOptions{PerPage: min(h.perPage, h.maxComments)}}
	for range maxPages {
		cs, resp, err := c.Issues.ListComments(ctx, owner, name, number, opts)
		if err != nil {
			return agent.Issue{}, wrapErr(fmt.Sprintf("list comments of %s#%d", repo, number), resp, err)
		}
		for _, cm := range cs {
			if len(out.Comments) == h.maxComments {
				return out, nil
			}
			out.Comments = append(out.Comments, agent.IssueComment{
				Author:    cm.GetUser().GetLogin(),
				Body:      cm.GetBody(),
				CreatedAt: cm.GetCreatedAt().Time,
			})
		}
		if resp.NextPage == 0 || len(out.Comments) == h.maxComments {
			return out, nil
		}
		opts.Page = resp.NextPage
	}
	return out, nil
}

// OpenPR opens a pull request from an agent/* branch already pushed to
// req.Repo. If a PR is already open for the same head and base it
// returns that one, so a retried call opens nothing new: it looks first,
// and also recovers from GitHub's 422 "already exists" if two calls
// race.
func (h *Host) OpenPR(ctx context.Context, req agent.PRRequest) (agent.PR, error) {
	owner, name, err := h.scoped(req.Repo)
	if err != nil {
		return agent.PR{}, err
	}
	if err := ValidateAgentBranch(req.Head); err != nil {
		return agent.PR{}, err
	}
	if req.Base == "" || req.Base == req.Head {
		return agent.PR{}, fmt.Errorf("github: invalid base branch %q", req.Base)
	}
	if strings.TrimSpace(req.Title) == "" {
		return agent.PR{}, errors.New("github: a pull request needs a title")
	}
	c, err := h.client(ctx)
	if err != nil {
		return agent.PR{}, err
	}

	if pr, ok, err := findOpenPR(ctx, c, owner, name, req); err != nil || ok {
		return pr, err
	}
	np := gh.CreatePullRequest{Title: &req.Title, Head: req.Head, Base: req.Base, Draft: &req.Draft}
	if req.Body != "" {
		np.Body = &req.Body
	}
	p, resp, err := c.PullRequests.Create(ctx, owner, name, np)
	if err != nil {
		if alreadyExists(resp, err) {
			if pr, ok, ferr := findOpenPR(ctx, c, owner, name, req); ferr == nil && ok {
				return pr, nil
			}
		}
		return agent.PR{}, wrapErr(fmt.Sprintf("create pull request %s %s->%s", req.Repo, req.Head, req.Base), resp, err)
	}
	return toPR(req.Repo, p), nil
}

// findOpenPR looks for an open PR from owner:head into base.
func findOpenPR(ctx context.Context, c *gh.Client, owner, name string, req agent.PRRequest) (agent.PR, bool, error) {
	prs, resp, err := c.PullRequests.List(ctx, owner, name, &gh.PullRequestListOptions{
		State:       "open",
		Head:        owner + ":" + req.Head,
		Base:        req.Base,
		ListOptions: gh.ListOptions{PerPage: 10},
	})
	if err != nil {
		return agent.PR{}, false, wrapErr(fmt.Sprintf("list pull requests of %s", req.Repo), resp, err)
	}
	for _, p := range prs {
		if p.GetHead().GetRef() != req.Head || p.GetBase().GetRef() != req.Base {
			continue
		}
		if hr := p.GetHead().GetRepo().GetFullName(); hr != "" && !strings.EqualFold(hr, owner+"/"+name) {
			continue
		}
		return toPR(req.Repo, p), true, nil
	}
	return agent.PR{}, false, nil
}

// alreadyExists reports GitHub's 422 for a duplicate pull request.
func alreadyExists(resp *gh.Response, err error) bool {
	if statusCode(resp) != http.StatusUnprocessableEntity {
		return false
	}
	var er *gh.ErrorResponse
	if !errors.As(err, &er) {
		return false
	}
	if strings.Contains(strings.ToLower(er.Message), "already exists") {
		return true
	}
	for _, e := range er.Errors {
		if strings.Contains(strings.ToLower(e.Message), "already exists") {
			return true
		}
	}
	return false
}

func toPR(repo string, p *gh.PullRequest) agent.PR {
	return agent.PR{
		Repo:   repo,
		Number: p.GetNumber(),
		URL:    p.GetHTMLURL(),
		Head:   p.GetHead().GetRef(),
		Base:   p.GetBase().GetRef(),
	}
}
