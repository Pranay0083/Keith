package github

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

// AgentBranchPrefix is the only branch namespace the agent may push to
// and open pull requests from.
const AgentBranchPrefix = "agent/"

// ValidateAgentBranch checks that name is agent/<something> and a safe
// git ref name: ASCII letters, digits, '.', '_', '-' and '/', no empty
// or dot-led components, no "..", no ".lock" component suffix, no
// trailing '.' or '/', at most 255 bytes.
func ValidateAgentBranch(name string) error {
	rest, ok := strings.CutPrefix(name, AgentBranchPrefix)
	if !ok || rest == "" {
		return fmt.Errorf("github: branch %q is not under %s", name, AgentBranchPrefix)
	}
	if len(name) > 255 {
		return fmt.Errorf("github: branch name is %d bytes, max 255", len(name))
	}
	for _, r := range rest {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' || r == '/') {
			return fmt.Errorf("github: branch %q has a disallowed character %q", name, r)
		}
	}
	if strings.Contains(rest, "..") || strings.HasSuffix(rest, ".") {
		return fmt.Errorf("github: branch %q is not a valid ref name", name)
	}
	for c := range strings.SplitSeq(rest, "/") {
		if c == "" || strings.HasPrefix(c, ".") || strings.HasSuffix(c, ".lock") {
			return fmt.Errorf("github: branch %q is not a valid ref name", name)
		}
	}
	return nil
}

// ParseIssueURL parses a github.com issue or pull request URL, such as
// https://github.com/acme/api/issues/123 or
// https://github.com/acme/api/pull/45, into "acme/api" and the number.
// Pull request URLs are accepted because RepoHost.ReadIssue reads pull
// requests too (their conversation comments). Only https on github.com
// (or www.github.com) is accepted; the path must be exactly
// owner/repo/{issues|pull}/N, with an optional trailing slash, query or
// fragment.
func ParseIssueURL(raw string) (repo string, number int, err error) {
	bad := func(why string) (string, int, error) {
		return "", 0, fmt.Errorf("github: %q is not a GitHub issue URL: %s", raw, why)
	}
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return bad("unparseable")
	}
	if u.Scheme != "https" {
		return bad("not https")
	}
	if host := strings.ToLower(u.Host); host != "github.com" && host != "www.github.com" {
		return bad("host is not github.com")
	}
	if u.User != nil || u.Opaque != "" || u.RawPath != "" {
		return bad("unexpected URL form")
	}
	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) != 4 || strings.Contains(u.Path, "//") {
		return bad("path is not owner/repo/issues/N")
	}
	owner, name, kind, num := parts[0], parts[1], parts[2], parts[3]
	if kind != "issues" && kind != "pull" {
		return bad("path is not owner/repo/issues/N")
	}
	if !validOwner(owner) || !validRepoName(name) {
		return bad("invalid owner or repo")
	}
	n, err := strconv.Atoi(num)
	if err != nil || n <= 0 || num[0] < '1' || num[0] > '9' {
		return bad("invalid number")
	}
	return owner + "/" + name, n, nil
}
