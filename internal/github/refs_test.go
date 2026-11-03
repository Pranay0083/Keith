package github

import (
	"strings"
	"testing"
)

func TestParseIssueURL(t *testing.T) {
	good := []struct {
		url  string
		repo string
		num  int
	}{
		{"https://github.com/acme/api/issues/123", "acme/api", 123},
		{"https://github.com/acme/api/issues/123/", "acme/api", 123},
		{"  https://github.com/acme/api/issues/9\n", "acme/api", 9},
		{"https://github.com/acme/api/issues/5#issuecomment-77", "acme/api", 5},
		{"https://github.com/acme/api/issues/5?foo=bar", "acme/api", 5},
		{"https://www.github.com/Acme-Co/my.repo_x/issues/1", "Acme-Co/my.repo_x", 1},
		{"https://GitHub.com/acme/api/issues/2", "acme/api", 2},
		{"https://github.com/acme/api/pull/45", "acme/api", 45},
	}
	for _, c := range good {
		repo, num, err := ParseIssueURL(c.url)
		if err != nil || repo != c.repo || num != c.num {
			t.Errorf("ParseIssueURL(%q) = %q, %d, %v; want %q, %d", c.url, repo, num, err, c.repo, c.num)
		}
	}

	bad := []string{
		"",
		"github.com/acme/api/issues/1",
		"http://github.com/acme/api/issues/1",
		"https://gitlab.com/acme/api/issues/1",
		"https://github.com.evil.com/acme/api/issues/1",
		"https://evil.com/github.com/acme/api/issues/1",
		"https://github.com:8443/acme/api/issues/1",
		"https://user@github.com/acme/api/issues/1",
		"https://api.github.com/repos/acme/api/issues/1",
		"https://github.com/acme/api",
		"https://github.com/acme/api/issues",
		"https://github.com/acme/api/issues/",
		"https://github.com/acme/api/issues/0",
		"https://github.com/acme/api/issues/-1",
		"https://github.com/acme/api/issues/+1",
		"https://github.com/acme/api/issues/01",
		"https://github.com/acme/api/issues/1x",
		"https://github.com/acme/api/issues/99999999999999999999",
		"https://github.com/acme/api/discussions/1",
		"https://github.com/acme/api/pull/1/files",
		"https://github.com/acme/api/issues/1/extra",
		"https://github.com/acme//issues/1",
		"https://github.com/acme/../issues/1",
		"https://github.com/ac%2Fme/api/issues/1",
		"https://github.com//acme/api/issues/1",
		"https://github.com/acme/api/issues/1//",
		"https://github.com/-acme/api/issues/1",
	}
	for _, u := range bad {
		if repo, num, err := ParseIssueURL(u); err == nil {
			t.Errorf("ParseIssueURL(%q) = %q, %d; want an error", u, repo, num)
		}
	}
}

func TestValidateAgentBranch(t *testing.T) {
	for _, ok := range []string{
		"agent/s1",
		"agent/0b9d5c3e-8f3a-4c6e-9a51-2f7f1d0e6b42",
		"agent/fix/login_bug-2",
		"agent/v1.2",
	} {
		if err := ValidateAgentBranch(ok); err != nil {
			t.Errorf("ValidateAgentBranch(%q) = %v", ok, err)
		}
	}
	for _, bad := range []string{
		"", "main", "master", "agent", "agent/", "Agent/x", "agents/x", "x/agent/y", "refs/heads/agent/x",
		"agent//x", "agent/x/", "agent/.hidden", "agent/x/.y", "agent/a..b", "agent/x.", "agent/x.lock",
		"agent/x.lock/y", "agent/x y", "agent/x~1", "agent/x^", "agent/x:y", "agent/x?", "agent/x*",
		"agent/x[", "agent/x\\y", "agent/x@{1}", "agent/\x00", "agent/é",
		"agent/" + strings.Repeat("a", 250),
	} {
		if err := ValidateAgentBranch(bad); err == nil {
			t.Errorf("ValidateAgentBranch(%q) accepted", bad)
		}
	}
}
