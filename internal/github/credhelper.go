package github

import (
	"fmt"
	"net/url"
	"strings"
)

// Paths inside the sandbox for the git credential helper.
const (
	// GitTokenPath holds the sandbox installation token, one line.
	GitTokenPath = "/run/keith/git-token"
	// CredentialHelperPath is where the worker installs
	// CredentialHelperScript.
	CredentialHelperPath = "/usr/local/bin/git-credential-keith"
)

// CredentialHelperScript returns a POSIX sh git credential helper that
// answers "get" for https://github.com with username x-access-token and
// the token read from GitTokenPath at that moment. It caches nothing, so
// rewriting the file rotates the token without re-cloning. It answers
// nothing for any other host or protocol, so a remote pointed elsewhere
// never receives the token, and it ignores store and erase.
//
// The worker installs it once per sandbox, before cloning:
//
//  1. Write the script to CredentialHelperPath, mode 0755, owned by root.
//  2. As the sandbox user, point git at it for github.com only, clearing
//     any helper the base image set:
//     git config --global credential.https://github.com.helper ""
//     git config --global --add credential.https://github.com.helper /usr/local/bin/git-credential-keith
//  3. Create /run/keith (mode 0700) and write the sandbox token to
//     GitTokenPath (mode 0600), both owned by the sandbox user. Write to
//     a temporary file in /run/keith and rename it over GitTokenPath, so
//     git never reads a partial token.
//  4. Clone https://github.com/<owner>/<repo>.git. Never put the token in
//     a URL: it would land in .git/config and in process listings.
//  5. Rewrite the file the same way from the sandbox TokenSource's
//     OnRefresh hook, and keep that source fresh with Run.
//  6. When the session ends, Revoke the TokenSource and delete the file.
//     A sandbox resumed after its token expired needs a new TokenSource
//     and a fresh file before git runs.
//
// The token is readable by anything in the sandbox; that is by design:
// it is contents: read on the session's repos (SandboxPermissions), for
// 1 hour, revoked at the end. It cannot push; the worker pushes the
// session branch from outside the sandbox.
func CredentialHelperScript() string {
	return credentialHelperScript(GitTokenPath, "github.com")
}

// GitRemote is where a sandbox clones from: a base URL such as
// https://github.com/ (the default) or, for local development and tests,
// a git server such as http://host.docker.internal:8081/git/.
type GitRemote struct {
	// Base is the parsed base URL, with a trailing slash.
	Base *url.URL
}

// ParseGitRemote checks base: http or https, a host, no user info, query
// or fragment.
func ParseGitRemote(base string) (GitRemote, error) {
	u, err := url.Parse(base)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.User != nil ||
		u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" {
		return GitRemote{}, fmt.Errorf("github: git base URL %q is not an http(s) base URL", base)
	}
	if !strings.HasSuffix(u.Path, "/") {
		u.Path += "/"
	}
	u.RawPath = ""
	return GitRemote{Base: u}, nil
}

// CloneURL returns the plain clone URL of owner/name: no token, ever.
func (r GitRemote) CloneURL(fullName string) (string, error) {
	if _, _, err := splitRepo(fullName); err != nil {
		return "", err
	}
	return r.Base.String() + fullName + ".git", nil
}

// CredentialKey is the URL prefix git matches the credential helper
// against: scheme://host[:port] (git config credential.<key>.helper).
func (r GitRemote) CredentialKey() string {
	return r.Base.Scheme + "://" + r.Base.Host
}

// CredentialHelperScript returns the helper for this remote: like the
// package-level CredentialHelperScript, but answering only for the
// remote's scheme and host (host:port when a port is set, as git sends
// it).
func (r GitRemote) CredentialHelperScript() string {
	return strings.NewReplacer("@TOKEN_PATH@", shellQuote(GitTokenPath), "@HOST@", shellQuote(r.Base.Host),
		"@PROTOCOL@", shellQuote(r.Base.Scheme)).Replace(credentialHelperTemplate)
}

func credentialHelperScript(tokenPath, host string) string {
	return strings.NewReplacer("@TOKEN_PATH@", shellQuote(tokenPath), "@HOST@", shellQuote(host),
		"@PROTOCOL@", shellQuote("https")).Replace(credentialHelperTemplate)
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

const credentialHelperTemplate = `#!/bin/sh
# git-credential-keith: git credential helper for Keith sandboxes.
# Answers "get" for one protocol and host with the installation token in a
# file the worker rewrites to rotate it. Nothing is cached.
token_file=@TOKEN_PATH@
allowed_host=@HOST@
allowed_protocol=@PROTOCOL@

[ "$1" = get ] || exit 0

protocol=
host=
while IFS= read -r line; do
	case $line in
	'') break ;;
	protocol=*) protocol=${line#protocol=} ;;
	host=*) host=${line#host=} ;;
	esac
done

[ "$protocol" = "$allowed_protocol" ] || exit 0
[ "$host" = "$allowed_host" ] || exit 0
[ -r "$token_file" ] || exit 0

token=
IFS= read -r token < "$token_file" || :
case $token in
'' | *[!A-Za-z0-9_]*) exit 0 ;;
esac

printf 'username=x-access-token\npassword=%s\n' "$token"
`
