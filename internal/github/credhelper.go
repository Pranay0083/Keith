package github

import "strings"

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
// The token is readable by anything in the sandbox; that is by design
// (contents: write on the session's repos, 1 hour, revoked at the end).
func CredentialHelperScript() string {
	return credentialHelperScript(GitTokenPath, "github.com")
}

func credentialHelperScript(tokenPath, host string) string {
	return strings.NewReplacer("@TOKEN_PATH@", shellQuote(tokenPath), "@HOST@", shellQuote(host)).Replace(credentialHelperTemplate)
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

const credentialHelperTemplate = `#!/bin/sh
# git-credential-keith: git credential helper for Keith sandboxes.
# Answers "get" for https on one host with the installation token in a
# file the worker rewrites to rotate it. Nothing is cached.
token_file=@TOKEN_PATH@
allowed_host=@HOST@

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

[ "$protocol" = https ] || exit 0
[ "$host" = "$allowed_host" ] || exit 0
[ -r "$token_file" ] || exit 0

token=
IFS= read -r token < "$token_file" || :
case $token in
'' | *[!A-Za-z0-9_]*) exit 0 ;;
esac

printf 'username=x-access-token\npassword=%s\n' "$token"
`
