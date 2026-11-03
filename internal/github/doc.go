// Package github is Keith's GitHub App integration: App authentication,
// repo-scoped installation tokens, user OAuth, installation webhooks, and
// Host, the agent.RepoHost implementation the read_issue and
// create_pull_request tools run through.
//
// Package boundaries: github imports agent only, for the RepoHost types.
// agent never imports github; the worker hands it a *Host.
//
// # Which token goes where
//
// The App private key and the OAuth client secret stay in the control
// plane (api and worker). Nothing derived from them except a short-lived
// installation token reaches a sandbox.
//
//   - Sandbox token: App.InstallationToken (or a TokenSource) with
//     SandboxPermissions (contents: read only), scoped to the session's
//     repos. It reaches the sandbox only as the file read by the git
//     credential helper (see CredentialHelperScript). It can clone and
//     fetch, and nothing else: it cannot push, read issues or open PRs.
//   - Push token: PushPermissions (contents: write only), minted by the
//     worker for one repo per push and revoked right after. It never
//     enters the sandbox: the worker sends the session branch, and only
//     that ref, itself (ReceivePack, ParseBundle).
//   - Worker token: a separate TokenSource with WorkerPermissions
//     (contents: read, pull_requests: write, issues: read, metadata: read),
//     also scoped to the session's repos. It backs Host and never enters
//     the sandbox, so a token leaked from the sandbox cannot touch issues
//     or pull requests.
//
// InstallationToken refuses an empty repo list or empty permissions, so
// this package never mints a token for every repo of an installation
// with every permission the installation has. The one exception is
// internal: ListInstallationRepos mints a metadata: read token for the
// whole installation, uses it in-process and revokes it.
//
// Revoke the sandbox and worker tokens when the session ends
// (TokenSource.Revoke).
//
// # Setup URL and OAuth
//
// Anyone can call the setup URL with someone else's installation_id. The
// API must exchange the OAuth code GitHub sends with the install
// (Exchange) and check UserCanAccessInstallation before linking the
// installation to an org. Start the install flow at InstallURL with a
// fresh state value and check it on the way back, as for sign-in.
package github
