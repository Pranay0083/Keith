# Keith HTTP and WebSocket API

This is the contract between the web app (`web/`) and `keith api`. The API implements it exactly. The web app's REST types live in `web/src/lib/api/types.ts`, and the event types in `web/src/lib/api/events.gen.ts` (generated from `internal/events` by `make gen`).

Keywords: **must** and **never** are requirements. Anything marked *informative* describes client behaviour the API can rely on but does not enforce.

## Conventions

- **Base URLs.** The API is served from `PUBLIC_URL` (for example `https://api.keith.dev`). The web app is served from `WEB_ORIGIN` (for example `https://app.keith.dev`). The web app reads the API base from `NEXT_PUBLIC_API_URL`.
- **Same site.** `PUBLIC_URL` and `WEB_ORIGIN` must share a registrable domain (`app.example.com` + `api.example.com`). The session cookie is `SameSite=Lax` and host-only on the API host, so a cross-site API never receives it from `fetch`. In development `http://localhost:3000` and `http://localhost:8080` are the same site.
- **JSON.** Request and response bodies are `application/json; charset=utf-8`. A request with a body must send `Content-Type: application/json`, or the API answers `415`. Bodies over 1 MiB get `413`. Unknown request fields are ignored.
- **IDs** are UUID strings. **Timestamps** are RFC 3339 strings in UTC (`2026-11-02T10:15:03Z`). Nullable fields are always present and set to `null`, never omitted, unless marked optional (`?`).
- **Lists** that are not paginated are returned as a bare JSON array. Paginated lists are objects with a `next_cursor`.
- **Success statuses.** `200` with a body, `201` when a resource was created, `202` when work was accepted for the worker, `204` with no body.

### Errors

Every non-2xx response from `/api/*` and `/auth/logout` has this body:

```json
{ "error": { "code": "validation_failed", "message": "repo_ids: at least one repo is required", "fields": { "repo_ids": "at least one repo is required" } } }
```

`code` is a stable snake_case string the UI may switch on. `message` is a human-readable English sentence. `fields` is optional and only appears on `422`, keyed by request field name.

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Malformed JSON, a malformed path or query parameter |
| 401 | `unauthenticated` | No valid `keith_session` cookie: missing, expired, revoked by logout, or for a deleted user |
| 403 | `not_a_member` | The user is not a member of the org that owns the resource |
| 403 | `forbidden_role` | The user's role in the org is too low for this action (see each endpoint) |
| 403 | `bad_origin` | A state-changing request without `Origin == WEB_ORIGIN` |
| 404 | `not_found` | The resource does not exist, or the user may not know it exists |
| 409 | `not_waiting` | `POST /answers` while the session is not `waiting_for_user` |
| 409 | `question_mismatch` | `POST /answers` with a `question_id` that is not the open question |
| 409 | `answer_required` | `POST /messages` while the session is `waiting_for_user` |
| 409 | `not_running` | `POST /stop` on a session in a terminal status |
| 409 | `repo_access_lost` | `POST /messages` to a finished session, or `POST /answers`, when the org can no longer use one of the session's repos |
| 413 | `too_large` | Body over 1 MiB |
| 415 | `unsupported_media_type` | Body without `Content-Type: application/json` |
| 422 | `validation_failed` | A field failed validation; see `fields` |
| 422 | `provider_unavailable` | `provider` is not configured on this server |
| 429 | `rate_limited` | The org is at its active-session cap (`MAX_ACTIVE_SESSIONS_PER_ORG`): creating a session, or resuming one with `POST /messages` or `POST /answers`. `Retry-After` header in seconds |
| 500 | `internal` | Anything else. The message never leaks internals |

A resource in an org the user does not belong to returns `404` for session, repo and secret lookups by ID (so IDs cannot be probed), and `403 not_a_member` for `/api/orgs/{org_id}/…` routes.

## Authentication and CSRF

- **Session cookie.** `keith_session` is set by the OAuth callback with `HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=2592000` (30 days) and no `Domain` attribute. Its value is opaque to the web app and is signed with `SESSION_SECRET`. It carries the time it was issued, so logout can revoke it (see `POST /auth/logout`). When `PUBLIC_URL` is `http://localhost…` the API may omit `Secure` for local development.
- **Every `/api/*` route** requires the cookie and returns `401 unauthenticated` without it. The web app sends `credentials: "include"` on every request.
- **CSRF.** Every `POST`, `PUT`, `PATCH` and `DELETE` to `/api/*` and `/auth/logout` must carry an `Origin` header equal to `WEB_ORIGIN`, or the API answers `403 bad_origin` before doing anything else. `/hooks/github` is exempt (it is authenticated by HMAC). The WebSocket upgrade checks `Origin` too (see Events).
- **CORS.** For requests whose `Origin` is `WEB_ORIGIN`, the API sends `Access-Control-Allow-Origin: <WEB_ORIGIN>`, `Access-Control-Allow-Credentials: true` and `Vary: Origin`. A preflight `OPTIONS` answers `204` with `Access-Control-Allow-Methods: GET, POST, PUT, DELETE`, `Access-Control-Allow-Headers: Content-Type` and `Access-Control-Max-Age: 600`. Other origins get no CORS headers.
- **Roles** are `owner`, `admin` and `member`. Endpoints that need more than `member` say so.

### `GET /auth/github/login`

Starts GitHub sign-in. Not JSON: a browser navigation.

| Query | Required | Meaning |
|---|---|---|
| `next` | no | Path on `WEB_ORIGIN` to land on after sign-in. Must start with `/` and not with `//`; anything else is replaced by `/sessions`. |

Response: `302` to `https://github.com/login/oauth/authorize?client_id=<GITHUB_APP_CLIENT_ID>&redirect_uri=<PUBLIC_URL>/auth/github/callback&state=<state>`. `state` is at least 32 random bytes, base64url. The API also sets `keith_oauth_state` (`HttpOnly; Secure; SameSite=Lax; Path=/auth/github; Max-Age=600`) holding the state and `next`, signed.

### `GET /auth/github/callback`

GitHub redirects here with `code` and `state`.

1. Check `state` against `keith_oauth_state`, then clear that cookie.
2. Exchange `code` for a user token and fetch `GET /user`.
3. Upsert `users` by `github_user_id` (login, name, email, avatar).
4. On a user's first sign-in, create a personal org named after their login with them as `owner`. Every signed-in user therefore has at least one org.
5. Set `keith_session` and respond `302` to `WEB_ORIGIN + next` (default `/sessions`).

On failure the API responds `302` to `WEB_ORIGIN/login?error=<code>`, where `code` is `oauth_state` (missing or mismatched state), `oauth_denied` (the user cancelled on GitHub) or `oauth_failed` (anything else). The login page shows a message for each.

### `POST /auth/logout`

No body. Requires `Origin`. Revokes the session server-side, clears `keith_session` (`Max-Age=0`) and returns `204`. Returns `204` even without a valid cookie.

- Revocation records the time on the user (`users.sessions_valid_after`). Every cookie issued to that user at or before it is refused from then on, on every device: `/api/*` answers `401`, `GET /github/setup` redirects to login, and an open event stream closes with `4401` at its next access check (within 30 seconds).
- If the revocation cannot be saved, the API answers `500 internal` and keeps the cookie, so the client can retry.

### `GET /api/me`

```json
{
  "user": { "id": "7d0c…", "login": "octocat", "name": "The Octocat", "avatar_url": "https://avatars.githubusercontent.com/u/583231" },
  "orgs": [ { "id": "1b9e…", "name": "octocat", "role": "owner" } ]
}
```

`orgs` is sorted by name and never empty for a signed-in user. `name` may be `""`. The web app keeps the selected org in the browser and defaults to the first one.

### `GET /api/providers`

The LLM providers this server has keys for, so `/new` can offer a choice.

```json
{ "default_provider": "anthropic", "providers": [ { "id": "anthropic", "default_model": "claude-opus-5-5" }, { "id": "gemini", "default_model": "gemini-3-pro" } ] }
```

## GitHub

### `GET /api/orgs/{org_id}/github/installations`

The org's GitHub App installations. The connect page uses it to show what is linked.

```json
[ { "id": "e3f1…", "installation_id": 51234567, "account_login": "acme", "status": "active", "created_at": "2026-11-01T09:00:00Z" } ]
```

`status` is `active`, `suspended` or `deleted`.

### `GET /api/orgs/{org_id}/github/install-url`

Role: `owner` or `admin` (else `403 forbidden_role`).

```json
{ "url": "https://github.com/apps/keith-agent/installations/new?state=<setup_state>" }
```

`setup_state` is signed by the API and binds `{org_id, user_id, expires_at}` (15 minutes). GitHub passes it back to the setup URL unchanged. The web app navigates the browser to `url`.

### `GET /github/setup`

Backend redirect handler, the GitHub App's Setup URL. GitHub sends the browser here after an install or a repo-selection change.

| Query | Meaning |
|---|---|
| `installation_id` | GitHub's installation ID |
| `setup_action` | `install`, `update` or `request` |
| `code` | User OAuth code ("Request user authorization during installation" is on) |
| `state` | The `setup_state` from `install-url` |

1. Require a valid `keith_session`. Verify `state`'s signature and expiry, and that its `user_id` is the cookie's user and the user is still `owner` or `admin` of its `org_id`.
2. If `setup_action=request` (an org member asked a GitHub org owner to approve), stop and `302` to `WEB_ORIGIN/connect?pending=1`.
3. Prove the user owns the installation. Anyone can hit this URL with someone else's ID; these checks are what stop that.
   - Exchange `code` for a user token, and require the token's user to be the cookie user's `github_user_id`.
   - Call `GET /user/installations` with the token and require `installation_id` in the result.
   - Seeing an installation is not enough: GitHub lists one for anyone with read access to one of its repos. Fetch the installation (`GET /app/installations/{id}`) and require ownership of its account:
     - a `User` account must be the user itself (`account.id == github_user_id`);
     - for an `Organization` account, `GET /user/memberships/orgs/{org}` with the user token must answer `state: active`, `role: admin` (an owner of the GitHub org);
     - any other account type is refused.
   - A failed check is `installation_not_verified`; a GitHub error (including `403` when the App lacks the Members organization permission) is `setup_failed`.
4. Upsert `github_installations` for `org_id`. If that `installation_id` already belongs to another org, stop with `installation_other_org`.
5. List the installation's repos with an installation token and upsert `repos` (`accessible = true`; repos no longer listed become `accessible = false`).
6. `302` to `WEB_ORIGIN/repos?installed=1`.

On failure: `302` to `WEB_ORIGIN/connect?error=<code>` with `code` one of `setup_state`, `installation_not_verified`, `installation_other_org`, `setup_failed`. Without a session cookie: `302` to `WEB_ORIGIN/login?next=/connect`.

### `POST /hooks/github`

The GitHub App webhook endpoint. No cookie, no `Origin` check.

- Verify `X-Hub-Signature-256` (HMAC-SHA256 of the raw body with `GITHUB_WEBHOOK_SECRET`, constant-time compare) before parsing anything. Invalid or missing: `401`.
- `installation`: `created` upserts the installation if it is already linked to an org (otherwise the setup handler links it); `deleted` sets its status to `deleted`, marks its repos `accessible = false` and stops the org's live sessions on those repos; `suspend` / `unsuspend` set `suspended` / `active`.
- `installation_repositories`: `added` upserts repos, `removed` marks them `accessible = false`.
- Other events: ignored with `204`.
- Handled events return `202` quickly; work happens on the queue. Handling is idempotent on `X-GitHub-Delivery`.

## Repos

### `GET /api/orgs/{org_id}/repos`

Every repo the org's installations can see, sorted by `full_name`, including inaccessible ones (so their history still makes sense).

```json
[
  {
    "id": "5a2c…",
    "full_name": "acme/web",
    "default_branch": "main",
    "private": true,
    "accessible": true,
    "profile": { "install_cmd": "npm ci", "build_cmd": "npm run build", "test_cmd": "npm test", "run_cmd": "npm run dev", "knowledge": "Uses pnpm workspaces…", "updated_at": "2026-11-01T09:10:00Z" }
  },
  { "id": "9f00…", "full_name": "acme/api", "default_branch": "main", "private": true, "accessible": true, "profile": null }
]
```

`profile` is `null` until a profile is saved.

A repo belongs to the org whose installation last listed it. When a repo moves to another org's installation (for example, the App is uninstalled from one org and installed into another), its profile and snapshot are deleted, and the old org's sessions on it can no longer be resumed (`409 repo_access_lost`).

### `GET /api/repos/{repo_id}/profile`

`200` with the profile. A repo without a saved profile returns empty strings and `"updated_at": null`.

```json
{ "repo_id": "5a2c…", "install_cmd": "npm ci", "build_cmd": "npm run build", "test_cmd": "npm test", "run_cmd": "npm run dev", "knowledge": "Uses pnpm workspaces…", "updated_at": "2026-11-01T09:10:00Z" }
```

### `PUT /api/repos/{repo_id}/profile`

Replaces the whole profile. Any member.

```json
{ "install_cmd": "npm ci", "build_cmd": "npm run build", "test_cmd": "npm test", "run_cmd": "npm run dev", "knowledge": "Uses pnpm workspaces…" }
```

All five fields are strings and required (`""` clears one). Each `*_cmd` is at most 2,000 characters; `knowledge` at most 20,000. Response: `200` with the saved profile, same shape as `GET`. `snapshot_*` columns are never exposed. Saving a changed `install_cmd` invalidates the repo's snapshot (`snapshot_id = NULL`).

## Secrets

Values are write-only: no endpoint ever returns a value, and the API never logs one.

A secret is identified by `name` plus an optional `repo_id`: an org secret (`repo_id = null`) applies to every session; a repo secret applies when that repo is tagged, and wins over an org secret of the same name.

### `GET /api/orgs/{org_id}/secrets`

```json
[
  { "name": "STRIPE_TEST_KEY", "repo_id": null, "repo_full_name": null, "created_at": "2026-11-01T09:00:00Z", "updated_at": "2026-11-01T09:00:00Z" },
  { "name": "DATABASE_URL", "repo_id": "5a2c…", "repo_full_name": "acme/web", "created_at": "2026-11-01T09:00:00Z", "updated_at": "2026-11-02T12:00:00Z" }
]
```

Sorted by `name`, then org secrets before repo secrets.

### `PUT /api/orgs/{org_id}/secrets/{name}`

Role: `owner` or `admin`. Creates or replaces the secret.

```json
{ "value": "sk_test_…", "repo_id": null }
```

- `name` (path) must match `^[A-Za-z_][A-Za-z0-9_]*$` and be at most 128 characters.
- `value` must be a non-empty string of at most 65,536 bytes.
- `repo_id` is optional; `null` or absent means an org secret. It must be a repo of this org.

Response: `200` with the secret's metadata (the list item shape).

### `DELETE /api/orgs/{org_id}/secrets/{name}?repo_id={repo_id}`

Role: `owner` or `admin`. Omit `repo_id` to delete the org secret. `204`, or `404` if no such secret exists.

## Sessions

### Status

`status` is one of `queued`, `provisioning`, `setting_up`, `running`, `waiting_for_user`, `completed`, `failed`, `cancelled`, `no_changes` (`events.SessionStatus`). The last four are terminal.

### The session object

```json
{
  "id": "3c4b…",
  "org_id": "1b9e…",
  "status": "running",
  "prompt": "Fix the flaky date test in utils",
  "issue_url": "https://github.com/acme/web/issues/42",
  "provider": "anthropic",
  "model": "claude-opus-5-5",
  "created_by": { "id": "7d0c…", "login": "octocat", "avatar_url": "https://avatars.githubusercontent.com/u/583231" },
  "cancel_requested": false,
  "created_at": "2026-11-02T10:15:00Z",
  "finished_at": null,
  "repos": [
    { "repo_id": "5a2c…", "full_name": "acme/web", "branch": "agent/3c4b…", "pr_number": null, "pr_url": null }
  ]
}
```

- `issue_url` is `null` when none was given. It is stored in `sessions.trigger` as `{"kind":"web","issue_url":…}`, so no schema change is needed.
- `created_by` is `null` for sessions no person started.
- `cancel_requested` is `sessions.cancel_requested_at IS NOT NULL`.
- `repos` keeps the order of `repo_ids` at creation.

### `POST /api/orgs/{org_id}/sessions`

Any member.

```json
{ "prompt": "Fix the flaky date test", "repo_ids": ["5a2c…"], "issue_url": "https://github.com/acme/web/issues/42", "provider": "anthropic", "model": "claude-opus-5-5" }
```

| Field | Rule |
|---|---|
| `prompt` | String, at most 32,000 characters. May be `""` only if `issue_url` is set. |
| `repo_ids` | 1 to 10 distinct IDs of `accessible` repos in this org. |
| `issue_url` | Optional. `https://github.com/{owner}/{repo}/issues/{number}`, and `{owner}/{repo}` must be one of the tagged repos (code `issue_repo_not_tagged` in `fields.issue_url`). |
| `provider` | Optional. `anthropic` or `gemini`; default is the org's `default_llm_provider`, then the server default. Unconfigured: `422 provider_unavailable`. |
| `model` | Optional, at most 100 characters; default is the provider's default model. |

Active sessions (`queued`, `provisioning`, `setting_up`, `running`) are capped per org at `MAX_ACTIVE_SESSIONS_PER_ORG`. At the cap: `429 rate_limited` with `Retry-After`.

The API inserts `sessions` (`queued`) and `session_repos` (`branch = agent/<session_id>`), enqueues `run_session`, and returns `201`:

```json
{ "id": "3c4b…" }
```

### `GET /api/orgs/{org_id}/sessions`

| Query | Meaning |
|---|---|
| `cursor` | Opaque; from a previous `next_cursor`. |
| `limit` | 1 to 100, default 25. |

Newest first (`created_at DESC, id DESC`).

```json
{ "sessions": [ { "id": "3c4b…", "org_id": "1b9e…", "status": "running", "…": "the session object" } ], "next_cursor": "eyJ0IjoiMjAyNi0xMS0wMlQxMDoxNTowMFoiLCJpZCI6IjNjNGIifQ" }
```

`next_cursor` is `null` on the last page.

### `GET /api/sessions/{id}`

`200` with the session object.

### `POST /api/sessions/{id}/messages`

Steer a running session, or send a follow-up to a finished one. Any member.

```json
{ "text": "Use dayjs instead of moment" }
```

`text` must be 1 to 16,000 characters after trimming. Response `202`:

```json
{ "message_id": "m_01J…" }
```

- Running statuses (`queued` … `running`): the message is queued and the worker adds it to the next model turn. It is never dropped.
- Terminal statuses: the message is queued and the API enqueues `resume_session`.
  - A resumed session is active again, so it counts against the org's cap: at the cap, `429 rate_limited` and nothing is stored.
  - If the org can no longer use one of the session's repos, the API answers `409 repo_access_lost` and stores nothing. That happens when the repo was removed from the installation, the App was uninstalled, or the repo now belongs to another org's installation. Start a new session instead.
- `waiting_for_user`: `409 answer_required`. Use `/answers`.

The API does not write an event. The worker emits `message.user` with this `message_id` when it hands the message to the model, so the UI shows the message as pending until then.

### `POST /api/sessions/{id}/answers`

Answer the open `question.asked`. Any member.

```json
{ "question_id": "toolu_01…", "text": "Use the sandbox key in STRIPE_TEST_KEY" }
```

`text` must be 1 to 16,000 characters after trimming. When `question.asked` offered `choices`, `text` may be one of them or free text.

- Status not `waiting_for_user`: `409 not_waiting`.
- `question_id` is not the open question (the latest `question.asked` with no answer): `409 question_mismatch`.
- The org can no longer use one of the session's repos (as for `/messages`): `409 repo_access_lost`. The question stays open.
- The org is at its active-session cap (a `waiting_for_user` session is not active; answering makes it active): `429 rate_limited`. The question stays open.

Otherwise the API stores the answer, enqueues `resume_session` and returns `202 { "message_id": "m_01J…" }`. The resuming worker emits `message.user` with this `message_id` and the `question_id`, then `session.status` `running`.

### `POST /api/sessions/{id}/stop`

No body. Any member. Sets `sessions.cancel_requested_at` (idempotent) and returns `202 {}`.

- The worker checks the flag between steps, interrupts the running shell command, and emits `session.status` `cancelled`.
- If no worker holds the session (`queued` before pickup, or `waiting_for_user`), the API enqueues `resume_session` so a worker records the cancellation.
- Terminal status: `409 not_running`.

### `GET /api/sessions/{id}/diff`

The latest unified diff the worker saved for each repo, against the repo's default branch at clone time. The worker saves it after every file-changing tool call, before emitting `file.changed`.

```json
{
  "repos": [
    { "full_name": "acme/web", "diff": "diff --git a/src/date.ts b/src/date.ts\n…", "truncated": false, "updated_at": "2026-11-02T10:20:11Z" }
  ]
}
```

- One entry per session repo, in session order. `diff` is `""` and `updated_at` is `null` before any change.
- `diff` is at most 1 MiB; `truncated` says whether it was cut. It never contains secret values (the worker redacts them).
- Storage is the backend's choice. Suggested: `session_repos.last_diff text` and `last_diff_at timestamptz` in a new migration.

## Events

### `GET /api/sessions/{id}/events?after_seq={n}`

A WebSocket upgrade. The browser opens `wss://<api host>/api/sessions/{id}/events?after_seq=N` (`ws://` for `http://`).

**Before the upgrade** the API checks, answering with a plain HTTP error and JSON body:
- `keith_session`: `401`.
- `Origin == WEB_ORIGIN`: `403 bad_origin`.
- Membership of the session's org: `404`.
- `after_seq` is an integer ≥ 0 (default `0`): `400`.

A browser sees any of these only as a failed connection, so the client probes `GET /api/sessions/{id}` to find out why (*informative*, see Client behaviour).

**After the upgrade** the server sends text frames, one JSON `Event` per frame:

```json
{ "session_id": "3c4b…", "seq": 42, "ts": "2026-11-02T10:15:03Z", "type": "tool_call.started", "payload": { "call_id": "toolu_9", "tool": "shell_exec", "input": { "shell_id": "dev", "command": "npm test" } } }
```

1. The server subscribes to the session's live channel, then sends every stored event with `seq > N` in `seq` order, then forwards live events. It never sends a stored `seq` twice on one connection, and never sends a stored `seq ≤ N`.
2. `message.delta` events are live-only and arrive with `seq` 0. They are never stored or replayed.
3. Stored events on one connection arrive in strictly increasing `seq` order. If the server's buffer for a slow client overflows, it closes with `4000` instead of skipping events.
4. The server sends a WebSocket ping control frame every 25 seconds and closes the connection if no pong arrives within 60 seconds. Browsers answer pings automatically.
5. The connection stays open after the session reaches a terminal status, because a follow-up message can resume it.
6. The client sends nothing. The server ignores any frames it receives.

**Close codes**

| Code | Meaning | Client does |
|---|---|---|
| 1000, 1001, 1011, 1012, 4000 | Normal close, server going away, server error, restart, resync | Reconnect with backoff and `after_seq` |
| 4401 | Session cookie expired or revoked | Go to `/login?next=…`; do not reconnect |
| 4403 | The user lost access to the org | Show an error; do not reconnect |
| 4404 | The session was deleted | Show an error; do not reconnect |

### Event types

The payload structs are defined in `internal/events/events.go`. The TypeScript in `web/src/lib/api/events.gen.ts` is generated from them.

| `type` | Payload | Stored |
|---|---|---|
| `session.status` | `{status, reason?}` | yes |
| `message.user` | `{message_id, user_id?, text, question_id?}` | yes |
| `message.assistant` | `{message_id, text}` | yes |
| `message.delta` | `{message_id, text}`, a fragment to append | no (`seq` 0) |
| `tool_call.started` | `{call_id, tool, input}` | yes |
| `tool_call.output` | `{call_id, shell_id?, chunk}`, raw terminal bytes with ANSI escapes | yes |
| `tool_call.finished` | `{call_id, tool, is_error, exit_code?, timed_out?, duration_ms, summary?}` | yes |
| `shell.created` | `{shell_id, call_id?}` | yes |
| `file.changed` | `{call_id?, repo?, path, action}`, where `action` is `created`, `modified` or `deleted` | yes |
| `question.asked` | `{question_id, question, choices?}` | yes |
| `pr.opened` | `{repo, number, url, title, branch}` | yes |
| `error` | `{code?, message, retryable?}` | yes |

### What the web app relies on

The worker emits events so that these hold:

- A new session's first stored event is `message.user` carrying the prompt (or, for an issue-only task, the text the agent was given), followed by `session.status` `provisioning`.
- Every `message_id` returned by `POST /messages` or `POST /answers` appears later in exactly one `message.user`. Answers also carry `question_id`.
- All `message.delta` fragments for a message precede its `message.assistant` with the same `message_id`. The UI replaces the streamed text with the final text and ignores later deltas for that ID.
- `shell.created` for a shell precedes the first `tool_call.output` with that `shell_id`. Shell output always sets `shell_id`.
- Each `tool_call.started` is followed by exactly one `tool_call.finished` with the same `call_id`, unless the session fails or is cancelled first.
- `file.changed` is emitted only after `GET /diff` returns the new diff.
- `question.asked` is followed by `session.status` `waiting_for_user`.
- `pr.opened` is emitted once per repo, after `session_repos.pr_number` and `pr_url` are set.
- Secret values never appear in any payload.

The worker's own steps look like agent steps, so the UI needs nothing special for them:

- Setup and verification run as tool calls named `setup` (a repo's install command) and `run_tests` (its test command), in shells named `setup` and `verify`; their `call_id`s start with `keith_`. `input` is `{repo, command}`.
- A `message.user` whose `message_id` starts with `m_keith` comes from the worker, not from `POST /messages`: the session's prompt (with the creator's `user_id`), or a note to the agent such as a failed automatic test run (no `user_id`).
- The worker opens the pull request itself (`pr.opened`, no surrounding tool call) when the agent ended its turn with changes but no PR.

### Client behaviour (*informative*)

- The client remembers the highest stored `seq` it applied and reconnects with `after_seq` set to it.
- It drops any stored event with `seq ≤` that value, and applies stored events strictly in `seq` order, buffering any that arrive ahead of a gap. If a gap persists for 3 seconds it reconnects to refill it.
- A stored event whose `type` it doesn't know (a newer server), or whose `payload` isn't an object, still takes its place in `seq` order: the client skips it and moves on, rather than waiting for that `seq` forever. It ignores an unknown live-only event.
- Reconnect delay is `min(15 s, 0.5 s × 2^attempt)`, scaled by a random factor in [0.5, 1). The attempt counter resets once a connection has stayed open for 5 seconds.
- After a failed connection the client calls `GET /api/sessions/{id}`: `401` sends it to the login page, `403`/`404` stops reconnecting, anything else keeps retrying.

## Other

### `GET /healthz`

No auth. `200 {"ok": true}` when the API can reach Postgres and Redis, else `503 {"ok": false}`.
