# Setting up Keith

This guide is for operators. It covers the accounts and keys Keith needs, creating the GitHub App, every configuration variable, local development, tests and deployment. For how the system works, see [docs/architecture.md](docs/architecture.md); for the HTTP API, [docs/api.md](docs/api.md).

## Contents

- [Overview](#overview)
- [Accounts and keys](#accounts-and-keys)
- [Create the GitHub App](#create-the-github-app)
- [Configuration](#configuration)
- [Local development](#local-development)
- [Run one task with no App: `keith run --local`](#run-one-task-with-no-app-keith-run---local)
- [Tests](#tests)
- [Sandbox benchmark: `keith sandbox-bench`](#sandbox-benchmark-keith-sandbox-bench)
- [Deploying](#deploying)
- [Troubleshooting](#troubleshooting)

## Overview

| Service | Run with | Needs |
|---|---|---|
| **api** | `keith api` | Postgres, Redis, the GitHub App (ID, private key, OAuth client, webhook secret), `SECRETS_KEY`, `SESSION_SECRET`, its own public URL and the web app's origin. Serves HTTP and WebSocket on `LISTEN_ADDR` (default `:8080`). |
| **worker** | `keith worker` | Postgres, Redis, the GitHub App (ID and private key), `SECRETS_KEY`, an LLM key, and a sandbox provider: Docker locally, Daytona in production. No inbound traffic. |
| **web** | `web/` (Next.js) | `NEXT_PUBLIC_API_URL`, the api's URL. |

`keith api` and `keith worker` are the same binary and the same container image (the root `Dockerfile`); the subcommand picks the role. Both apply the database migrations at start, behind a Postgres advisory lock, so several replicas can start at once.

The worker runs the agent loop and drives one sandbox per session. Sandboxes start from the image built by `sandbox/base.Dockerfile` (or a per-repo snapshot of it). LLM keys, the App private key and the OAuth client secret never enter a sandbox.

## Accounts and keys

Before you start, create:

1. **A GitHub App**, ideally two: one for local development and one for production, because an App has a single webhook URL. See [Create the GitHub App](#create-the-github-app).
2. **An Anthropic API key** (console.anthropic.com), for the default provider and model, `claude-opus-5-5`. Optionally, or instead, **a Gemini API key** (Google AI Studio). Each session stays on the provider it started with.
3. **For production:**
   - a **Daytona account and API key** (app.daytona.io), for sandboxes;
   - hosting for the api and worker (Fly.io or Railway);
   - Vercel for the web app;
   - managed Postgres and Redis;
   - **a domain** for the api and web app (see [Cookies and domains](#cookies-and-domains)).

To run one task with only Docker, an LLM key and a GitHub personal access token, see [`keith run --local`](#run-one-task-with-no-app-keith-run---local).

## Create the GitHub App

The App does two things:

- Its **OAuth client** signs users in to Keith and authorizes them during installation.
- Its **installations** give Keith short-lived, repo-scoped tokens for cloning, pushing and opening PRs.

Below, `<api>` is the api's public base URL, the value of `PUBLIC_URL`: for example `https://api.example.com`, or `http://localhost:8080` for a local development App.

1. Go to GitHub → **Settings** → **Developer settings** → **GitHub Apps** → **New GitHub App**. For an App owned by a GitHub organization, use the organization's **Settings** → **Developer settings** instead.

2. **Basic information**
   - **GitHub App name**: for example `Keith` or `Keith (dev)`. It must be unique on GitHub. Its URL form becomes the App's slug (`GITHUB_APP_SLUG`).
   - **Homepage URL**: the web app's URL (`WEB_ORIGIN`).

3. **Identifying and authorizing users**
   - **Callback URL**: `<api>/auth/github/callback`.
     - Sign-in sends exactly this URL as `redirect_uri`, so it must match `PUBLIC_URL`.
     - If you add more than one callback URL, put this one first: the authorization during installation returns to the App's own (first) callback URL.
   - **Expire user authorization tokens**: leave it as is. Keith uses the user token only once, at sign-in or install, and does not keep it.
   - **Request user authorization (OAuth) during installation**: **check it.**
     - The api needs the OAuth code that comes back with an install to prove that the person installing can see the installation. Without it, linking fails with `installation_not_verified`.
     - With it checked, GitHub returns users to the Callback URL after an install, not to a Setup URL. The api recognizes those redirects (they carry `installation_id` and `setup_action`) and handles them as the setup step.
   - **Enable Device Flow**: leave it unchecked.

4. **Post installation**
   - **Setup URL**: `<api>/github/setup`, if GitHub lets you set it. With the OAuth box above checked, GitHub usually disables this field, and the Callback URL is used instead.
   - **Redirect on update**: check it if it is available.

5. **Webhook**
   - **Active**: checked.
   - **Webhook URL**: `<api>/hooks/github`. It must be reachable from the internet; for local development see [Webhooks on localhost](#webhooks-on-localhost).
   - **Webhook secret**: a random value (`openssl rand -hex 32`). Put the same value in `GITHUB_WEBHOOK_SECRET`.
   - **SSL verification**: enabled.

   Webhooks keep Keith in sync after an install: uninstalls, suspensions, and repos added to or removed from the installation. If you untick Active on a development App, sign-in, install and sessions still work, but Keith misses those changes.

6. **Permissions**
   - **Repository permissions**:

     | Permission | Access | Why |
     |---|---|---|
     | Contents | Read and write | Clone, and push `agent/*` branches |
     | Pull requests | Read and write | Open the PR |
     | Issues | Read-only | `read_issue`, for sessions started from an issue link |
     | Metadata | Read-only | Required by GitHub; selected automatically |

   - **Organization permissions**: none.
   - **Account permissions**: none. Sign-in reads only the user's public profile email. Email addresses (read-only) is optional and unused today.

7. **Subscribe to events**: none needed. GitHub delivers `installation` and `installation_repositories` events to every App with an active webhook; there is no checkbox for them. The api answers every other event with `204` and ignores it.

8. **Where can this GitHub App be installed?**
   - **Only on this account**: for a development App, or a single-team deployment.
   - **Any account**: so other users and organizations can install it. A multi-tenant deployment needs this.

9. Click **Create GitHub App**. Then, on the App's **General** page:
   - **App ID**: put it in `GITHUB_APP_ID`.
   - **Client ID**: put it in `GITHUB_APP_CLIENT_ID`.
   - **Client secrets** → **Generate a new client secret**: put it in `GITHUB_APP_CLIENT_SECRET`. GitHub shows it once.
   - **Private keys** → **Generate a private key**: this downloads a `.pem`.
     - Store it outside the repo (never commit it).
     - Give it to the api and worker as `GITHUB_APP_PRIVATE_KEY_PATH` or `GITHUB_APP_PRIVATE_KEY`.
     - It never goes into a sandbox, and it must not be readable by one.
   - **Slug**: the last path segment of the App's public page, `https://github.com/apps/<slug>`. Optionally put it in `GITHUB_APP_SLUG`; otherwise the api looks it up once at runtime.

10. **Install the App from Keith, not from GitHub.** Sign in to the web app, open **Connect**, and use its install button.
    - That link carries a signed `state` that ties the installation to your Keith org.
    - An install started from GitHub's own App page has no `state`, so Keith refuses to link it (`setup_state`).
    - The state expires after 15 minutes.

### Webhooks on localhost

GitHub's servers deliver webhooks, so a local api needs a public tunnel to receive them. Sign-in and install are browser redirects and work on `http://localhost` without one.

```bash
cloudflared tunnel --url http://localhost:8080      # prints https://<random>.trycloudflare.com
# GitHub App → Webhook URL: https://<random>.trycloudflare.com/hooks/github
```

- `ngrok http 8080` works the same way.
- A quick tunnel's URL changes on every run, so update the App's webhook URL each time, or use a named tunnel or a reserved domain.
- smee.io (`npx smee-client --url https://smee.io/<channel> --target http://localhost:8080/hooks/github`) also works. If deliveries start failing with `401`, the relay changed the body, which breaks the signature check; use a tunnel instead.

Use the tunnel for the webhook only. Keep `PUBLIC_URL`, the Callback URL and `NEXT_PUBLIC_API_URL` on `http://localhost:8080`. The sign-in state cookie is set on the host the web app links to (`NEXT_PUBLIC_API_URL`), and GitHub must return the browser to that same host. The session cookie must also be same-site with the web app (see [Cookies and domains](#cookies-and-domains)).

App → **Advanced** → **Recent Deliveries** shows every delivery and its response, and can redeliver one.

## Configuration

`keith api` and `keith worker` read only environment variables (`internal/config`). They do not read `.env` files themselves:

- `docker compose` loads `.env` for the containers.
- For processes on the host, export the file first: `set -a; . ./.env; set +a`.

`keith api` and `keith worker` validate their configuration at start and name every missing or malformed variable. `Config.String` redacts the secrets in the startup log.

### Generate the secrets

```bash
openssl rand -base64 32   # SECRETS_KEY: standard base64 of exactly 32 random bytes
openssl rand -hex 32      # SESSION_SECRET (at least 32 characters)
openssl rand -hex 32      # GITHUB_WEBHOOK_SECRET (same value as in the GitHub App)
```

Keep `SECRETS_KEY` stable and backed up. It encrypts the org and repo secrets stored in Postgres, and if it changes, those secrets can no longer be decrypted. Rotating `SESSION_SECRET` signs every user out.

### api and worker

| Variable | Used by | Required | Default | Description |
|---|---|---|---|---|
| `DATABASE_URL` | both | yes | (none) | Postgres connection string, for example `postgres://keith:keith@localhost:5432/keith?sslmode=disable`. |
| `REDIS_URL` | both | yes | (none) | Redis URL: `redis://` or `rediss://` (TLS). It carries the job queue, event pub/sub, leases and control signals. |
| `SECRETS_KEY` | both | yes | (none) | Standard base64 of 32 random bytes. The key-encryption key for org and repo secrets: the api encrypts them, and the worker decrypts them into the sandbox. |
| `GITHUB_APP_ID` | both | yes | (none) | The App's numeric ID. |
| `GITHUB_APP_PRIVATE_KEY` | both | one of these two | (none) | The App's PEM private key itself. Literal `\n` sequences become newlines, so it can sit on one line. |
| `GITHUB_APP_PRIVATE_KEY_PATH` | both | one of these two | (none) | Path to the `.pem` file. Set only one of the two. |
| `GITHUB_APP_CLIENT_ID` | api | yes | (none) | The App's OAuth client ID, for sign-in and install-time authorization. |
| `GITHUB_APP_CLIENT_SECRET` | api | yes | (none) | The App's OAuth client secret. |
| `GITHUB_WEBHOOK_SECRET` | api | yes | (none) | The App's webhook secret. The api checks `X-Hub-Signature-256` with it. |
| `GITHUB_APP_SLUG` | api | no | looked up once | The App's URL name (`github.com/apps/<slug>`), for the install link. |
| `PUBLIC_URL` | api | yes | (none) | The api's external base URL, with no trailing slash. Sign-in sends `PUBLIC_URL/auth/github/callback` as `redirect_uri`. Must be on the same host as `NEXT_PUBLIC_API_URL`. An `http://` URL drops the cookies' `Secure` flag, which is for local development only. |
| `WEB_ORIGIN` | api | yes | (none) | The web app's origin, exactly `scheme://host[:port]` with no trailing slash. Used for CORS, the CSRF `Origin` check, the WebSocket origin check and redirects back to the web app. |
| `SESSION_SECRET` | api | yes | (none) | Signs the `keith_session` cookie and the OAuth and setup state. At least 32 characters. |
| `LISTEN_ADDR` | api | no | `:8080` | The api's listen address. |
| `MAX_ACTIVE_SESSIONS_PER_ORG` | api | no | `5` | Caps an org's queued and running sessions. Beyond it, creating a session answers `429`. |
| `ANTHROPIC_API_KEY` | both | worker: yes if the default provider is `anthropic` | (none) | Claude API key. The worker calls the model with it. The api only checks that it is set: it accepts `anthropic` sessions and lists the provider only when it is. |
| `GEMINI_API_KEY` | both | worker: yes if the default provider is `gemini` | (none) | Gemini API key. Same rules as `ANTHROPIC_API_KEY`. |
| `DEFAULT_LLM_PROVIDER` | both | no | `anthropic` | `anthropic` or `gemini`. The provider for new sessions when neither the request nor the org picks one. |
| `DEFAULT_MODEL` | both | yes for `gemini` | `claude-opus-5-5` (for `anthropic`) | The model for the default provider's new sessions. |
| `DEFAULT_GEMINI_MODEL` | both | no | (none) | The Gemini model for sessions that name none, when Gemini is not the default provider. Without it, such sessions must name a model. |
| `SANDBOX_PROVIDER` | worker | no | `docker` | `docker`: local containers, for development only. `daytona`: production. |
| `DAYTONA_API_KEY` | worker | if `daytona` | (none) | Daytona API key. |
| `DAYTONA_API_URL` | worker | no | the Daytona SDK's default (`https://app.daytona.io/api`) | Daytona API base URL. |
| `SANDBOX_TEMPLATE` | worker | no | the provider's default | What new sandboxes start from when no repo snapshot applies. Docker: an image (default `keith-sandbox:latest`). Daytona: `snapshot:<name>` or `image:<ref>` (see [Sandboxes in production](#sandboxes-in-production)). |
| `GIT_BASE_URL` | worker | no | `https://github.com/` | Where sandboxes clone from: `GIT_BASE_URL` + `owner/name.git`. The credential helper answers only for this URL's scheme and host. Change it only for tests or a local git server. |
| `MAX_SESSION_DURATION` | worker | no | `2h` | A Go duration. Caps one run of a session (per wakeup); beyond it, the session fails with `time_limit`. |
| `MAX_SESSION_TOKENS` | worker | no | `20000000` | Caps a session's cumulative model tokens (input, output, cache reads and writes). `0` turns the cap off. |
| `SANDBOX_IDLE_TIMEOUT` | worker | no | `30m` | How long a parked sandbox (session finished or waiting for an answer) is kept before the reaper deletes it. A later message then re-provisions it from the pushed branch. |
| `WORKER_CONCURRENCY` | worker | no | `10` | Sessions one worker process runs at once. |

### Web app

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | yes, except in mock mode | `http://localhost:8080` | Base URL of `keith api`. The WebSocket URL is derived from it. Next.js inlines it at build time, so rebuild after changing it. |
| `NEXT_PUBLIC_API_MOCK` | no | `0` | `1` runs the whole UI against an in-browser mock API, with no backend. `npm run dev:mock` sets it. |

`web/.env.example` has both; copy it to `web/.env.local`.

### CLI and tests

| Variable | Used by | Default | Description |
|---|---|---|---|
| `GITHUB_TOKEN` | `keith run --local` | (none) | A fine-grained personal access token, used instead of the App. See [`keith run --local`](#run-one-task-with-no-app-keith-run---local). |
| `KEITH_API_URL` | `keith sessions`, `keith logs -f` (experimental) | `http://localhost:8080` | The api these client commands talk to. Experimental: the name may change. |
| `KEITH_SESSION_COOKIE` | `keith sessions`, `keith logs -f` (experimental) | (none) | The value of your browser's `keith_session` cookie for the api. Treat it like a password. |
| `KEITH_IT` | `go test` | unset | `1` runs the integration tests (Postgres, Redis, Docker). |
| `KEITH_IT_LABEL` | `go test` | `keith.test=it` | Label (`key=value`) put on every container and image the sandbox tests create, so leftovers can be found. |

### docker compose knobs

| Variable | Default | Description |
|---|---|---|
| `KEITH_POSTGRES_PORT` | `5432` | Host port for Postgres (bound to 127.0.0.1). |
| `KEITH_REDIS_PORT` | `6379` | Host port for Redis (bound to 127.0.0.1). |
| `KEITH_API_PORT` | `8080` | Host port for the `api` container (app profile). |
| `KEITH_DOCKER_GID` | `0` | Group ID of the docker socket, given to the `worker` container. `0` suits Docker Desktop. On Linux: `stat -c %g /var/run/docker.sock`. |
| `KEITH_DOCKER_SOCKET` | `/var/run/docker.sock` | Host docker socket to mount into the `worker` (for example `$XDG_RUNTIME_DIR/docker.sock` for rootless Docker). |
| `KEITH_VERSION` | `compose` | Version string baked into the image built by compose. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `keith` | The compose Postgres credentials and database. |

### Example `.env`

Placeholders only; never commit a filled-in `.env`. This file works both with `set -a; . ./.env; set +a` on the host and with docker compose. The compose `api` and `worker` services replace `DATABASE_URL` and `REDIS_URL` with the compose service names.

```bash
# --- api and worker ---
DATABASE_URL=postgres://keith:keith@localhost:5432/keith?sslmode=disable
REDIS_URL=redis://localhost:6379/0
SECRETS_KEY=<openssl rand -base64 32>

GITHUB_APP_ID=<app id>
# One of these two. The inline form also works inside containers; generate it with:
#   printf "GITHUB_APP_PRIVATE_KEY='%s'\n" "$(awk '{printf "%s\\n", $0}' keith-dev.pem)" >> .env
GITHUB_APP_PRIVATE_KEY_PATH=/absolute/path/to/keith-dev.private-key.pem
# GITHUB_APP_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n'

ANTHROPIC_API_KEY=<sk-ant-...>
# GEMINI_API_KEY=<...>
# DEFAULT_LLM_PROVIDER=anthropic
# DEFAULT_MODEL=claude-opus-5-5

# --- api ---
GITHUB_APP_CLIENT_ID=<client id>
GITHUB_APP_CLIENT_SECRET=<client secret>
GITHUB_WEBHOOK_SECRET=<openssl rand -hex 32>
# GITHUB_APP_SLUG=keith-dev
PUBLIC_URL=http://localhost:8080
WEB_ORIGIN=http://localhost:3000
SESSION_SECRET=<openssl rand -hex 32>

# --- worker ---
SANDBOX_PROVIDER=docker
# DAYTONA_API_KEY=<...>
# SANDBOX_TEMPLATE=
# MAX_SESSION_DURATION=2h
# MAX_SESSION_TOKENS=20000000
# SANDBOX_IDLE_TIMEOUT=30m
# WORKER_CONCURRENCY=10
```

## Local development

Prerequisites:

- Go 1.26.10 or newer
- Docker
- Node.js 20.9 or newer (for `web/`)
- a development GitHub App and a filled-in `.env` (above)

```bash
# 1. Postgres and Redis (127.0.0.1:5432 and :6379).
docker compose up -d --wait

# 2. The sandbox image the docker provider starts sessions from.
docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox

# 3. api and worker, each in its own terminal (export .env in each).
set -a; . ./.env; set +a; go run ./cmd/keith api
set -a; . ./.env; set +a; go run ./cmd/keith worker

# 4. The web app on http://localhost:3000.
cd web
npm ci
NEXT_PUBLIC_API_URL=http://localhost:8080 npm run dev
```

Open http://localhost:3000, sign in with GitHub, open **Connect** to install the App on a repo, then start a session.

- Use `localhost` everywhere (`PUBLIC_URL`, `WEB_ORIGIN`, the App's Callback URL, `NEXT_PUBLIC_API_URL`). Browsers treat `127.0.0.1` as a different site, so mixing the two breaks the cookies.
- For UI-only work, `cd web && npm run dev:mock` runs the whole UI against an in-browser mock, with no backend.
- If ports 5432 or 6379 are taken, see the comment at the top of `docker-compose.yml`: set `KEITH_POSTGRES_PORT` and `KEITH_REDIS_PORT`, and put the same ports in the URLs.
- Stop everything with `docker compose down`. Add `-v` to also delete the data.

### Everything in containers

The api and worker also run from compose, under the `app` profile, built from the root `Dockerfile` and configured from `.env`:

```bash
docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox
docker compose --profile app up -d --build --wait
docker compose --profile app logs -f api worker
docker compose --profile app down
```

The web app still runs with `npm run dev`. Plain `docker compose up -d --wait` (no profile) starts only Postgres and Redis, as before.

> **Security: the worker container mounts the host's docker socket.** That gives it root-equivalent control of the host. It is acceptable for local development only. In production, use `SANDBOX_PROVIDER=daytona` and never mount a socket.

Notes for the containerized worker:

- **Sandboxes are siblings, not children.** They run on the host's Docker daemon, so `keith-sandbox:latest` must exist on the host. Their published ports bind to the host's 127.0.0.1.
- **Leave `GIT_BASE_URL` at `https://github.com/`.** Sandboxes clone from it directly.
- **Socket access.** The image runs as the non-root user `keith` (uid 10001). Compose adds the socket's group (`KEITH_DOCKER_GID`, default `0` for Docker Desktop). On Linux, set `KEITH_DOCKER_GID=$(stat -c %g /var/run/docker.sock)`.
- **Private key.** `GITHUB_APP_PRIVATE_KEY_PATH` is resolved inside the container, so use the inline `GITHUB_APP_PRIVATE_KEY` form in `.env`, or mount the key.

## Run one task with no App: `keith run --local`

`keith run --local` runs one session end to end on your machine:

- no api, Postgres, Redis or GitHub App;
- a personal access token instead of App tokens;
- a local Docker sandbox.

It is the quickest way to check that the agent works on a real repo.

```bash
docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox

GITHUB_TOKEN=<fine-grained PAT> \
ANTHROPIC_API_KEY=<sk-ant-...> \
go run ./cmd/keith run --local --repo owner/name "Fix the flaky date test in utils"

# Start from an issue, and/or skip opening the PR:
go run ./cmd/keith run --local --repo owner/name --issue https://github.com/owner/name/issues/42 --no-pr "Fix this issue"
```

- **Token.** Create a fine-grained PAT (GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens) limited to that one repo:
  - Contents: Read and write
  - Pull requests: Read and write
  - Issues: Read-only, to use `--issue` on a private repo
  - a short expiry
- **What you see.** It streams the session's events to the terminal: messages, tool calls and shell output. When the agent asks a question, it reads your answer from stdin. At the end it prints the final diff.
- **`--no-pr`** skips opening the pull request.
- **The PAT reaches the sandbox.** It goes in through the same path as an App token: the file `/run/keith/git-token`, read by the git credential helper. The same token also reads the issue and opens the PR, so, unlike with the App, there is no separate worker-only token. Anything running in the sandbox can read it, which is why it should be fine-grained, limited to one repo and short-lived.
- **Needs** Docker and the `keith-sandbox:latest` image. To use Gemini, set `DEFAULT_LLM_PROVIDER=gemini`, `GEMINI_API_KEY` and `DEFAULT_MODEL`.

## Tests

```bash
go build ./... && go vet ./...
go test ./...                       # unit tests; integration tests skip themselves

# Integration tests: Postgres, Redis and Docker.
docker compose up -d --wait
KEITH_IT=1 go test ./...            # uses the compose URLs unless DATABASE_URL / REDIS_URL are set
docker compose down -v

# Web app
cd web
npm test                            # vitest
npm run lint                        # eslint + tsc
```

- **Integration tests** create scratch databases on the compose Postgres. Their default URLs are `postgres://keith:keith@localhost:5432/keith?sslmode=disable` and `redis://localhost:6379/0`.
- **Sandbox and end-to-end tests** need a Docker daemon.
  - The end-to-end test, `internal/e2e`, runs a whole session against a local Docker sandbox. It uses the small test image (`docker build -f sandbox/test.Dockerfile -t keith-sandbox-test sandbox`); the tests also build it themselves (tagged by the Dockerfile's hash) when it is missing.
  - The sandboxes clone from a git server the test runs on the host, which they reach through `host.docker.internal`. `KEITH_E2E_GIT_BIND` overrides the address it listens on.
  - The test uses Redis database 7 (`KEITH_E2E_REDIS_DB`), so its queues stay apart from other packages' tests.
  - Containers and images the tests leave behind carry the label `keith.test=it` (or `KEITH_IT_LABEL`).
- **After changing event payloads** in `internal/events`, run `make gen`. It regenerates `web/src/lib/api/events.gen.ts`; commit the result.

## Sandbox benchmark: `keith sandbox-bench`

`keith sandbox-bench` times the sandbox lifecycle on the configured provider:

- create (cold start)
- first exec
- pause and resume
- an exec after resume
- delete
- with `--shells`, also creating a tmux shell and running a command in it

Every sandbox it creates is labelled `keith.bench=1` and deleted at the end, including on Ctrl-C.

```bash
# Docker (needs keith-sandbox:latest, or pass --template)
go run ./cmd/keith sandbox-bench -n 5 -p 5 --shells

# Daytona
SANDBOX_PROVIDER=daytona DAYTONA_API_KEY=<key> \
  go run ./cmd/keith sandbox-bench -n 5 -p 5 --shells --template snapshot:<name> --json
```

| Flag | Default | Meaning |
|---|---|---|
| `-n`, `--n` | `3` | Number of sandboxes to create. |
| `-p`, `--parallel` | `1` | How many run at once. |
| `--template` | provider default | Image or provider template. Docker: an image. Daytona: `snapshot:<name>`, `image:<ref>`, or a bare snapshot name. |
| `--cpus` | `0` (provider default) | vCPUs per sandbox. |
| `--memory-mb` | `0` (provider default) | Memory per sandbox in MB. |
| `--shells` | off | Also time a tmux shell. The image needs tmux and bash. |
| `--json` | off | Print the report as JSON. |
| `--label key=value` | none | Extra sandbox label (repeatable). |

It reads `SANDBOX_PROVIDER`, `DAYTONA_API_KEY` and `DAYTONA_API_URL` from the environment. Nothing else is required.

## Deploying

The production shape:

- **api and worker:** two services on Fly.io or Railway, from the same image.
- **web:** on Vercel.
- **Postgres and Redis:** managed.
- **sandboxes:** on Daytona.

### api and worker

Both services build from the root `Dockerfile`. The image's entrypoint is `keith` and its default command is `api`:

| Service | Command (platform overrides only CMD) | Command (platform replaces the whole command) |
|---|---|---|
| api | `api` (or leave the default) | `keith api` |
| worker | `worker` | `keith worker` |

Set the configuration through the platform's secret store, never in the image.

- **Both services:** `DATABASE_URL`, `REDIS_URL`, `SECRETS_KEY`, `GITHUB_APP_ID`, and `GITHUB_APP_PRIVATE_KEY` (the PEM as a multi-line secret, or one line with `\n`), plus the LLM key(s).
- **api only:** the OAuth client ID and secret, `GITHUB_WEBHOOK_SECRET`, `PUBLIC_URL`, `WEB_ORIGIN` and `SESSION_SECRET`.
- **worker only:** `SANDBOX_PROVIDER=daytona`, `DAYTONA_API_KEY` and `SANDBOX_TEMPLATE`.

**api**

- Expose port 8080 (or set `LISTEN_ADDR` to the port the platform routes to) over HTTPS.
- Health check: `GET /healthz`. It returns `200 {"ok":true}` when Postgres and Redis are reachable, `503` otherwise.
- The api is stateless and can run several replicas. WebSocket connections need the platform's proxy to allow long-lived connections.
- On SIGTERM it drains for up to 25 seconds.

**worker**

- No inbound port and no health check endpoint.
- Never use the docker provider in production: set `SANDBOX_PROVIDER=daytona`.
- Scale horizontally: run more worker processes or machines. A Redis lease per session ensures only one worker runs a session at a time, so extra workers and rolling deploys are safe.
- Give it a stop timeout of 45 to 60 seconds, so running sessions can checkpoint and park on SIGTERM. On Fly.io, set `kill_timeout`; on Railway, set `RAILWAY_DEPLOYMENT_DRAINING_SECONDS`.

**Migrations** run automatically when the api and worker start, behind an advisory lock. There is no separate migrate step.

**Example (Fly.io).** A sketch of a `fly.toml` with one app and two process groups:

```toml
app = "keith"
kill_timeout = 60          # seconds between SIGINT and SIGKILL; workers park sessions

[build]
  dockerfile = "Dockerfile"

[processes]                # each value replaces the image's CMD; ENTRYPOINT stays keith
  api = "api"
  worker = "worker"

[http_service]
  processes = ["api"]
  internal_port = 8080
  force_https = true

  [[http_service.checks]]
    method = "GET"
    path = "/healthz"
    interval = "15s"
    timeout = "5s"
    grace_period = "20s"
```

```bash
fly secrets set GITHUB_APP_PRIVATE_KEY="$(cat keith.private-key.pem)" SECRETS_KEY=... # and the rest
fly deploy
fly scale count worker=2
```

**Railway:**

- Create two services from the repo, both building the Dockerfile.
- Give the worker the start command from the table above.
- Set the api's healthcheck path to `/healthz`, and route its domain to port 8080.

### Sandboxes in production

Daytona sandboxes should start from the Keith sandbox image, which has tmux, git, the `agent` user and `/workspace`:

1. Build `sandbox/base.Dockerfile` for `linux/amd64` and push it to a registry Daytona can pull from.
2. Choose one of:
   - `SANDBOX_TEMPLATE=image:<registry>/keith-sandbox:<tag>`: Daytona builds it on first use, which can take minutes.
   - Better: create a Daytona snapshot from that image and set `SANDBOX_TEMPLATE=snapshot:<name>`. Snapshots boot in seconds.
3. If `SANDBOX_TEMPLATE` is empty, Daytona uses the organization's default snapshot, which may lack tmux.

Compare cold starts with `keith sandbox-bench`. The worker also snapshots single-repo sessions after their first successful setup, and later sessions on that repo start from the snapshot.

On Daytona, the worker restricts sandbox egress to GitHub and the common package registries (npm, Yarn, PyPI, the Go proxy, crates.io, RubyGems, Maven Central). The list is `DefaultEgressAllowlist` in `internal/worker`. A repo whose install step needs another host fails setup there. The local docker provider does not restrict egress.

### Web app (Vercel)

- Import the repo and set **Root Directory** to `web`. The framework preset is Next.js.
- Set `NEXT_PUBLIC_API_URL` to the api's public URL, for example `https://api.example.com`. It is inlined at build time, so redeploy after changing it.
- On the api, set `WEB_ORIGIN` to the web app's exact origin, for example `https://app.example.com`.
- Preview deployments have other origins, which the api rejects (one `WEB_ORIGIN`). Use `NEXT_PUBLIC_API_MOCK=1` for the Preview environment if you want previews to render.

### Cookies and domains

The api's session cookie, `keith_session`:

- is `HttpOnly; Secure; SameSite=Lax`;
- is host-only on the api's host (no `Domain` attribute);
- is sent with the web app's credentialed `fetch` and WebSocket calls only if the web app and the api are the **same site**, meaning the same registrable domain.

So:

- `app.example.com` with `api.example.com` works.
- `keith.vercel.app` with `keith.fly.dev` does **not** work. Both `vercel.app` and `fly.dev` are public suffixes, so every app there is its own site. Users would sign in and then get `401` on every call.

Give the web app and the api custom domains under one domain you own.

Locally, `http://localhost:3000` and `http://localhost:8080` are the same site. Ports do not count.

### Postgres

Use any managed Postgres (Neon, Supabase, RDS, Fly Postgres):

- Use the **direct** connection string with `sslmode=require` (or stricter).
- Avoid transaction-mode poolers such as PgBouncer, the Supabase pooler or Neon's `-pooler` host. Migrations take a session-level advisory lock, and pgx uses prepared statements.
- The database user must be able to create tables: the services run the migrations themselves.

### Redis

Use a managed Redis with Lua scripting (`EVAL`) and pub/sub. asynq needs both, and Keith uses pub/sub for live events and Stop signals.

- Upstash and Redis Cloud both work. Use the TLS URL (`rediss://...`).
- Use a single endpoint: Keith connects with a plain (non-cluster) client.
- Set the eviction policy to `noeviction`, so queued jobs and leases are never evicted.
- asynq polls Redis continuously. On per-request pricing (Upstash pay-as-you-go), prefer a fixed-price plan.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| A session fails at provisioning with `No such image: keith-sandbox:latest` | Build the sandbox image on the host that runs the containers: `docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox`. With the compose worker, build it on the host, not in the worker container. |
| `permission denied while trying to connect to the Docker daemon socket` in the worker container | The container lacks the socket's group. On Linux, set `KEITH_DOCKER_GID=$(stat -c %g /var/run/docker.sock)`. For rootless Docker, also set `KEITH_DOCKER_SOCKET`. |
| GitHub says "The redirect_uri is not associated with this application" | `PUBLIC_URL` + `/auth/github/callback` is not one of the App's Callback URLs. They must match exactly: scheme, host and port. |
| Sign-in returns to `/login?error=oauth_state` | The state cookie was set on one host and the callback hit another. `NEXT_PUBLIC_API_URL` and `PUBLIC_URL` must share a host: `localhost` vs `127.0.0.1` counts as different, as does a tunnel URL in only one of them. The state also expires after 10 minutes. |
| Signed in, but every `/api/*` call is `401` | The browser is not sending `keith_session`. Either the web app and api are not same-site (see [Cookies and domains](#cookies-and-domains)), or `PUBLIC_URL` is `https://` while you browse the api over `http://` (`Secure` cookie). |
| `403 bad_origin` on writes, or the WebSocket is refused | `WEB_ORIGIN` does not exactly equal the browser's origin: scheme, host, port, no trailing slash. |
| `/connect?error=installation_not_verified` | "Request user authorization (OAuth) during installation" is not checked on the App. |
| `/connect?error=setup_state` | The App was installed from GitHub's page, not from Keith's **Connect** page, or the install took more than 15 minutes. |
| Webhook deliveries fail with `401` | `GITHUB_WEBHOOK_SECRET` differs from the App's webhook secret, or a relay modified the body. |
| Creating a session answers `422 provider_unavailable` | The api has no key for that provider. Set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` on the api too, not only on the worker. |
| Sessions stay `queued` | No worker is running, or it uses a different `REDIS_URL` than the api. Check the worker's logs. |
| `config: SECRETS_KEY: decodes to N bytes, want 32` | Generate it with `openssl rand -base64 32`, not `-hex`. |
| `GITHUB_APP_PRIVATE_KEY: not a PEM-encoded key` | The newlines were lost. Use real newlines, `\n` escapes, or `GITHUB_APP_PRIVATE_KEY_PATH`. With `docker run --env-file`, do not quote the value. |
| `GITHUB_APP_PRIVATE_KEY and GITHUB_APP_PRIVATE_KEY_PATH: set only one` | Remove one of them, including any copy in the shell environment. |
