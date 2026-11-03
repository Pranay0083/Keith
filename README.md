# Keith

Keith is a multi-tenant cloud coding agent. You connect GitHub, pick one or more repos, and describe a task or paste an issue link. An agent does the work in its own cloud sandbox, runs the tests and opens a pull request. You can watch every step live: chat, tool calls, terminal output and the diff.

## Architecture

- **Control plane in Go.** One `keith` binary runs the `api` service (HTTP + WebSocket, GitHub App, sessions) and the `worker` service (the agent loop).
- **Agent loop outside the sandbox.** The worker calls the LLM and runs each tool call against the sandbox, so LLM keys, the GitHub App key and agent code never enter it.
- **One sandbox per session** (Daytona in the cloud, Docker locally), with repos at `/workspace/<repo>` and named tmux shells.
- **Provider-agnostic LLM layer.** Claude (`claude-opus-5-5`) first, Gemini second; each session stays on one provider.
- **Event log as source of truth.** Every step is an event in Postgres, fanned out over Redis and replayed to the web app (Next.js + xterm.js) on reconnect.
- **GitHub App** with short-lived, repo-scoped installation tokens per session.

See [docs/architecture.md](docs/architecture.md) for the package map, the trust boundary and the event rules.

## Status

Phase 23, cloud pivot: the V1 backend is wired end to end (api + worker + web). The macOS assistant this repo used to be has been removed (it is in git history).

What works:

- **Sign-in and connect.** GitHub sign-in, App install and repo sync (setup flow plus webhooks), repo profiles, and org and repo secrets (encrypted at rest, redacted from events).
- **Sessions.** Create a session from a prompt or an issue link on one or more repos; send follow-up messages, answer questions and stop it; watch it live over WebSocket with replay on reconnect.
- **Worker.** One sandbox per session, on Docker locally or Daytona. The worker:
  - clones the repos onto an `agent/<id>` branch with short-lived installation tokens;
  - runs setup, with per-repo snapshots;
  - runs the agent loop on Claude or Gemini;
  - runs the repo's tests and opens the PR;
  - pauses idle sandboxes, and a reaper deletes them later.
- **`keith run --local`** runs one task against a real repo with a personal access token: no App, Postgres or Redis.
- **`keith sandbox-bench`** compares sandbox providers.

See [SETUP.md](SETUP.md) to create the GitHub App, configure, run locally, test and deploy.

## Development

Prerequisites:
- Go 1.26.10 or newer (required by the Daytona SDK)
- Docker (local sandboxes, Postgres and Redis for integration tests)

```bash
go build ./...
go vet ./...
go test ./...

# Integration tests against the compose Postgres and Redis. These URLs are
# the defaults when DATABASE_URL and REDIS_URL are unset.
docker compose up -d --wait
export DATABASE_URL='postgres://keith:keith@localhost:5432/keith?sslmode=disable'
export REDIS_URL='redis://localhost:6379/0'
KEITH_IT=1 go test ./...
docker compose down -v
```

If ports 5432 or 6379 are taken, set `KEITH_POSTGRES_PORT` and `KEITH_REDIS_PORT` for compose and put the same ports in the URLs.

Configuration comes from environment variables; see `internal/config`.

## License

MIT
