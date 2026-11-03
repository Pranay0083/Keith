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

Phase 23: cloud pivot in progress; V1 being built. The macOS assistant this repo used to be has been removed (it is in git history). The service commands are stubs for now.

## Development

Prerequisites:
- Go 1.26.10 or newer (required by the Daytona SDK)
- Docker (local sandboxes, Postgres and Redis for integration tests)

```bash
go build ./...
go vet ./...
go test ./...

# Migration test against a Postgres you can create databases on
KEITH_IT=1 DATABASE_URL=postgres://keith:keith@localhost:5432/keith?sslmode=disable go test ./migrations
```

Configuration comes from environment variables; see `internal/config`.

## License

MIT
