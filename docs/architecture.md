# Keith architecture

Keith is a multi-tenant cloud coding agent. A user connects GitHub, tags one or more repos and describes a task. An agent then works in its own cloud sandbox, streams every step to the browser and opens a pull request.

## Services

One `keith` binary runs every role, chosen by subcommand:

| Role | Command | Does |
|---|---|---|
| API | `keith api` | HTTP + WebSocket API, GitHub OAuth sign-in, GitHub App setup and webhooks, writes `sessions` rows and enqueues `run_session` jobs, replays and streams events. |
| Worker | `keith worker` | Takes a session lease, mints a repo-scoped installation token, provisions the sandbox, runs the agent loop, opens the PR, parks the sandbox. |
| Web | `web/` (Next.js) | Renders only from the event stream. |

Postgres holds state and the event log. Redis carries asynq jobs, event pub/sub and session leases.

## Package map

```
cmd/keith ── wires config and services
  ├── internal/api ──────┬── store, events, queue, github, secrets, config
  │                      └── never imports agent
  └── internal/worker ───┬── agent, llm/<provider>, sandbox/<provider>,
                         │   github, store, events, queue, secrets, config
                         └── the only package that imports everything

internal/agent      → llm, sandbox, events          (loop, tools, prompt)
internal/llm        → (nothing internal)             contract + llmtest fake
internal/llm/{anthropic,gemini}   → llm              SDKs stay inside
internal/sandbox    → (nothing internal)             contract
internal/sandbox/{docker,daytona,fake} → sandbox     SDKs stay inside
internal/events     → (nothing internal)             contract + memevents fake
internal/store      → events                         pgx, implements events.Store
internal/github     → agent (for RepoHost types)     implements agent.RepoHost
internal/config     → (nothing internal)
internal/notify/*   → (nothing internal)             V2 channels
migrations          → (nothing)                      goose SQL, embedded
```

Rules:
- `agent` imports `llm`, `sandbox` and `events` only. It reaches GitHub through the `agent.RepoHost` interface and never imports `github`, `store` or a concrete provider.
- `api` never imports `agent`. The API only writes rows, enqueues jobs and streams events; it never runs the loop.
- Provider SDK types (Anthropic, Gemini, Daytona, Docker) stay inside their provider package. Everything else uses the contract types.
- A session is pinned to one LLM provider and model for life (`sessions.provider`, `sessions.model`, `agent_state.provider`). Thinking signatures only replay on the provider that made them.

## Trust boundary

The sandbox is untrusted. It runs whatever the repo and the agent run, and users can open shells in it.

- Never inside a sandbox: LLM API keys, the GitHub App private key, the App's OAuth client secret, agent code, prompts or the conversation.
- Inside a sandbox: the repos at `/workspace/<repo>`, a git credential helper holding a 1-hour installation token scoped to the session's repos with `contents: write` only, and the org's own secrets as env vars.
- The worker runs every tool call. Shell and file tools become sandbox calls. `read_issue`, `create_pull_request`, `message_user` and `ask_user` run in the worker with credentials that never reach the sandbox.
- The agent pushes only to `agent/*` branches. Sandbox egress is allowlisted. The agent runs as a non-root user.
- Org and repo secrets are encrypted at rest (envelope encryption under `SECRETS_KEY`) and redacted from every event.
- In V1 the GitHub token service is a library (`internal/github`) that the API and worker both call in-process, so both control-plane services hold the App private key. Moving it into its own service is a later hardening step.

## Event log is the source of truth

Each session has one append-only log in `events`, keyed `(session_id, seq)`. The UI renders only from it, so refresh, reconnect and a second tab all behave the same way.

1. One writer per session. The worker holding the session's Redis lease assigns `seq`.
2. Store, then publish. Insert into Postgres, then publish on Redis. `message.delta` is the only ephemeral type: it is published and never stored, and the finished `message.assistant` is stored. `events.Emitter` enforces this order.
3. Replay on connect. The client sends `{session_id, after_seq}`. The server subscribes, sends stored events after `after_seq`, then forwards live ones. The client drops any `seq` it already has. Pub/sub is best-effort; the store is not.
4. User input goes through plain API requests. A mid-run message is queued for the next model turn. Stop sets `sessions.cancel_requested_at`, which the worker checks between steps, and interrupts the running shell.

Chat has no table of its own: it is the `message.*` rows in `events`. The model's own conversation lives in `agent_state.llm_messages` so a paused session resumes exactly.
