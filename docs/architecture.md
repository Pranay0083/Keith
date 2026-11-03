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
internal/agent/agenttest → agent, events, llm, sandbox  fakes + append-only check
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

## Agent loop (`internal/agent`)

The loop runs in the worker, outside the sandbox. It reaches the sandbox only through `sandbox.Sandbox` and `sandbox.Shells`, and GitHub only through `agent.RepoHost`.

### Wiring (wave D)

```go
ids := newIDFunc() // unique message IDs
r, err := agent.NewRunner(agent.Config{
	Provider: provider, Model: session.Model, Effort: llm.EffortHigh,
	Tools:      agent.DefaultTools(agent.ToolOptions{NewID: ids}),
	MaxSteps:   100,         // model calls per Run
	TokenBudget: budget,     // per session, over State.Aux.Usage
	Inputs:     pollQueuedMessages, // new user messages / answers
	ShouldStop: cancelRequested,    // sessions.cancel_requested_at
	Delivered:  emitRunningAfterAnswer,
	Checkpoint: saveAgentState,
	NewID:      ids,
})
run := &agent.Run{
	SessionID: id, Sandbox: sb, Emit: redactingEmit, Host: githubHost,
	Repos: repos, // RepoContext with Profile from repo_profiles
	GitAuthor: botIdentity,
	AfterFileChange: saveDiff, // runs before file.changed
	OnPROpened: recordPR,      // runs before pr.opened; once per repo
}
run.Shells = tmuxshell.New(sb, tmuxshell.WithOutput(func(shell string, chunk []byte) {
	emit(events.TypeToolCallOutput, events.ToolCallOutputPayload{
		CallID: run.CurrentCall(shell), ShellID: shell, Chunk: redact(chunk)})
}))
out := r.Run(ctx, run, st)
```

A new session starts from `agent.NewState(agent.BuildSystemPrompt(...))`. The system prompt is built once and never rebuilt, because Opus 5.5 binds thinking blocks to the exact prefix.

### State and `agent_state`

`State.MarshalColumns` returns `llm_messages` (the append-only `[]llm.Message`) and `raw_messages`, which holds `{"system": ..., "aux": ...}`. `Aux` contains:

- the raw provider responses
- the edit history behind `undo_edit`
- the repeat-detection counters
- the pending user turn, with any tool results held back by a pause
- the PRs already opened
- cumulative usage
- the IDs of delivered inputs

Load the state with `agent.UnmarshalState(llm_messages, raw_messages)`.

### Outcomes

The state is consistent after every outcome; save it, then:

| `Outcome.Kind` | Meaning | Suggested status |
|---|---|---|
| `end_turn` | The model finished and no input is waiting. `Text` is its final message. | `completed`, or `no_changes` if no PR was opened |
| `paused` | `Question` is open. Set `open_question_id = Question.ID`. | `waiting_for_user` |
| `cancelled` | `ShouldStop` reported true. | `cancelled` |
| `step_cap` / `budget_exceeded` | A limit was hit. | `failed` with that reason, or wait for the user |
| `failed` | `Reason` is `context_exhausted`, `llm_error`, `refusal`, `tool_error`, `interrupted`, `inputs_failed`, `checkpoint_failed` or `emit_failed`. `Err` holds the cause. | `failed` |

To resume after a pause, deliver the answer through `Inputs` with `QuestionID` set. It becomes the `ask_user` call's `tool_result`, sent with the held sibling results in one user message. A plain message answers too. Clear `open_question_id` whenever `Run` returns anything other than `paused`.

### Events

The agent emits:

- `message.user` (when an input is handed to the model)
- `message.delta` and `message.assistant` (under one message ID)
- `tool_call.started` and `tool_call.finished`
- `shell.created`
- `file.changed` (after `AfterFileChange`)
- `question.asked` (after the whole tool batch, so the worker's `session.status waiting_for_user` follows it directly)
- `pr.opened` (after `OnPROpened`)

The worker emits `session.status`, `tool_call.output` and `error`. A publish failure after a store (`events.ErrPublish`) is ignored. Any other `Emit` error stops the run with `emit_failed`.

### Known limitations

- There is no compaction yet. A conversation that outgrows the window fails with `context_exhausted`. The append-only design is ready for the provider's server-side compaction.
- Path checks are lexical. A symlink inside `/workspace` that points outside it is followed by the sandbox, and the sandbox remains the trust boundary.
- `ShouldStop` is polled between steps and between tool calls. A long `shell_exec` ends early only when the worker also calls `Shells.Interrupt`.
