# Keith architecture

Keith is a multi-tenant cloud coding agent. A user connects GitHub, tags one or more repos and describes a task. An agent then works in its own cloud sandbox, streams every step to the browser and opens a pull request.

## Services

One `keith` binary runs every role, chosen by subcommand:

| Role | Command | Does |
|---|---|---|
| API | `keith api` | HTTP + WebSocket API, GitHub OAuth sign-in, GitHub App setup, verifies webhooks and queues them, writes `sessions` rows and enqueues `run_session` / `resume_session` jobs, replays and streams events. |
| Worker | `keith worker` | Runs sessions (see [Worker](#worker-internalworker)) and processes queued webhooks. Takes a session lease, mints repo-scoped installation tokens, provisions and sets up the sandbox, runs the agent loop, verifies and ships the PR, parks the sandbox. A reaper deletes idle sandboxes. |
| Web | `web/` (Next.js) | Renders only from the event stream. |

Postgres holds state and the event log. Redis carries asynq jobs, event pub/sub, session leases and control signals (Stop). Both services apply the migrations at start, behind an advisory lock. [SETUP.md](../SETUP.md) covers configuration and deployment.

## Package map

```
cmd/keith ── wires config and services: api, worker, run --local, sandbox-bench
  ├── internal/api ──────┬── store, events, queue, control, hooks, github,
  │                      │   secrets, config
  │                      └── never imports agent
  └── internal/worker ───┬── agent, llm, sandbox, sandbox/tmuxshell, github,
                         │   store, events, queue, lease, control, secrets,
                         │   config
                         └── the only package that runs the agent loop

internal/agent             → llm, sandbox, events         loop, tools, system prompt
internal/agent/agenttest   → agent, events, llm, sandbox  fakes + append-only check
internal/llm               → (nothing internal)           provider contract
internal/llm/{anthropic,gemini} → llm                     SDKs stay inside
internal/llm/llmtest       → llm                          scripted provider for tests
internal/llm/providers     → config, llm, llm/{anthropic,gemini}   builds a provider by name
internal/sandbox           → (nothing internal)           contract: Provider, Sandbox, Shells
internal/sandbox/docker    → sandbox                      local containers via the docker CLI
internal/sandbox/daytona   → sandbox                      Daytona; the SDK stays inside
internal/sandbox/memsandbox → sandbox                     in-memory sandbox for tests
internal/sandbox/tmuxshell → sandbox                      named tmux shells over any Sandbox
internal/sandbox/providers → config, sandbox, sandbox/{docker,daytona}  provider from config
internal/events            → (nothing internal)           event types, Store and Bus contracts
internal/events/memevents  → events                       in-memory Store and Bus
internal/events/pgevents   → events                       Postgres events.Store
internal/events/redisbus   → events                       Redis pub/sub events.Bus
internal/store             → events, events/pgevents, migrations   pgx data layer
internal/github            → agent (for RepoHost types)   App auth, tokens, OAuth, webhooks,
                                                          implements agent.RepoHost
internal/hooks             → store, github, queue, control, events   webhook processing,
                                                          repo sync, StopSession
internal/queue             → (nothing internal)           asynq task types, client, server
internal/lease             → (nothing internal)           exclusive, expiring Redis lease
internal/control           → (nothing internal)           Stop signals on Redis pub/sub
internal/secrets           → events                       envelope encryption + Redactor
internal/config            → (nothing internal)           environment configuration
internal/worker/memstore   → events, store (model types)  in-memory worker.Store (tests, run --local)
internal/e2e               → (tests only)                 end-to-end session on Docker
internal/notify/telegram   → (nothing internal)           V2 channel
migrations                 → (nothing)                    goose SQL, embedded

Test helpers: internal/{queue/queuetest, store/storetest, redistest},
internal/sandbox/internal/{sbutil (shared provider helpers), sbtest}.
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
- The worker runs every tool call. Shell and file tools become sandbox calls. `read_issue`, `open_pr`, `message_user` and `ask_user` run in the worker with credentials that never reach the sandbox.
- The agent pushes only to `agent/*` branches. Sandbox egress is allowlisted on Daytona (GitHub and the common package registries); the local docker provider does not enforce it. The agent runs as a non-root user.
- Org and repo secrets are encrypted at rest (envelope encryption under `SECRETS_KEY`) and redacted from every event.
- Root in the sandbox: the providers run every root command (their own, and the worker's setup) with `/bin/sh` and `PATH=/usr/sbin:/usr/bin:/sbin:/bin` (`sandbox.SystemPath`), so nothing the agent writes, such as `~/.local/bin/kill`, runs as root. Docker sandboxes drop every capability except `CHOWN` and `KILL` and set `no-new-privileges`. In Daytona's own snapshots the default user has passwordless sudo, so there the sandbox user is effectively root.
- In V1 the GitHub token service is a library (`internal/github`) that the API and worker both call in-process, so both control-plane services hold the App private key. Moving it into its own service is a later hardening step.

## Event log is the source of truth

Each session has one append-only log in `events`, keyed `(session_id, seq)`. The UI renders only from it, so refresh, reconnect and a second tab all behave the same way.

1. One writer per session. The worker holding the session's Redis lease assigns `seq`.
2. Store, then publish. Insert into Postgres, then publish on Redis. `message.delta` is the only ephemeral type: it is published and never stored, and the finished `message.assistant` is stored. `events.Emitter` enforces this order.
3. Replay on connect. The client sends `{session_id, after_seq}`. The server subscribes, sends stored events after `after_seq`, then forwards live ones. The client drops any `seq` it already has. Pub/sub is best-effort; the store is not.
4. User input goes through plain API requests. A mid-run message is queued for the next model turn. Stop sets `sessions.cancel_requested_at`, which the worker checks between steps, and interrupts the running shell.

Chat has no table of its own: it is the `message.*` rows in `events`. The model's own conversation lives in `agent_state.llm_messages` so a paused session resumes exactly.

## Worker (`internal/worker`)

`keith worker` runs:

- two asynq servers on Redis: one for the `sessions` queue (`run_session`, `resume_session`; `WORKER_CONCURRENCY` at once), one for the `webhooks` queue (`github_webhook`, 4 at once, handled by `hooks.Handler`). Long sessions therefore never starve uninstall handling.
- the reaper loop.

On SIGTERM (or SIGINT) running sessions stop at once. Each one saves its state, parks its sandbox, releases its lease and returns `ErrBusy`, so the task is retried, possibly on another worker, and resumes from `agent_state`.

Workers scale horizontally: the per-session lease, not the queue, guarantees one writer per session.

### One task

`run_session` and `resume_session` share one handler. A `resume_session` is a wakeup, not a command: the worker decides what to do from the stored session (status, pending `session_inputs`, cancel flag, `agent_state`), so duplicate or late wakeups are harmless.

1. **Lease.** Take the session's Redis lease, or return `queue.ErrBusy`; the task is then retried in 5 s without counting as a failure.
   - `KeepAlive` renews the lease, and if it is lost, the run's context is cancelled.
   - The event emitter checks the lease before every append. pgevents does not check the fencing token, so this closes the fencing gap.
   - The emitter also redacts every payload with a `secrets.Redactor`, built from the session's decrypted org and repo secrets plus the sandbox git token as it rotates. It streams shell output (one stream per shell) and `message.delta` (one stream per message, flushed before its `message.assistant`) through `Redactor.Stream`, so a secret split across chunks or deltas is still caught.
2. **Provision** (a new session, or one whose sandbox was reaped).
   - **Two GitHub installation tokens** per session:
     - *Sandbox token*: `contents: write` on the session's repos only. It reaches the sandbox only as `/run/keith/git-token`, read by the git credential helper. The helper is installed as root at `/usr/local/bin/git-credential-keith` and answers only for `GIT_BASE_URL`'s host. The token file is rewritten from the token source's refresh hook, so the token rotates without re-cloning.
     - *Worker token*: `WorkerPermissions`, on the session's repos plus the issue's repo. It backs the `RepoHost` behind `read_issue` and `open_pr`, and never enters the sandbox.
     - In V1 all of a session's repos must be on one installation. The API refuses to create a session that spans two (`422`); a session whose repos end up on two anyway fails with `repos_span_installations`.
     - A repo the session's org can no longer use (removed from the installation, the App uninstalled, or moved to another org: `SessionRepo.Inaccessible`, or no installation) fails the session with `repo_access_lost` before any token is minted, on a new session and on every resume.
   - **The sandbox** starts from the repo's snapshot (a single-repo session on the same provider) or from the base template (`SANDBOX_TEMPLATE`). Org and repo secrets are its environment variables; a repo secret overrides an org secret of the same name.
   - **Repos** are cloned as the sandbox user to `/workspace/<name>`, from plain URLs (`GIT_BASE_URL` + `owner/name.git`; never a token in a URL). The worker checks out `agent/<session-id>`: the remote branch if it exists, else a new branch from the default branch.
3. **Set up** (status `setting_up`).
   - Each repo profile's install command runs in a tmux shell, streamed to the UI as a synthetic tool call. A failing install does not fail the session: the agent is told, and can fix it.
   - A Stop during setup interrupts the running install, starts no further install, and cancels the session before the agent starts.
   - After the first successful setup of a single-repo session that has no current snapshot, the worker snapshots the sandbox. Only a session's first provision, checked out from `origin/<default>`, is snapshotted: a re-provision checks out the agent's pushed branch (agent-written lockfiles and install scripts), which must never reach other sessions through the shared snapshot. Before the snapshot it removes the token file and shell logs, and the docker provider blanks the env vars, so neither the token nor the secrets are baked in. It then records the snapshot with a hash of the repo's lockfiles.
   - Later sessions on that repo start from the snapshot. They fetch and check out, and re-run the install only when the lockfile hash has changed.
4. **Run the loop** (status `running`).
   - The worker emits the session's first `message.user` (the prompt) itself, then builds a new `agent.State` or loads the saved one.
   - It wires `agent.Runner`:
     - `Inputs`: pending `session_inputs`.
     - `Delivered`: marks them consumed, and sets status `running` after an answer.
     - `ShouldStop`: the cancel flag, a control-channel Stop (which also interrupts every shell), or the time limit.
     - `Checkpoint`: saves `agent_state`.
     - `AfterFileChange`: saves the repo diff against the base commit, untracked files included.
     - `OnPROpened`: records the PR.
   - Limits: `MAX_SESSION_DURATION` per run, `MAX_SESSION_TOKENS` per session, and a step cap per run.
5. **Act on the outcome.**

   | Outcome | What the worker does | Status |
   |---|---|---|
   | `end_turn` | **Verify:** run each changed repo's test command. On failure, feed the result back to the model as a message and loop again, at most 2 cycles. **Ship:** if there are changes and the agent opened no PR, commit, push and open one itself. | `completed`, or `no_changes` when nothing changed |
   | `paused` | Record the open question. | `waiting_for_user` |
   | `cancelled` | Interrupt running commands, clear the open question and the cancel flag. | `cancelled` |
   | `step_cap`, `budget_exceeded`, `failed` (and the time limit) | Emit an `error` event with the reason. | `failed` |

6. **Re-check, then park.** After any terminal or `waiting_for_user` status, and before releasing the lease, the worker re-checks pending inputs and the cancel flag (the queue contract in `internal/queue`): input that arrived while it was finishing has no other wakeup. Then it parks:
   - delete the token file;
   - revoke both tokens;
   - pause the sandbox;
   - release the lease.

   A worker that has lost the lease never touches the sandbox again.

### Recovery

A task error is retried (three times; `ErrBusy` is not counted), but some session states must not depend on that:

- **End of turn.** An error from verify or ship (a git command failing, say) fails the session with `ship_failed` rather than failing the task: a retry would meet the same git state.
- **Exhausted tasks.** asynq archives a task whose retries are used up, even when its last error is `ErrBusy`. The session server's `ErrorHandler` (`queue.SessionErrorHandler`) calls `Worker.TaskExhausted` first. After `ErrBusy` (another worker held the lease, or this one was shut down or lost it) it enqueues a fresh `resume_session` (reason `wakeup`). After any other error it fails the session with `internal_error`, under its lease, if the session still needs a run.
- **The sweeper.** Each reaper pass also wakes (`resume_session`, reason `wakeup`) every active session (`queued`, `provisioning`, `setting_up`, `running`) that no worker holds and that has not been updated for 10 minutes: its task is gone, and a Stop on it would otherwise never be honoured. A duplicate wakeup is harmless.

### Resume

A follow-up message, an answer or a Stop on a parked session enqueues `resume_session`.

The API checks the org's active-session cap (`MAX_ACTIVE_SESSIONS_PER_ORG`) before it queues a resume, but two resumes queued together can both pass there. So the worker checks again when it makes a parked session active (its first `provisioning` or `running`): `store.SetSessionStatusCapped` counts the org's active sessions and writes the status in one transaction, under the org row lock (org first, then the session row, as in the API). Over the cap the session fails with `session_cap` and the input stays pending for the next message.

- **Sandbox still parked:** the worker resumes it, writes a fresh token, reattaches the tmux shells and continues from `agent_state`.
- **Sandbox never fully provisioned** (a crash, a lost lease or a failed clone before provisioning finished): the worker deletes it and provisions a new one. Provisioning ends by writing `/run/keith/provisioned`, holding the sandbox's ID; a sandbox without it is not resumed. A Stop during setup also leaves it unwritten.
- **Sandbox reaped:** the worker provisions a new one and clones again. It checks out `agent/<id>` if that branch exists on the remote, else starts fresh from the default branch. Uncommitted work from the reaped sandbox is lost; this is a known limitation.

### Reaper

Every minute, the reaper deletes:

- the sandboxes of sessions parked (terminal or `waiting_for_user`) for longer than `SANDBOX_IDLE_TIMEOUT`, checked under the session's lease;
- orphaned docker containers: labelled `keith.role=worker` (set by docker-provider workers), referenced by no session, and older than a grace period (10 minutes) that covers the moment between create and record.

A sandbox that cannot be deleted is logged and skipped; it never holds up the rest of the pass. The reaper keeps the session's lease alive while it deletes, and then clears `sessions.sandbox_id` only if it still names the deleted sandbox (compare-and-set), so a sandbox another worker recorded meanwhile is never forgotten.

### Local mode

`keith run --local` runs the same session code in one process:

- `internal/worker/memstore` in place of Postgres;
- in-memory events and an in-process lease in place of Redis;
- the docker provider;
- one personal access token in place of both App tokens. It reaches the sandbox through the same token file and credential helper, and it also backs the `RepoHost` (`github.NewPATHost`).

## Agent loop (`internal/agent`)

The loop runs in the worker, outside the sandbox. It reaches the sandbox only through `sandbox.Sandbox` and `sandbox.Shells`, and GitHub only through `agent.RepoHost`.

### Wiring

`internal/worker` wires the loop (see [Worker](#worker-internalworker)). The seams it fills:

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
	Redact: redactSecrets,     // open_pr's PR title, body and commit message
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

A worker that stops while a tool runs (a crash, a lost lease) leaves the state of the checkpoint after the model call: tool calls with no results. The next `Run` answers each missing one before its first step, with an error result telling the model the call was interrupted and to check the current state, and emits `tool_call.finished` for it with summary `interrupted`. The model then carries on.

### Events

The agent emits:

- `message.user` (when an input is handed to the model)
- `message.delta` and `message.assistant` (under one message ID)
- `tool_call.started` and `tool_call.finished` (when the worker's context is cancelled mid-call, `tool_call.finished` still goes out, on a short context detached from the cancellation, with summary `interrupted` if the tool failed)
- `shell.created`
- `file.changed` (after `AfterFileChange`)
- `question.asked` (after the whole tool batch, so the worker's `session.status waiting_for_user` follows it directly)
- `pr.opened` (after `OnPROpened`)

The worker emits `session.status`, `tool_call.output` and `error`. A publish failure after a store (`events.ErrPublish`) is ignored. Any other `Emit` error stops the run with `emit_failed`.

### Known limitations

- There is no compaction yet. A conversation that outgrows the window fails with `context_exhausted`. The append-only design is ready for the provider's server-side compaction.
- Path checks are lexical. A symlink inside `/workspace` that points outside it is followed by the sandbox, and the sandbox remains the trust boundary.
- `ShouldStop` is polled between steps and between tool calls. A long `shell_exec` ends early only when the worker also calls `Shells.Interrupt`.

## Known limitations (V1)

These are known and accepted for V1, beyond the agent loop's own (see [Agent loop](#known-limitations)).

- **Event appends are not fenced.** The emitter checks the session lease before every append, but `pgevents` does not check the lease's fencing token. A worker that stalls between the check and the insert (a long GC pause, a suspended VM) can still append after its lease expired, interleaved with the new holder's events.
- **Webhook bodies are read before they are verified.** `POST /hooks/github` buffers up to 25 MiB (GitHub's own maximum payload) before it checks the HMAC signature, so an unauthenticated client can make the api hold that much per request. Rate-limit `/hooks/github` at your proxy if that matters for your deployment.
- **Repo renames are not tracked.** Keith does not subscribe to `repository` events, so a repo renamed on GitHub keeps its old `owner/name` until the installation's repos are synced again (link it again from **Connect**). GitHub redirects the old name for git and most API calls, but the redirect breaks if another repo takes the old name, and an `issue_url` that uses the new name is refused (`issue_repo_not_tagged`).
- **A follow-up cannot get past `budget_exceeded`.** `MAX_SESSION_TOKENS` caps a session's cumulative usage, and the cap is checked before every model call. A session that failed with `budget_exceeded` therefore fails again at once on a follow-up message, although the failure message suggests sending one. To continue, raise `MAX_SESSION_TOKENS` on the worker first.
- **Orphaned Daytona sandboxes are not reaped.** The reaper deletes orphaned sandboxes (created, but never recorded on a session, for example because the worker died in between) only on the docker provider, which can list its containers by label. The Daytona provider has no `List`, so such a sandbox stops after 15 idle minutes and is archived 7 days later (the provider's defaults), but is never deleted; remove it from the Daytona dashboard.
