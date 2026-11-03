// Folds a session's event log into the state the session view renders.
//
// The UI renders only from SessionState. Events go through applyEvent,
// which applies stored events exactly once and in seq order (a duplicate is
// dropped; an event that arrives ahead of a gap waits in `ahead` until the
// gap fills; one this build can't read only advances seq) and applies
// ephemeral message.delta events as they come. The few local actions (an
// optimistic message, connection status, dismissing errors) go through
// sessionReducer, which also accepts event batches.
//
// Everything here is pure: same state and input, same output, no clocks.

import {
  isEphemeral,
  isSessionStatus,
  isSkipped,
  isTerminal,
  type FileActionName,
  type SessionEvent,
  type SessionStatusName,
  type StreamEvent,
} from "@/lib/api/events";
import type { Session } from "@/lib/api/types";

// --- State ---

export interface ChatMessage {
  kind: "message";
  /** message_id. */
  id: string;
  role: "user" | "assistant";
  /** The author's users.id, from message.user; null when no person wrote it. */
  userId: string | null;
  text: string;
  /** Only deltas have arrived; the final message.assistant has not. */
  streaming: boolean;
  /** Accepted by the API but not yet echoed back as message.user. */
  pending: boolean;
  /** Set when this message answers a question. */
  questionId: string | null;
  ts: string;
  seq: number;
}

export interface ChatQuestion {
  kind: "question";
  /** question_id. */
  id: string;
  question: string;
  choices: string[];
  /** message_id of the answer, once message.user with this question_id arrives. */
  answerId: string | null;
  /** The session ended before the question was answered. */
  expired: boolean;
  ts: string;
  seq: number;
}

export type ChatItem = ChatMessage | ChatQuestion;

export type StepStatus = "running" | "ok" | "error" | "interrupted";

export interface ToolStep {
  callId: string;
  tool: string;
  input: unknown;
  /** The shell this call runs in, from its input or its output. */
  shellId: string | null;
  status: StepStatus;
  startedAt: string;
  finishedAt: string | null;
  durationMs: number | null;
  exitCode: number | null;
  timedOut: boolean;
  summary: string;
  /** Output of a call that is not tied to a shell, capped. */
  output: string;
  seq: number;
}

export interface ShellTab {
  id: string;
  createdAt: string;
  /** The retained tail of the shell's output. */
  output: string;
  /** Characters dropped from the head of `output` to stay under the cap. */
  dropped: number;
  /** The call that last wrote to this shell. */
  lastCallId: string | null;
}

export interface FileChange {
  path: string;
  repo: string | null;
  action: FileActionName;
  ts: string;
  seq: number;
}

export interface PullRequest {
  repo: string;
  number: number;
  url: string;
  title: string;
  branch: string;
}

export interface SessionError {
  seq: number;
  code: string;
  message: string;
  retryable: boolean;
  ts: string;
}

export type ConnectionState =
  | { state: "idle" }
  | { state: "connecting"; attempt: number }
  | { state: "open" }
  | { state: "reconnecting"; attempt: number; delayMs: number }
  | { state: "closed"; reason: "unauthenticated" | "forbidden" | "not_found" | "stopped" };

export interface SessionState {
  sessionId: string;
  /** The highest stored seq applied; everything at or below it is applied. */
  lastSeq: number;
  /** Stored events received ahead of a gap, sorted by seq. */
  ahead: StreamEvent[];

  status: SessionStatusName | null;
  statusReason: string;
  /** seq of the session.status event that set `status`; 0 if it came from REST. */
  statusSeq: number;

  chat: ChatItem[];
  /** The open question's id, if the agent is waiting on one. */
  openQuestionId: string | null;

  steps: ToolStep[];
  shells: ShellTab[];
  files: FileChange[];
  /** Increments on every file.changed; the diff view refetches when it moves. */
  diffVersion: number;
  prs: PullRequest[];
  errors: SessionError[];
  /** Errors with seq at or below this are dismissed. */
  dismissedErrorSeq: number;

  connection: ConnectionState;
}

/** Shell output kept per tab, in characters. */
export const SHELL_OUTPUT_CAP = 1_000_000;
/** What a capped shell is trimmed back to. */
export const SHELL_OUTPUT_KEEP = 750_000;
/** Output kept for a call that has no shell. */
export const STEP_OUTPUT_CAP = 64_000;

export function initialSessionState(sessionId: string): SessionState {
  return {
    sessionId,
    lastSeq: 0,
    ahead: [],
    status: null,
    statusReason: "",
    statusSeq: 0,
    chat: [],
    openQuestionId: null,
    steps: [],
    shells: [],
    files: [],
    diffVersion: 0,
    prs: [],
    errors: [],
    dismissedErrorSeq: 0,
    connection: { state: "idle" },
  };
}

// --- Event application ---

/**
 * Applies one event received from the stream. Stored events are deduped by
 * seq and applied in seq order; message.delta (seq 0) applies immediately.
 * A skipped event takes its place in seq order but changes nothing else.
 * Events for another session are ignored.
 */
export function applyEvent(state: SessionState, event: StreamEvent): SessionState {
  if (!sameId(event.session_id, state.sessionId)) return state;

  if (event.seq === 0) {
    return isEphemeral(event.type) && !isSkipped(event) ? fold(state, event) : state;
  }
  if (isEphemeral(event.type)) return state; // an ephemeral type never carries a seq
  if (event.seq <= state.lastSeq) return state; // already applied
  if (event.seq > state.lastSeq + 1) {
    // Ahead of a gap: hold it until the missing events arrive.
    if (state.ahead.some((e) => e.seq === event.seq)) return state;
    const ahead = [...state.ahead, event].sort((a, b) => a.seq - b.seq);
    return { ...state, ahead };
  }

  let next = applyStored(state, event);
  // Drain whatever the gap was holding back.
  if (next.ahead.length > 0) {
    let ahead = next.ahead;
    while (ahead.length > 0 && ahead[0]!.seq <= next.lastSeq + 1) {
      const [head, ...rest] = ahead;
      ahead = rest;
      if (head!.seq === next.lastSeq + 1) next = applyStored(next, head!);
    }
    next = { ...next, ahead };
  }
  return next;
}

/** Applies the next stored event, the one at lastSeq + 1. */
function applyStored(state: SessionState, event: StreamEvent): SessionState {
  const next = isSkipped(event) ? state : fold(state, event);
  return { ...next, lastSeq: event.seq };
}

/** Applies events in order with applyEvent. */
export function applyEvents(state: SessionState, events: readonly StreamEvent[]): SessionState {
  let s = state;
  for (const e of events) s = applyEvent(s, e);
  return s;
}

/** True when stored events are waiting on a gap in seq. */
export function hasGap(state: SessionState): boolean {
  return state.ahead.length > 0;
}

/**
 * Whether two session IDs name the same session. The server sends IDs in
 * canonical form (lower-case, hyphenated), but the view's ID comes from the
 * URL, which may spell the same UUID differently.
 */
function sameId(a: string, b: string): boolean {
  return a === b || canonicalUuid(a) === canonicalUuid(b);
}

/**
 * The canonical form of a UUID in any spelling Go's uuid.Parse accepts
 * (any case, braces, a urn:uuid: prefix, no hyphens); other strings as
 * they are.
 */
function canonicalUuid(id: string): string {
  let s = id.toLowerCase();
  if (s.startsWith("urn:uuid:")) s = s.slice("urn:uuid:".length);
  else if (s.startsWith("{") && s.endsWith("}")) s = s.slice(1, -1);
  if (/^[0-9a-f]{32}$/.test(s)) {
    s = `${s.slice(0, 8)}-${s.slice(8, 12)}-${s.slice(12, 16)}-${s.slice(16, 20)}-${s.slice(20)}`;
  }
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(s) ? s : id;
}

function fold(state: SessionState, e: SessionEvent): SessionState {
  switch (e.type) {
    case "session.status": {
      const { status, reason } = e.payload;
      if (!isSessionStatus(status)) return state;
      let next: SessionState = { ...state, status, statusReason: reason ?? "", statusSeq: e.seq };
      if (isTerminal(status)) next = settleOnTerminal(next);
      return next;
    }

    case "message.user": {
      const p = e.payload;
      const msg: ChatMessage = {
        kind: "message",
        id: p.message_id,
        role: "user",
        userId: p.user_id || null,
        text: p.text,
        streaming: false,
        pending: false,
        questionId: p.question_id || null,
        ts: e.ts,
        seq: e.seq,
      };
      let chat = upsertMessage(state.chat, msg, () => msg);
      let openQuestionId = state.openQuestionId;
      if (msg.questionId) {
        chat = chat.map((item) =>
          item.kind === "question" && item.id === msg.questionId ? { ...item, answerId: msg.id } : item,
        );
        if (openQuestionId === msg.questionId) openQuestionId = null;
      }
      return { ...state, chat, openQuestionId };
    }

    case "message.assistant": {
      const p = e.payload;
      const msg: ChatMessage = {
        kind: "message",
        id: p.message_id,
        role: "assistant",
        userId: null,
        text: p.text,
        streaming: false,
        pending: false,
        questionId: null,
        ts: e.ts,
        seq: e.seq,
      };
      // The final text replaces whatever the deltas built up.
      return { ...state, chat: upsertMessage(dropAbandonedStreams(state.chat, msg.id), msg, () => msg) };
    }

    case "message.delta": {
      const p = e.payload;
      const i = findMessage(state.chat, p.message_id);
      if (i === -1) {
        const msg: ChatMessage = {
          kind: "message",
          id: p.message_id,
          role: "assistant",
          userId: null,
          text: p.text,
          streaming: true,
          pending: false,
          questionId: null,
          ts: e.ts,
          seq: 0,
        };
        return { ...state, chat: [...state.chat, msg] };
      }
      const existing = state.chat[i] as ChatMessage;
      if (!existing.streaming) return state; // already final: a late delta changes nothing
      const chat = state.chat.slice();
      chat[i] = { ...existing, text: existing.text + p.text };
      return { ...state, chat };
    }

    case "tool_call.started": {
      const p = e.payload;
      if (state.steps.some((s) => s.callId === p.call_id)) return state;
      const chat = dropAbandonedStreams(state.chat);
      const step: ToolStep = {
        callId: p.call_id,
        tool: p.tool,
        input: p.input,
        shellId: shellIdFromInput(p.input),
        status: "running",
        startedAt: e.ts,
        finishedAt: null,
        durationMs: null,
        exitCode: null,
        timedOut: false,
        summary: "",
        output: "",
        seq: e.seq,
      };
      return { ...state, chat, steps: [...state.steps, step] };
    }

    case "tool_call.output": {
      const p = e.payload;
      const si = state.steps.findIndex((s) => s.callId === p.call_id);
      const step = si === -1 ? undefined : state.steps[si];
      const shellId = p.shell_id || step?.shellId || null;
      if (shellId) {
        let shells = state.shells;
        let ti = shells.findIndex((t) => t.id === shellId);
        if (ti === -1) {
          shells = [...shells, newShell(shellId, e.ts)];
          ti = shells.length - 1;
        } else {
          shells = shells.slice();
        }
        shells[ti] = appendShellOutput(shells[ti]!, p.chunk, p.call_id);
        let steps = state.steps;
        if (step && step.shellId !== shellId) {
          steps = steps.slice();
          steps[si] = { ...step, shellId };
        }
        return { ...state, shells, steps };
      }
      if (!step) return state;
      const steps = state.steps.slice();
      steps[si] = { ...step, output: capTail(step.output + p.chunk, STEP_OUTPUT_CAP) };
      return { ...state, steps };
    }

    case "tool_call.finished": {
      const p = e.payload;
      const si = state.steps.findIndex((s) => s.callId === p.call_id);
      const base: ToolStep =
        si === -1
          ? {
              callId: p.call_id,
              tool: p.tool,
              input: null,
              shellId: null,
              status: "running",
              startedAt: e.ts,
              finishedAt: null,
              durationMs: null,
              exitCode: null,
              timedOut: false,
              summary: "",
              output: "",
              seq: e.seq,
            }
          : state.steps[si]!;
      const done: ToolStep = {
        ...base,
        status: p.is_error || p.timed_out ? "error" : "ok",
        finishedAt: e.ts,
        durationMs: p.duration_ms,
        exitCode: p.exit_code ?? null,
        timedOut: p.timed_out ?? false,
        summary: p.summary ?? "",
      };
      const steps = state.steps.slice();
      if (si === -1) steps.push(done);
      else steps[si] = done;
      return { ...state, steps };
    }

    case "shell.created": {
      const p = e.payload;
      if (state.shells.some((t) => t.id === p.shell_id)) return state;
      return { ...state, shells: [...state.shells, newShell(p.shell_id, e.ts)] };
    }

    case "file.changed": {
      const p = e.payload;
      const change: FileChange = {
        path: p.path,
        repo: p.repo || null,
        action: (["created", "modified", "deleted"].includes(p.action) ? p.action : "modified") as FileActionName,
        ts: e.ts,
        seq: e.seq,
      };
      const i = state.files.findIndex((f) => f.path === change.path && f.repo === change.repo);
      const files = state.files.slice();
      if (i === -1) {
        files.push(change);
      } else {
        // A file created in this session stays "created" until it is deleted.
        const prev = files[i]!;
        const action = prev.action === "created" && change.action === "modified" ? "created" : change.action;
        files[i] = { ...change, action };
      }
      return { ...state, files, diffVersion: state.diffVersion + 1 };
    }

    case "question.asked": {
      const p = e.payload;
      if (state.chat.some((c) => c.kind === "question" && c.id === p.question_id)) return state;
      const q: ChatQuestion = {
        kind: "question",
        id: p.question_id,
        question: p.question,
        choices: p.choices ?? [],
        answerId: null,
        expired: false,
        ts: e.ts,
        seq: e.seq,
      };
      return { ...state, chat: [...state.chat, q], openQuestionId: q.id };
    }

    case "pr.opened": {
      const p = e.payload;
      const pr: PullRequest = { repo: p.repo, number: p.number, url: p.url, title: p.title, branch: p.branch };
      const prs = state.prs.filter((x) => x.repo !== pr.repo);
      prs.push(pr);
      return { ...state, prs };
    }

    case "error": {
      const p = e.payload;
      const err: SessionError = {
        seq: e.seq,
        code: p.code ?? "",
        message: p.message,
        retryable: p.retryable ?? false,
        ts: e.ts,
      };
      return { ...state, errors: [...state.errors, err] };
    }
  }
}

/**
 * When a session ends, running steps stop, a message still streaming stops
 * where it got to (the run was cut short), and an open question lapses.
 */
function settleOnTerminal(state: SessionState): SessionState {
  const steps = state.steps.some((s) => s.status === "running")
    ? state.steps.map((s) => (s.status === "running" ? { ...s, status: "interrupted" as const } : s))
    : state.steps;
  const qid = state.openQuestionId;
  const chat = state.chat.map((c) => {
    if (c.kind === "message") return c.streaming ? { ...c, streaming: false } : c;
    return c.id === qid ? { ...c, expired: true } : c;
  });
  return { ...state, steps, chat, openQuestionId: null };
}

/**
 * Drops assistant messages still streaming, other than `keep`. A model call
 * streams its deltas, then stores its message.assistant (if it wrote text),
 * then runs its tools. So once a later message.assistant or tool call
 * arrives, a message still streaming will never get its final text: it was
 * a model call that failed and was retried, or was cut short. Its deltas
 * were never stored, so a reload would not show it either.
 */
function dropAbandonedStreams(chat: ChatItem[], keep?: string): ChatItem[] {
  const abandoned = (c: ChatItem) => c.kind === "message" && c.streaming && c.id !== keep;
  return chat.some(abandoned) ? chat.filter((c) => !abandoned(c)) : chat;
}

function findMessage(chat: readonly ChatItem[], id: string): number {
  return chat.findIndex((c) => c.kind === "message" && c.id === id);
}

function upsertMessage(chat: ChatItem[], msg: ChatMessage, merge: (prev: ChatMessage) => ChatMessage): ChatItem[] {
  const i = findMessage(chat, msg.id);
  if (i === -1) return [...chat, msg];
  const next = chat.slice();
  next[i] = merge(chat[i] as ChatMessage);
  return next;
}

function shellIdFromInput(input: unknown): string | null {
  if (typeof input === "object" && input !== null && "shell_id" in input) {
    const v = (input as { shell_id: unknown }).shell_id;
    if (typeof v === "string" && v) return v;
  }
  return null;
}

function newShell(id: string, ts: string): ShellTab {
  return { id, createdAt: ts, output: "", dropped: 0, lastCallId: null };
}

function appendShellOutput(tab: ShellTab, chunk: string, callId: string): ShellTab {
  let output = tab.output + chunk;
  let dropped = tab.dropped;
  if (output.length > SHELL_OUTPUT_CAP) {
    const cut = output.length - SHELL_OUTPUT_KEEP;
    output = output.slice(cut);
    dropped += cut;
  }
  return { ...tab, output, dropped, lastCallId: callId };
}

function capTail(s: string, cap: number): string {
  return s.length > cap ? s.slice(s.length - cap) : s;
}

// --- Local actions ---

export type SessionAction =
  | { type: "events"; events: StreamEvent[] }
  /** The session detail from REST; it only fills in what events have not. */
  | { type: "session_loaded"; session: Session }
  /** A message or answer the API accepted, shown as pending until its event arrives. */
  | { type: "message_sent"; messageId: string; text: string; ts: string; questionId?: string }
  | { type: "connection"; connection: ConnectionState }
  | { type: "dismiss_errors" };

export function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "events":
      return applyEvents(state, action.events);

    case "session_loaded": {
      const s = action.session;
      let next = state;
      if (next.statusSeq === 0 && isSessionStatus(s.status)) {
        next = { ...next, status: s.status };
      }
      const known = new Set(next.prs.map((p) => p.repo));
      const fromRest = s.repos
        .filter((r) => r.pr_url && r.pr_number !== null && !known.has(r.full_name))
        .map<PullRequest>((r) => ({
          repo: r.full_name,
          number: r.pr_number ?? 0,
          url: r.pr_url ?? "",
          title: "",
          branch: r.branch,
        }));
      if (fromRest.length > 0) next = { ...next, prs: [...next.prs, ...fromRest] };
      return next;
    }

    case "message_sent": {
      if (findMessage(state.chat, action.messageId) !== -1) return state; // its event already arrived
      const msg: ChatMessage = {
        kind: "message",
        id: action.messageId,
        role: "user",
        userId: null, // the viewer's own, until its message.user says so
        text: action.text,
        streaming: false,
        pending: true,
        questionId: action.questionId ?? null,
        ts: action.ts,
        seq: 0,
      };
      return { ...state, chat: [...state.chat, msg] };
    }

    case "connection":
      return { ...state, connection: action.connection };

    case "dismiss_errors": {
      const last = state.errors[state.errors.length - 1];
      return last ? { ...state, dismissedErrorSeq: last.seq } : state;
    }
  }
}

// --- Selectors ---

export function openQuestion(state: SessionState): ChatQuestion | null {
  if (!state.openQuestionId) return null;
  const q = state.chat.find((c) => c.kind === "question" && c.id === state.openQuestionId);
  return (q as ChatQuestion | undefined) ?? null;
}

/** The answer the user sent for the open question, while it is still pending. */
export function pendingAnswer(state: SessionState): ChatMessage | null {
  const q = state.openQuestionId;
  if (!q) return null;
  const m = state.chat.find((c) => c.kind === "message" && c.pending && c.questionId === q);
  return (m as ChatMessage | undefined) ?? null;
}

/** The newest error that has not been dismissed. */
export function visibleError(state: SessionState): SessionError | null {
  const last = state.errors[state.errors.length - 1];
  return last && last.seq > state.dismissedErrorSeq ? last : null;
}

/** The step currently running, if any (the newest one). */
export function currentStep(state: SessionState): ToolStep | null {
  for (let i = state.steps.length - 1; i >= 0; i--) {
    if (state.steps[i]!.status === "running") return state.steps[i]!;
  }
  return null;
}
