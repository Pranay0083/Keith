import { describe, expect, it } from "vitest";
import { parseEvent, type EventPayloadMap, type EventType, type SessionEvent, type StreamEvent } from "@/lib/api/events";
import type { Session } from "@/lib/api/types";
import {
  applyEvent,
  applyEvents,
  currentStep,
  hasGap,
  initialSessionState,
  openQuestion,
  pendingAnswer,
  SHELL_OUTPUT_CAP,
  SHELL_OUTPUT_KEEP,
  sessionReducer,
  visibleError,
  type ChatMessage,
  type ChatQuestion,
  type SessionState,
} from "./reducer";

const SID = "s_1";

function ev<K extends EventType>(seq: number, type: K, payload: EventPayloadMap[K], sessionId = SID): SessionEvent {
  return {
    session_id: sessionId,
    seq,
    ts: `2026-11-02T10:00:${String(seq % 60).padStart(2, "0")}Z`,
    type,
    payload,
  } as SessionEvent;
}

const fresh = (): SessionState => initialSessionState(SID);

function messages(s: SessionState): ChatMessage[] {
  return s.chat.filter((c): c is ChatMessage => c.kind === "message");
}

describe("seq ordering and dedupe", () => {
  it("applies stored events in seq order even when they arrive out of order", () => {
    const e1 = ev(1, "session.status", { status: "provisioning" });
    const e2 = ev(2, "session.status", { status: "setting_up" });
    const e3 = ev(3, "session.status", { status: "running" });

    let s = applyEvent(fresh(), e3);
    expect(s.lastSeq).toBe(0);
    expect(s.status).toBeNull();
    expect(hasGap(s)).toBe(true);

    s = applyEvent(s, e1);
    expect(s.lastSeq).toBe(1);
    expect(s.status).toBe("provisioning");
    expect(hasGap(s)).toBe(true);

    s = applyEvent(s, e2);
    expect(s.lastSeq).toBe(3);
    expect(s.status).toBe("running");
    expect(s.statusSeq).toBe(3);
    expect(s.ahead).toEqual([]);
    expect(hasGap(s)).toBe(false);
  });

  it("drops a stored event it has already applied", () => {
    const events = [
      ev(1, "message.user", { message_id: "m1", text: "Fix the test" }),
      ev(2, "tool_call.started", { call_id: "c1", tool: "read_file", input: { path: "/a" } }),
    ];
    const once = applyEvents(fresh(), events);
    const twice = applyEvents(once, events);
    expect(twice).toBe(once);
    expect(twice.chat).toHaveLength(1);
    expect(twice.steps).toHaveLength(1);
    expect(twice.lastSeq).toBe(2);
  });

  it("drops duplicates that arrive while waiting on a gap", () => {
    const e3 = ev(3, "error", { message: "boom" });
    let s = applyEvents(fresh(), [ev(1, "session.status", { status: "running" }), e3, e3]);
    expect(s.ahead).toHaveLength(1);
    s = applyEvent(s, ev(2, "session.status", { status: "running" }));
    expect(s.errors).toHaveLength(1);
    expect(s.lastSeq).toBe(3);
  });

  it("replays after reconnect without double-applying (after_seq overlap)", () => {
    const log = [
      ev(1, "message.user", { message_id: "m1", text: "hi" }),
      ev(2, "message.assistant", { message_id: "a1", text: "hello" }),
      ev(3, "file.changed", { path: "/w/a.ts", action: "modified" }),
    ];
    let s = applyEvents(fresh(), log.slice(0, 2));
    // The reconnect replays from an older seq, so 2 arrives again.
    s = applyEvents(s, log.slice(1));
    expect(s.chat).toHaveLength(2);
    expect(s.diffVersion).toBe(1);
    expect(s.lastSeq).toBe(3);
  });

  it("ignores events for another session", () => {
    const s = applyEvent(fresh(), ev(1, "session.status", { status: "running" }, "other"));
    expect(s).toEqual(fresh());
  });

  it("matches the canonical session ID the server sends however the URL spells it", () => {
    const uuid = "3c4b0000-0000-4000-8000-0000000000ab";
    for (const spelling of [
      uuid.toUpperCase(),
      `{${uuid}}`,
      `urn:uuid:${uuid}`,
      uuid.replaceAll("-", "").toUpperCase(),
    ]) {
      const s = applyEvent(initialSessionState(spelling), ev(1, "session.status", { status: "running" }, uuid));
      expect(s.lastSeq, spelling).toBe(1);
      expect(s.status, spelling).toBe("running");
    }
    // Other IDs still have to match exactly.
    const other = "3c4b0000-0000-4000-8000-0000000000ac";
    expect(applyEvent(initialSessionState(uuid), ev(1, "session.status", { status: "running" }, other)).lastSeq).toBe(0);
    expect(applyEvent(initialSessionState("Demo"), ev(1, "session.status", { status: "running" }, "demo")).lastSeq).toBe(0);
  });

  it("ignores a delta that claims a stored seq", () => {
    const s = applyEvent(fresh(), ev(1, "message.delta", { message_id: "a1", text: "x" }));
    expect(s.chat).toHaveLength(0);
    expect(s.lastSeq).toBe(0);
  });
});

describe("stored events this build can't read", () => {
  // Frames as the server sends them, through the same parser the stream uses.
  const frame = (seq: number, type: string, payload: unknown) =>
    parseEvent(JSON.stringify({ session_id: SID, seq, ts: "2026-11-02T10:00:00Z", type, payload }))!;
  const log: StreamEvent[] = [
    frame(1, "message.user", { message_id: "m_keithprompt", text: "Fix it" }),
    frame(2, "sandbox.ready", { id: "x" }), // a type newer than this client
    frame(3, "session.status", { status: "running" }),
    frame(4, "error", null), // a known type with no payload
    frame(5, "message.assistant", { message_id: "msg_1", text: "Done" }),
  ];

  it("skips them in seq order instead of waiting on their seq forever", () => {
    const s = applyEvents(fresh(), log);
    expect(s.lastSeq).toBe(5);
    expect(hasGap(s)).toBe(false); // so the gap watchdog never reconnects
    expect(s.status).toBe("running");
    expect(s.errors).toEqual([]);
    expect(messages(s).map((m) => m.id)).toEqual(["m_keithprompt", "msg_1"]);
  });

  it("holds one that arrives ahead of a gap, like any other stored event", () => {
    let s = applyEvents(fresh(), [log[2]!, log[1]!]);
    expect(s.lastSeq).toBe(0);
    expect(s.ahead.map((e) => e.seq)).toEqual([2, 3]);
    s = applyEvents(s, [log[0]!, ...log]); // the gap fills, then a resync replays it all
    expect(s.lastSeq).toBe(5);
    expect(hasGap(s)).toBe(false);
    expect(messages(s)).toHaveLength(2);
  });
});

describe("streaming assistant messages", () => {
  it("types deltas in place, then replaces them with the final message", () => {
    let s = applyEvents(fresh(), [
      ev(0, "message.delta", { message_id: "a1", text: "I'll start " }),
      ev(0, "message.delta", { message_id: "a1", text: "by reading" }),
    ]);
    expect(messages(s)).toHaveLength(1);
    expect(messages(s)[0]).toMatchObject({ id: "a1", role: "assistant", text: "I'll start by reading", streaming: true });
    expect(s.lastSeq).toBe(0);

    s = applyEvent(s, ev(1, "message.assistant", { message_id: "a1", text: "I'll start by reading the issue." }));
    expect(messages(s)).toHaveLength(1);
    expect(messages(s)[0]).toMatchObject({ text: "I'll start by reading the issue.", streaming: false, seq: 1 });
    expect(s.lastSeq).toBe(1);
  });

  it("ignores a delta that arrives after the final message", () => {
    let s = applyEvent(fresh(), ev(1, "message.assistant", { message_id: "a1", text: "Done." }));
    s = applyEvent(s, ev(0, "message.delta", { message_id: "a1", text: " extra" }));
    expect(messages(s)[0]?.text).toBe("Done.");
  });

  it("keeps interleaved streams for different messages apart", () => {
    const s = applyEvents(fresh(), [
      ev(0, "message.delta", { message_id: "a1", text: "one" }),
      ev(0, "message.delta", { message_id: "a2", text: "two" }),
      ev(0, "message.delta", { message_id: "a1", text: "!" }),
    ]);
    expect(messages(s).map((m) => [m.id, m.text])).toEqual([
      ["a1", "one!"],
      ["a2", "two"],
    ]);
  });

  it("drops a stream whose model call was retried once the retry's final message arrives", () => {
    let s = applyEvents(fresh(), [
      ev(1, "message.user", { message_id: "m_keithprompt", text: "Fix it" }),
      ev(2, "session.status", { status: "running" }),
      ev(0, "message.delta", { message_id: "a1", text: "I'll start by rea" }), // the call fails mid-stream
      ev(0, "message.delta", { message_id: "a2", text: "I'll start by reading" }), // the retry
    ]);
    expect(messages(s).filter((m) => m.streaming).map((m) => m.id)).toEqual(["a1", "a2"]);

    s = applyEvent(s, ev(3, "message.assistant", { message_id: "a2", text: "I'll start by reading the tests." }));
    expect(messages(s).map((m) => [m.id, m.streaming])).toEqual([
      ["m_keithprompt", false],
      ["a2", false],
    ]);
  });

  it("drops a stream that never finished once the agent runs a tool", () => {
    const s = applyEvents(fresh(), [
      ev(0, "message.delta", { message_id: "a1", text: "Let me" }), // the call fails mid-stream
      ev(1, "tool_call.started", { call_id: "toolu_1", tool: "read_file", input: { path: "a.ts" } }), // the retry wrote no text
    ]);
    expect(messages(s)).toEqual([]);
    expect(s.steps).toHaveLength(1);
  });

  it("stops a stream where it got to when the session ends", () => {
    let s = applyEvents(fresh(), [
      ev(0, "message.delta", { message_id: "a1", text: "Looking at the" }),
      ev(1, "session.status", { status: "cancelled", reason: "Stopped by user" }),
    ]);
    expect(messages(s)).toMatchObject([{ id: "a1", text: "Looking at the", streaming: false }]);
    s = applyEvent(s, ev(0, "message.delta", { message_id: "a1", text: " tests" }));
    expect(messages(s)[0]?.text).toBe("Looking at the");
  });

  it("shows a final message that had no deltas", () => {
    const s = applyEvent(fresh(), ev(1, "message.assistant", { message_id: "a9", text: "Opened the PR." }));
    expect(messages(s)[0]).toMatchObject({ role: "assistant", streaming: false });
  });
});

describe("user messages", () => {
  it("shows a sent message as pending until its message.user arrives", () => {
    let s = sessionReducer(fresh(), { type: "message_sent", messageId: "m2", text: "Use dayjs", ts: "t" });
    expect(messages(s)[0]).toMatchObject({ id: "m2", pending: true, seq: 0, userId: null });

    s = applyEvent(s, ev(1, "message.user", { message_id: "m2", text: "Use dayjs", user_id: "u1" }));
    expect(messages(s)).toHaveLength(1);
    expect(messages(s)[0]).toMatchObject({ id: "m2", pending: false, seq: 1, userId: "u1" });
  });

  it("does not duplicate a message whose event arrived before the API response", () => {
    let s = applyEvent(fresh(), ev(1, "message.user", { message_id: "m2", text: "Use dayjs" }));
    s = sessionReducer(s, { type: "message_sent", messageId: "m2", text: "Use dayjs", ts: "t" });
    expect(messages(s)).toHaveLength(1);
    expect(messages(s)[0]?.pending).toBe(false);
  });
});

describe("questions", () => {
  const asked = ev(5, "question.asked", {
    question_id: "q1",
    question: "Pin tests to UTC or make it timezone-aware?",
    choices: ["Pin tests to UTC", "Make it timezone-aware"],
  });
  const base = () => applyEvents(fresh(), [1, 2, 3, 4].map((n) => ev(n, "session.status", { status: "running" })));

  it("opens a question and closes it when the answer arrives", () => {
    let s = applyEvents(base(), [asked, ev(6, "session.status", { status: "waiting_for_user" })]);
    expect(openQuestion(s)).toMatchObject({ id: "q1", choices: ["Pin tests to UTC", "Make it timezone-aware"] });
    expect(s.status).toBe("waiting_for_user");

    s = sessionReducer(s, {
      type: "message_sent",
      messageId: "m3",
      text: "Make it timezone-aware",
      ts: "t",
      questionId: "q1",
    });
    expect(pendingAnswer(s)).toMatchObject({ id: "m3", pending: true });
    expect(openQuestion(s)).not.toBeNull();

    s = applyEvents(s, [
      ev(7, "message.user", { message_id: "m3", text: "Make it timezone-aware", question_id: "q1" }),
      ev(8, "session.status", { status: "running" }),
    ]);
    expect(openQuestion(s)).toBeNull();
    expect(pendingAnswer(s)).toBeNull();
    const q = s.chat.find((c): c is ChatQuestion => c.kind === "question");
    expect(q).toMatchObject({ id: "q1", answerId: "m3", expired: false });
    expect(messages(s).filter((m) => m.id === "m3")).toHaveLength(1);
    expect(messages(s).find((m) => m.id === "m3")?.questionId).toBe("q1");
  });

  it("lets an unanswered question lapse when the session ends", () => {
    const s = applyEvents(base(), [asked, ev(6, "session.status", { status: "cancelled", reason: "stopped by user" })]);
    expect(openQuestion(s)).toBeNull();
    expect(s.chat.find((c) => c.kind === "question")).toMatchObject({ expired: true, answerId: null });
    expect(s.statusReason).toBe("stopped by user");
  });

  it("does not open the same question twice", () => {
    const s = applyEvents(base(), [asked, { ...asked, seq: 6 }]);
    expect(s.chat.filter((c) => c.kind === "question")).toHaveLength(1);
  });
});

describe("tool calls and shells", () => {
  it("opens a terminal tab on shell.created and appends output to it", () => {
    const s = applyEvents(fresh(), [
      ev(1, "shell.created", { shell_id: "dev", call_id: "c1" }),
      ev(2, "tool_call.started", { call_id: "c2", tool: "shell_exec", input: { shell_id: "dev", command: "npm test" } }),
      ev(3, "tool_call.output", { call_id: "c2", shell_id: "dev", chunk: "> vitest run\r\n" }),
      ev(4, "tool_call.output", { call_id: "c2", shell_id: "dev", chunk: "\u001b[32m✓\u001b[39m 12 passed\r\n" }),
    ]);
    expect(s.shells.map((t) => t.id)).toEqual(["dev"]);
    expect(s.shells[0]).toMatchObject({
      output: "> vitest run\r\n\u001b[32m✓\u001b[39m 12 passed\r\n",
      dropped: 0,
      lastCallId: "c2",
    });
    expect(s.steps[0]).toMatchObject({ callId: "c2", shellId: "dev", status: "running" });
    expect(currentStep(s)?.callId).toBe("c2");
  });

  it("opens a tab for output to a shell it has not seen, and routes by the call's shell", () => {
    const s = applyEvents(fresh(), [
      ev(1, "tool_call.started", { call_id: "c1", tool: "shell_exec", input: { shell_id: "setup", command: "npm ci" } }),
      ev(2, "tool_call.output", { call_id: "c1", chunk: "added 812 packages\r\n" }),
    ]);
    expect(s.shells.map((t) => t.id)).toEqual(["setup"]);
    expect(s.shells[0]?.output).toBe("added 812 packages\r\n");
  });

  it("keeps output of a call without a shell on the step", () => {
    const s = applyEvents(fresh(), [
      ev(1, "tool_call.started", { call_id: "c1", tool: "search", input: { pattern: "Date" } }),
      ev(2, "tool_call.output", { call_id: "c1", chunk: "src/date.ts:4" }),
    ]);
    expect(s.shells).toHaveLength(0);
    expect(s.steps[0]?.output).toBe("src/date.ts:4");
  });

  it("does not create a second tab for the same shell", () => {
    const s = applyEvents(fresh(), [
      ev(1, "shell.created", { shell_id: "dev" }),
      ev(2, "shell.created", { shell_id: "dev" }),
      ev(3, "shell.created", { shell_id: "server" }),
    ]);
    expect(s.shells.map((t) => t.id)).toEqual(["dev", "server"]);
  });

  it("caps a shell's output and counts what it dropped", () => {
    const big = "x".repeat(SHELL_OUTPUT_CAP);
    const s = applyEvents(fresh(), [
      ev(1, "shell.created", { shell_id: "dev" }),
      ev(2, "tool_call.output", { call_id: "c1", shell_id: "dev", chunk: big }),
      ev(3, "tool_call.output", { call_id: "c1", shell_id: "dev", chunk: "tail" }),
    ]);
    const tab = s.shells[0]!;
    expect(tab.output.length).toBe(SHELL_OUTPUT_KEEP);
    expect(tab.dropped).toBe(SHELL_OUTPUT_CAP + 4 - SHELL_OUTPUT_KEEP);
    expect(tab.output.endsWith("tail")).toBe(true);
  });

  it("marks finished steps with duration, exit code and error state", () => {
    const s = applyEvents(fresh(), [
      ev(1, "tool_call.started", { call_id: "c1", tool: "shell_exec", input: { command: "npm test" } }),
      ev(2, "tool_call.finished", { call_id: "c1", tool: "shell_exec", is_error: false, exit_code: 1, duration_ms: 4210, summary: "2 failed" }),
      ev(3, "tool_call.started", { call_id: "c2", tool: "edit_file", input: { path: "/a" } }),
      ev(4, "tool_call.finished", { call_id: "c2", tool: "edit_file", is_error: true, duration_ms: 12, summary: "old_str not found" }),
      ev(5, "tool_call.started", { call_id: "c3", tool: "shell_exec", input: { command: "sleep 999" } }),
      ev(6, "tool_call.finished", { call_id: "c3", tool: "shell_exec", is_error: false, timed_out: true, duration_ms: 600000 }),
    ]);
    expect(s.steps.map((x) => [x.callId, x.status, x.exitCode, x.durationMs])).toEqual([
      ["c1", "ok", 1, 4210],
      ["c2", "error", null, 12],
      ["c3", "error", null, 600000],
    ]);
    expect(s.steps[0]?.summary).toBe("2 failed");
    expect(s.steps[2]?.timedOut).toBe(true);
    expect(currentStep(s)).toBeNull();
  });

  it("interrupts running steps when the session ends", () => {
    const s = applyEvents(fresh(), [
      ev(1, "tool_call.started", { call_id: "c1", tool: "shell_exec", input: {} }),
      ev(2, "session.status", { status: "cancelled" }),
    ]);
    expect(s.steps[0]?.status).toBe("interrupted");
  });
});

describe("files, PRs and errors", () => {
  it("bumps diffVersion per file change and keeps one row per file", () => {
    const s = applyEvents(fresh(), [
      ev(1, "file.changed", { repo: "acme/web", path: "/workspace/web/src/new.ts", action: "created" }),
      ev(2, "file.changed", { repo: "acme/web", path: "/workspace/web/src/new.ts", action: "modified" }),
      ev(3, "file.changed", { repo: "acme/web", path: "/workspace/web/src/old.ts", action: "deleted" }),
    ]);
    expect(s.diffVersion).toBe(3);
    expect(s.files.map((f) => [f.path, f.action])).toEqual([
      ["/workspace/web/src/new.ts", "created"],
      ["/workspace/web/src/old.ts", "deleted"],
    ]);
  });

  it("adds a PR card on pr.opened, one per repo", () => {
    const pr = { repo: "acme/web", number: 128, url: "https://github.com/acme/web/pull/128", title: "Fix date test", branch: "agent/s_1" };
    let s = applyEvent(fresh(), ev(1, "pr.opened", pr));
    expect(s.prs).toEqual([pr]);
    s = applyEvent(s, ev(2, "pr.opened", { ...pr, title: "Fix date test (v2)" }));
    expect(s.prs).toHaveLength(1);
    expect(s.prs[0]?.title).toBe("Fix date test (v2)");
    s = applyEvent(s, ev(3, "pr.opened", { ...pr, repo: "acme/api", number: 7, url: "https://github.com/acme/api/pull/7" }));
    expect(s.prs.map((p) => p.repo)).toEqual(["acme/web", "acme/api"]);
  });

  it("shows the newest error until it is dismissed", () => {
    let s = applyEvents(fresh(), [
      ev(1, "error", { code: "llm_rate_limited", message: "Rate limited", retryable: true }),
      ev(2, "error", { message: "Sandbox lost" }),
    ]);
    expect(visibleError(s)).toMatchObject({ seq: 2, message: "Sandbox lost", retryable: false, code: "" });
    s = sessionReducer(s, { type: "dismiss_errors" });
    expect(visibleError(s)).toBeNull();
    s = applyEvent(s, ev(3, "error", { message: "Again" }));
    expect(visibleError(s)?.seq).toBe(3);
  });
});

describe("REST seed", () => {
  const session: Session = {
    id: SID,
    org_id: "o1",
    status: "completed",
    prompt: "Fix it",
    issue_url: null,
    provider: "anthropic",
    model: "claude-opus-5-5",
    created_by: null,
    cancel_requested: false,
    created_at: "2026-11-02T10:00:00Z",
    finished_at: null,
    repos: [{ repo_id: "r1", full_name: "acme/web", branch: "agent/s_1", pr_number: 128, pr_url: "https://github.com/acme/web/pull/128" }],
  };

  it("fills status and PRs when no events have arrived", () => {
    const s = sessionReducer(fresh(), { type: "session_loaded", session });
    expect(s.status).toBe("completed");
    expect(s.prs).toEqual([
      { repo: "acme/web", number: 128, url: "https://github.com/acme/web/pull/128", title: "", branch: "agent/s_1" },
    ]);
  });

  it("never overrides what events already said", () => {
    let s = applyEvents(fresh(), [
      ev(1, "session.status", { status: "running" }),
      ev(2, "pr.opened", { repo: "acme/web", number: 128, url: "u", title: "T", branch: "b" }),
    ]);
    s = sessionReducer(s, { type: "session_loaded", session });
    expect(s.status).toBe("running");
    expect(s.prs).toHaveLength(1);
    expect(s.prs[0]?.title).toBe("T");
  });
});
