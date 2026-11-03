import { describe, expect, it } from "vitest";
import { GENERATED_FILE_ACTIONS, GENERATED_STATUSES, SESSION_STATUSES, isSkipped, isTerminal, parseEvent } from "./events";

describe("generated types", () => {
  it("lists the same session statuses as internal/events", () => {
    expect([...SESSION_STATUSES]).toEqual(GENERATED_STATUSES);
  });

  it("knows every file action", () => {
    expect(GENERATED_FILE_ACTIONS).toEqual(["created", "modified", "deleted"]);
  });

  it("treats the four end states as terminal", () => {
    expect(SESSION_STATUSES.filter((s) => isTerminal(s))).toEqual(["completed", "failed", "cancelled", "no_changes"]);
    expect(isTerminal(null)).toBe(false);
  });
});

describe("parseEvent", () => {
  const ok = {
    session_id: "s",
    seq: 3,
    ts: "2026-11-02T10:15:03Z",
    type: "tool_call.started",
    payload: { call_id: "c", tool: "shell_exec", input: { command: "ls" } },
  };

  it("accepts a well-formed event", () => {
    expect(parseEvent(JSON.stringify(ok))).toEqual(ok);
    expect(parseEvent(JSON.stringify({ ...ok, seq: 0, type: "message.delta", payload: { message_id: "m", text: "x" } }))).not.toBeNull();
  });

  it("keeps only the envelope fields", () => {
    expect(parseEvent(JSON.stringify({ ...ok, skipped: true, extra: 1 }))).toEqual(ok);
  });

  it("rejects malformed frames, and live events it can't use", () => {
    for (const bad of [
      "",
      "null",
      "[]",
      "{",
      JSON.stringify({ ...ok, seq: 1.5 }),
      JSON.stringify({ ...ok, seq: "3" }),
      JSON.stringify({ ...ok, seq: -1 }),
      JSON.stringify({ ...ok, ts: undefined }),
      JSON.stringify({ ...ok, type: 7 }),
      JSON.stringify({ ...ok, session_id: undefined }),
      JSON.stringify({ ...ok, seq: 0, type: "message.thinking" }),
      JSON.stringify({ ...ok, seq: 0, type: "message.delta", payload: null }),
    ]) {
      expect(parseEvent(bad)).toBeNull();
    }
  });

  it("turns a stored event it can't use into a skip that keeps its seq", () => {
    for (const frame of [{ ...ok, type: "routine.fired" }, { ...ok, payload: null }, { ...ok, payload: undefined }]) {
      const e = parseEvent(JSON.stringify(frame));
      expect(e).toEqual({ session_id: "s", seq: 3, ts: ok.ts, type: frame.type, skipped: true });
      expect(isSkipped(e!)).toBe(true);
    }
    expect(isSkipped(parseEvent(JSON.stringify(ok))!)).toBe(false);
  });
});
