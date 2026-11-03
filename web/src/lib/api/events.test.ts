import { describe, expect, it } from "vitest";
import { GENERATED_FILE_ACTIONS, GENERATED_STATUSES, SESSION_STATUSES, isTerminal, parseEvent } from "./events";

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

  it("rejects malformed frames and unknown types", () => {
    for (const bad of [
      "",
      "null",
      "[]",
      "{",
      JSON.stringify({ ...ok, type: "routine.fired" }),
      JSON.stringify({ ...ok, seq: 1.5 }),
      JSON.stringify({ ...ok, seq: "3" }),
      JSON.stringify({ ...ok, payload: null }),
      JSON.stringify({ ...ok, session_id: undefined }),
    ]) {
      expect(parseEvent(bad)).toBeNull();
    }
  });
});
