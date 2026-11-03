import { describe, expect, it } from "vitest";
import type { ChatMessage } from "@/lib/session/reducer";
import { messageAuthor, type ChatViewer } from "./format";

function msg(over: Partial<ChatMessage>): ChatMessage {
  return {
    kind: "message",
    id: "m_1",
    role: "user",
    userId: null,
    text: "x",
    streaming: false,
    pending: false,
    questionId: null,
    ts: "2026-11-02T10:00:00Z",
    seq: 5,
    ...over,
  };
}

const viewer: ChatViewer = { meId: "u_me", creator: { id: "u_mona", login: "mona", avatar_url: "" } };

describe("messageAuthor", () => {
  it("names the agent Keith", () => {
    expect(messageAuthor(msg({ role: "assistant", id: "msg_1" }), viewer)).toBe("Keith");
  });

  it("tells the viewer's own messages from other people's", () => {
    expect(messageAuthor(msg({ userId: "u_me" }), viewer)).toBe("You");
    expect(messageAuthor(msg({ userId: "u_mona" }), viewer)).toBe("mona");
    expect(messageAuthor(msg({ userId: "u_sam" }), viewer)).toBe("A teammate");
    // Before /api/me loads, nobody is "You".
    expect(messageAuthor(msg({ userId: "u_me" }), { meId: null, creator: null })).toBe("A teammate");
  });

  it("calls a message this page sent You while it is pending", () => {
    expect(messageAuthor(msg({ pending: true, seq: 0 }), viewer)).toBe("You");
  });

  it("labels the worker's notes to the agent as Keith", () => {
    expect(messageAuthor(msg({ id: "m_keith0a1b2c3d4e5f6a7b" }), viewer)).toBe("Keith");
  });

  it("labels the prompt of a session no person started as from a trigger", () => {
    expect(messageAuthor(msg({ id: "m_keithprompt", seq: 1 }), viewer)).toBe("Trigger");
    expect(messageAuthor(msg({ id: "m_keithprompt", seq: 1, userId: "u_me" }), viewer)).toBe("You");
    expect(messageAuthor(msg({ id: "m_01J" }), viewer)).toBe("Trigger");
  });
});
