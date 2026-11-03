// @vitest-environment happy-dom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SID = "3c4b0000-0000-4000-8000-000000000001";

const h = vi.hoisted(() => {
  interface Sock {
    onopen: (() => void) | null;
    onmessage: ((d: string) => void) | null;
    onclose: ((c: number, r: string) => void) | null;
    close(): void;
    afterSeq: number;
  }
  const sockets: Sock[] = [];
  const detail = {
    id: "3c4b0000-0000-4000-8000-000000000001",
    org_id: "o1",
    status: "running",
    prompt: "Fix it",
    issue_url: null,
    provider: "anthropic",
    model: "m",
    created_by: null,
    cancel_requested: false,
    created_at: "2026-11-02T10:00:00Z",
    finished_at: null,
    repos: [],
  };
  const me = {
    user: { id: "u_me", login: "octocat", name: "Mona", avatar_url: "" },
    orgs: [{ id: "o1", name: "acme", role: "owner" }],
  };
  type Send = (...args: unknown[]) => Promise<{ message_id: string }>;
  const accept: Send = async () => ({ message_id: "m_1" });
  return { sockets, detail, me, stopCalls: 0, accept, sendMessage: accept, answer: accept, calls: [] as unknown[][] };
});

vi.mock("@/lib/api/client", () => {
  class ApiError extends Error {
    status: number;
    code: string;
    fields: Record<string, string> = {};
    constructor(status: number, code: string, message: string) {
      super(message);
      this.status = status;
      this.code = code;
    }
  }
  const impl = {
    me: async () => h.me,
    logout: async () => {},
    getSession: async () => ({ ...h.detail }),
    stop: async () => {
      h.stopCalls++;
    },
    sendMessage: (...args: unknown[]) => {
      h.calls.push(["sendMessage", ...args]);
      return h.sendMessage(...args);
    },
    answer: (...args: unknown[]) => {
      h.calls.push(["answer", ...args]);
      return h.answer(...args);
    },
    diff: async () => ({ repos: [] }),
    openEvents: (_id: string, afterSeq: number) => {
      const s = { onopen: null, onmessage: null, onclose: null, close() {}, afterSeq };
      h.sockets.push(s);
      return s;
    },
  };
  return {
    api: impl,
    getApi: async () => impl,
    isApiError: (e: unknown) => e instanceof ApiError,
    ApiError,
    loginUrl: (n?: string) => n ?? "/sessions",
    isMock: false,
  };
});
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace() {}, push() {} }),
  usePathname: () => "/sessions/x",
}));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: unknown }) =>
    createElement("a", { href: String(href), ...rest }, children as never),
}));
// xterm and the diff renderer need a real layout engine; neither matters here.
vi.mock("@/components/session/terminal", () => ({ TerminalView: () => null }));
vi.mock("@/components/session/diff-view", () => ({ DiffView: () => null }));

const { AppShell } = await import("@/components/app-shell");
const { ApiError } = await import("@/lib/api/client");
const { SessionView } = await import("@/components/session/session-view");

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
const frame = (seq: number, type: string, payload: unknown) =>
  JSON.stringify({ session_id: SID, seq, ts: "2026-11-02T10:00:00Z", type, payload });

function buttons(text: string): HTMLButtonElement[] {
  return [...document.querySelectorAll("button")].filter((b) => b.textContent?.trim() === text);
}

let root: Root;
let host: HTMLDivElement;

beforeEach(async () => {
  window.matchMedia = ((q: string) => ({
    matches: q.includes("1024"), // the two-column layout
    media: q,
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
  h.sockets.length = 0;
  h.stopCalls = 0;
  h.sendMessage = h.accept;
  h.answer = h.accept;
  h.calls.length = 0;
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root.render(createElement(AppShell, null, createElement(SessionView, { id: SID })));
    await wait(10);
  });
  const sock = h.sockets.at(-1)!;
  await act(async () => {
    sock.onopen!();
    sock.onmessage!(frame(1, "message.user", { message_id: "m_keithprompt", user_id: "u_me", text: "Fix it" }));
    sock.onmessage!(frame(2, "session.status", { status: "running" }));
    await wait(40);
  });
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

/** Delivers frames on the open socket and lets the batch render. */
async function send(...frames: string[]) {
  const sock = h.sockets.at(-1)!;
  await act(async () => {
    for (const f of frames) sock.onmessage!(f);
    await wait(40);
  });
}

async function click(el: HTMLElement) {
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await wait(10);
  });
}

describe("StopControl", () => {
  it("focuses 'Stop run' when the confirm opens, and leaves focus alone after that", async () => {
    await click(buttons("Stop")[0]!);
    expect(document.activeElement).toBe(buttons("Stop run")[0]);

    // The user goes back to the composer while the agent keeps streaming.
    const textarea = document.querySelector("textarea")!;
    await act(async () => textarea.focus());
    await send(
      frame(0, "message.delta", { message_id: "msg_1", text: "Looking at " }),
      frame(0, "message.delta", { message_id: "msg_1", text: "the tests" }),
    );
    await send(frame(3, "tool_call.started", { call_id: "toolu_1", tool: "read_file", input: { path: "a.ts" } }));
    expect(buttons("Stop run")).toHaveLength(1);
    expect(document.activeElement).toBe(textarea);
  });

  it("can stop a run that a follow-up resumed after an earlier stop", async () => {
    await click(buttons("Stop")[0]!);
    await click(buttons("Stop run")[0]!);
    expect(h.stopCalls).toBe(1);
    expect(buttons("Stopping…")).toHaveLength(1);

    await send(frame(3, "session.status", { status: "cancelled", reason: "Stopped by user" }));
    expect(buttons("Stopping…")).toHaveLength(0); // hidden while terminal
    expect(buttons("Stop")).toHaveLength(0);

    // A follow-up resumes the run; the worker cleared cancel_requested_at.
    await send(
      frame(4, "message.user", { message_id: "m_follow", text: "Actually, carry on" }),
      frame(5, "session.status", { status: "running" }),
    );
    expect(buttons("Stopping…")).toHaveLength(0);
    expect(buttons("Stop")).toHaveLength(1);

    await click(buttons("Stop")[0]!);
    await click(buttons("Stop run")[0]!);
    expect(h.stopCalls).toBe(2);
    expect(buttons("Stopping…")).toHaveLength(1);
  });
});

describe("Chat", () => {
  /** The name above each chat message, in order. */
  const authors = () =>
    [...document.querySelectorAll('[role="log"] article > header > span:first-child')].map((el) => el.textContent);

  it("says who wrote each message, using /api/me", async () => {
    await send(
      frame(3, "message.assistant", { message_id: "msg_1", text: "Running the tests." }),
      frame(4, "message.user", { message_id: "m_keith0123", text: "The automatic test run failed:" }),
      frame(5, "message.user", { message_id: "m_9", user_id: "u_other", text: "Use dayjs" }),
      frame(6, "message.user", { message_id: "m_10", user_id: "u_me", text: "Thanks" }),
    );
    expect(authors()).toEqual(["You", "Keith", "Keith", "A teammate", "You"]);
  });
});

describe("Composer errors", () => {
  const textarea = () => document.querySelector<HTMLTextAreaElement>("form textarea")!;
  const sendButton = () => document.querySelector<HTMLButtonElement>('button[aria-label="Send"]')!;
  const chatText = () => document.querySelector('[role="log"]')?.textContent ?? "";
  const alerts = () => [...document.querySelectorAll('[role="alert"]')].map((el) => el.textContent ?? "");
  const newLinks = () => [...document.querySelectorAll<HTMLAnchorElement>('a[href="/new"]')];

  /** Types into a React-controlled textarea. */
  async function type(el: HTMLTextAreaElement, value: string) {
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(el, value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  const reject = (status: number, code: string, message: string) => async () => {
    throw new ApiError(status, code, message);
  };

  const lost = reject(
    409,
    "repo_access_lost",
    "acme/web is no longer available to this organization, so this session cannot continue. Start a new session.",
  );
  const capped = reject(429, "rate_limited", "This organization already has 5 active sessions. Wait for one to finish.");

  async function ask() {
    await send(
      frame(3, "question.asked", { question_id: "toolu_q", question: "Which key?", choices: ["Sandbox", "Live"] }),
      frame(4, "session.status", { status: "waiting_for_user" }),
    );
  }

  it("locks the composer and links to /new when a follow-up gets repo_access_lost", async () => {
    await send(frame(3, "session.status", { status: "completed" }));
    h.sendMessage = lost;
    await type(textarea(), "Also add a test");
    await click(sendButton());

    expect(h.calls).toEqual([["sendMessage", SID, "Also add a test"]]);
    expect(alerts().join(" ")).toContain("This session can't continue");
    expect(alerts().join(" ")).toContain("One of its repos is no longer available to this organization.");
    expect(newLinks().map((a) => a.textContent)).toContain("Start a new session");
    // Nothing pending or optimistic: the text stays put and is not in the chat.
    expect(chatText()).not.toContain("Also add a test");
    expect(textarea().value).toBe("Also add a test");
    expect(textarea().disabled).toBe(true);
    expect(sendButton().disabled).toBe(true);
    expect(sendButton().getAttribute("aria-busy")).toBeNull();
  });

  it("locks the answer form when an answer gets repo_access_lost", async () => {
    await ask();
    h.answer = lost;
    await click(buttons("Sandbox")[0]!);

    expect(h.calls).toEqual([["answer", SID, "toolu_q", "Sandbox"]]);
    expect(alerts().join(" ")).toContain("This session can't continue");
    expect(newLinks()).not.toHaveLength(0);
    expect(document.body.textContent).not.toContain("Answer sent");
    expect(buttons("Sandbox")[0]!.disabled).toBe(true);
    expect(buttons("Sandbox")[0]!.getAttribute("aria-busy")).toBeNull();
    expect(chatText()).toContain("Which key?"); // the question stays open
  });

  it("shows a rate limit on a follow-up and keeps the text for a retry", async () => {
    await send(frame(3, "session.status", { status: "completed" }));
    h.sendMessage = capped;
    await type(textarea(), "Try again please");
    await click(sendButton());

    expect(alerts()).toContain("This organization already has 5 active sessions. Wait for one to finish.");
    expect(alerts().join(" ")).not.toContain("can't continue");
    expect(chatText()).not.toContain("Try again please");
    expect(textarea().value).toBe("Try again please");
    expect(textarea().disabled).toBe(false);
    expect(sendButton().disabled).toBe(false);
    expect(sendButton().getAttribute("aria-busy")).toBeNull();

    // Once there's room, the same text goes through and shows as pending.
    h.sendMessage = h.accept;
    await click(sendButton());
    expect(alerts()).toEqual([]);
    expect(chatText()).toContain("Try again please");
    expect(textarea().value).toBe("");
  });

  it("shows a rate limit on an answer and leaves the question open", async () => {
    await ask();
    h.answer = capped;
    await click(buttons("Sandbox")[0]!);

    expect(alerts()).toContain("This organization already has 5 active sessions. Wait for one to finish.");
    expect(document.body.textContent).not.toContain("Answer sent");
    expect(buttons("Sandbox")[0]!.disabled).toBe(false);
    expect(buttons("Live")[0]!.disabled).toBe(false);
  });
});
