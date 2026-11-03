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
  return { sockets, detail, me, stopCalls: 0 };
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
    sendMessage: async () => ({ message_id: "m_1" }),
    answer: async () => ({ message_id: "m_2" }),
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
