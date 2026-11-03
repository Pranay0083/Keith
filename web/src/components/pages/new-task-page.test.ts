// @vitest-environment happy-dom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const h = vi.hoisted(() => ({
  me: {
    user: { id: "u_me", login: "octocat", name: "Mona", avatar_url: "" },
    orgs: [{ id: "o1", name: "acme", role: "owner" }],
  },
  repos: [
    { id: "r1", full_name: "acme/web", default_branch: "main", private: true, accessible: true, profile: null },
  ],
  createSession: null as null | ((...args: unknown[]) => Promise<{ id: string }>),
  pushed: [] as string[],
}));

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
    repos: async () => h.repos,
    providers: async () => null,
    createSession: (...args: unknown[]) => h.createSession!(...args),
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
  useRouter: () => ({
    replace() {},
    push(href: string) {
      h.pushed.push(href);
    },
  }),
  usePathname: () => "/new",
  useSearchParams: () => new URLSearchParams(),
}));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: unknown }) =>
    createElement("a", { href: String(href), ...rest }, children as never),
}));

const { AppShell } = await import("@/components/app-shell");
const { NewTaskPage } = await import("@/components/pages/new-task-page");
const { ApiError } = await import("@/lib/api/client");

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

let root: Root;
let host: HTMLDivElement;

beforeEach(async () => {
  h.pushed.length = 0;
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root.render(createElement(AppShell, null, createElement(NewTaskPage)));
    await wait(20);
  });
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

async function startTask(prompt: string) {
  const el = document.querySelector<HTMLTextAreaElement>("#new-prompt")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(el, prompt);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const start = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Start task")!;
  await act(async () => {
    start.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await wait(10);
  });
  return start;
}

describe("New task", () => {
  it("shows a rate limit readably and lets the user try again", async () => {
    h.createSession = async () => {
      throw new ApiError(429, "rate_limited", "This organization already has 5 active sessions. Wait for one to finish.");
    };
    const start = await startTask("Fix the flaky test");

    const alert = document.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain("Couldn't start the task");
    expect(alert?.textContent).toContain("This organization already has 5 active sessions. Wait for one to finish.");
    expect(start.disabled).toBe(false);
    expect(start.getAttribute("aria-busy")).toBeNull();
    expect(h.pushed).toEqual([]);

    h.createSession = async () => ({ id: "s_new" });
    await act(async () => {
      start.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await wait(10);
    });
    expect(h.pushed).toEqual(["/sessions/s_new"]);
  });
});
