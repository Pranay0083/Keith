// Scripted sessions for mock mode. A script is a list of beats; the mock
// server plays them on timers, emitting the same events a real worker would
// (docs/api.md, "What the web app relies on").

import type { EventPayloadMap, EventType, QuestionAskedPayload } from "../events";

export type Beat =
  /** Emit one event after `wait` ms. */
  | { wait: number; emit: Emit }
  /** Stream an assistant message as deltas, then store it. */
  | { wait: number; say: { id: string; text: string } }
  /** Emit tool_call.output chunks for a call, `every` ms apart. */
  | { wait: number; output: { callId: string; shellId?: string; chunks: string[]; every: number } }
  /** Replace a file's content in the session's diff (no event by itself). */
  | { wait: number; file: { repo: string; path: string; content: string | null } }
  /** Ask a question and pause until it is answered. */
  | { wait: number; ask: QuestionAskedPayload };

export type Emit = { [K in EventType]: { type: K; payload: EventPayloadMap[K] } }[EventType];

const e = <K extends EventType>(type: K, payload: EventPayloadMap[K]): Emit => ({ type, payload }) as Emit;

export interface ScriptContext {
  sessionId: string;
  userId: string;
  prompt: string;
  repo: string;
  issueUrl: string | null;
}

export interface Script {
  beats: Beat[];
  /** Beats played after the user answers a question, given the answer. */
  afterAnswer?: (answer: string) => Beat[];
  /** The repo files as they were at clone time, for the diff. */
  baseFiles?: Record<string, Record<string, string>>;
}

let callSeq = 0;
const call = () => `toolu_${(++callSeq).toString(36).padStart(4, "0")}${Math.random().toString(36).slice(2, 8)}`;

/** A tool call with optional output and a result. */
function tool(
  name: string,
  input: Record<string, unknown>,
  result: {
    ms: number;
    summary?: string;
    exitCode?: number;
    isError?: boolean;
    shellId?: string;
    chunks?: string[];
    every?: number;
    between?: (callId: string) => Beat[];
  },
  wait = 350,
): Beat[] {
  const id = call();
  const beats: Beat[] = [{ wait, emit: e("tool_call.started", { call_id: id, tool: name, input }) }];
  if (result.chunks?.length) {
    beats.push({ wait: 150, output: { callId: id, shellId: result.shellId, chunks: result.chunks, every: result.every ?? 90 } });
  }
  if (result.between) beats.push(...result.between(id));
  beats.push({
    wait: Math.min(result.ms, 1600),
    emit: e("tool_call.finished", {
      call_id: id,
      tool: name,
      is_error: result.isError ?? false,
      exit_code: result.exitCode,
      duration_ms: result.ms,
      summary: result.summary,
    }),
  });
  return beats;
}

/** An edit tool call that changes the diff and emits file.changed. */
function edit(name: string, repo: string, dir: string, rel: string, content: string | null, input: Record<string, unknown>, summary: string, action: "created" | "modified" | "deleted"): Beat[] {
  const path = `/workspace/${dir}/${rel}`;
  return tool(name, { path, ...input }, {
    ms: 40 + Math.round(Math.random() * 60),
    summary,
    between: (callId) => [
      { wait: 250, file: { repo, path: rel, content } },
      { wait: 0, emit: e("file.changed", { call_id: callId, repo, path, action }) },
    ],
  });
}

// ANSI helpers for realistic terminal output.
const dim = (s: string) => `\u001b[2m${s}\u001b[22m`;
const bold = (s: string) => `\u001b[1m${s}\u001b[22m`;
const red = (s: string) => `\u001b[31m${s}\u001b[39m`;
const green = (s: string) => `\u001b[32m${s}\u001b[39m`;
const yellow = (s: string) => `\u001b[33m${s}\u001b[39m`;
const cyan = (s: string) => `\u001b[36m${s}\u001b[39m`;
const prompt = (cmd: string) => `${green("agent@sandbox")}:${cyan("/workspace")}$ ${cmd}\r\n`;

const DATE_V0 = `const DAY_MS = 24 * 60 * 60 * 1000;

export function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

/** "today", "yesterday", or "3 days ago" relative to \`now\`. */
export function formatRelative(date: Date, now: Date = new Date()): string {
  const days = Math.round((startOfDay(now).getTime() - startOfDay(date).getTime()) / DAY_MS);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  return \`\${days} days ago\`;
}
`;

const DATE_V1 = `const DAY_MS = 24 * 60 * 60 * 1000;

/** Midnight of \`d\`'s calendar day in \`timeZone\` (default: the runtime's). */
export function startOfDay(d: Date, timeZone?: string): Date {
  if (!timeZone) return new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const [y, m, day] = new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" })
    .format(d)
    .split("-")
    .map(Number);
  return new Date(Date.UTC(y!, m! - 1, day!));
}

/** "today", "yesterday", or "3 days ago" relative to \`now\`. */
export function formatRelative(date: Date, now: Date = new Date()): string {
  const days = Math.round((startOfDay(now).getTime() - startOfDay(date).getTime()) / DAY_MS);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  return \`\${days} days ago\`;
}
`;

const DATE_V2 = DATE_V1.replace(
  `/** "today", "yesterday", or "3 days ago" relative to \`now\`. */
export function formatRelative(date: Date, now: Date = new Date()): string {
  const days = Math.round((startOfDay(now).getTime() - startOfDay(date).getTime()) / DAY_MS);`,
  `/** "today", "yesterday", or "3 days ago" relative to \`now\`, in \`timeZone\`. */
export function formatRelative(date: Date, now: Date = new Date(), timeZone?: string): string {
  const days = Math.round((startOfDay(now, timeZone).getTime() - startOfDay(date, timeZone).getTime()) / DAY_MS);`,
);

const TEST_V0 = `import { describe, expect, it } from "vitest";
import { formatRelative } from "./date";

describe("formatRelative", () => {
  it("says today for the same day", () => {
    expect(formatRelative(new Date())).toBe("today");
  });

  it("says yesterday for the previous day", () => {
    const d = new Date(Date.now() - 24 * 60 * 60 * 1000);
    expect(formatRelative(d)).toBe("yesterday");
  });
});
`;

const TEST_V1 = `import { describe, expect, it } from "vitest";
import { formatRelative } from "./date";

const NOW = new Date("2026-11-02T23:30:00Z");

describe("formatRelative", () => {
  it("says today for the same day", () => {
    expect(formatRelative(new Date("2026-11-02T08:00:00Z"), NOW, "UTC")).toBe("today");
  });

  it("says yesterday for the previous day", () => {
    expect(formatRelative(new Date("2026-11-01T23:59:00Z"), NOW, "UTC")).toBe("yesterday");
  });

  it("uses the given time zone, not the runner's", () => {
    // 23:30 UTC is already Nov 3 in Tokyo.
    expect(formatRelative(new Date("2026-11-02T12:00:00Z"), NOW, "Asia/Tokyo")).toBe("yesterday");
  });
});
`;

const CHANGESET = `---
"@acme/web": patch
---

formatRelative takes an optional time zone, and its tests no longer depend on the runner's clock or zone.
`;

/**
 * The flagship demo: an issue about a flaky date test, fixed end to end
 * (clone, install, investigate, ask, edit, test, open a PR).
 */
export function flakyDateScript(ctx: ScriptContext): Script {
  const { repo, sessionId } = ctx;
  const dir = repo.split("/")[1] ?? "repo";
  const branch = `agent/${sessionId}`;
  const issue = ctx.issueUrl ?? `https://github.com/${repo}/issues/42`;
  const issueNo = /\/issues\/(\d+)/.exec(issue)?.[1] ?? "42";

  const beats: Beat[] = [
    { wait: 300, emit: e("message.user", { message_id: `m_prompt_${sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
    { wait: 700, emit: e("session.status", { status: "provisioning" }) },
    { wait: 1500, emit: e("session.status", { status: "setting_up" }) },
    { wait: 200, emit: e("shell.created", { shell_id: "setup" }) },
    ...tool("clone", { repo, path: `/workspace/${dir}`, branch }, {
      ms: 2140,
      summary: `Cloned ${repo} and checked out ${branch}`,
      shellId: "setup",
      every: 140,
      chunks: [
        prompt(`git clone https://github.com/${repo} /workspace/${dir}`),
        `Cloning into '/workspace/${dir}'...\r\n`,
        "remote: Enumerating objects: 1532, done.\r\n",
        "remote: Counting objects: 100% (1532/1532), done.\r\n",
        "remote: Compressing objects: 100% (611/611), done.\r\n",
        "Receiving objects: 100% (1532/1532), 2.81 MiB | 18.4 MiB/s, done.\r\n",
        "Resolving deltas: 100% (904/904), done.\r\n",
        prompt(`git checkout -b ${branch}`),
        `Switched to a new branch '${branch}'\r\n`,
      ],
    }),
    ...tool("shell_exec", { shell_id: "setup", command: "npm ci" }, {
      ms: 14210,
      summary: "added 812 packages in 14s",
      shellId: "setup",
      every: 160,
      chunks: [
        prompt("npm ci"),
        `${yellow("npm warn")} ${dim("deprecated inflight@1.0.6: This module is not supported, and leaks memory.")}\r\n`,
        `${yellow("npm warn")} ${dim("deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported")}\r\n`,
        "⠙ reify:typescript: timing reifyNode:node_modules/typescript Completed in 2210ms\r",
        "⠹ reify:@esbuild/darwin-arm64: timing build:run:postinstall Completed in 380ms\r",
        "\u001b[2K\r\n",
        "added 812 packages, and audited 813 packages in 14s\r\n",
        "\r\n142 packages are looking for funding\r\n  run `npm fund` for details\r\n",
        `\r\nfound ${green(bold("0"))} vulnerabilities\r\n`,
      ],
    }),
    { wait: 300, emit: e("session.status", { status: "running" }) },
    { wait: 600, say: { id: `a1_${sessionId}`, text: "I'll start by reading the issue, then find the test that flakes and reproduce it." } },
    ...tool("read_issue", { url: issue }, { ms: 410, summary: `#${issueNo} formatRelative test fails between 00:00 and 01:00 UTC` }),
    ...tool(
      "search",
      { pattern: "formatRelative", path: `/workspace/${dir}/src` },
      {
        ms: 95,
        summary: "5 matches in 3 files",
        chunks: ["src/utils/date.ts:8\nsrc/utils/date.test.ts:2,6,11\nsrc/components/Timestamp.tsx:14\n"],
      },
    ),
    ...tool("read_file", { path: `/workspace/${dir}/src/utils/date.ts` }, { ms: 30, summary: "14 lines" }),
    ...tool("read_file", { path: `/workspace/${dir}/src/utils/date.test.ts` }, { ms: 28, summary: "14 lines" }),
    { wait: 400, emit: e("shell.created", { shell_id: "dev" }) },
    ...tool("create_shell", { shell_id: "dev" }, { ms: 120, summary: "Opened shell dev" }, 0),
    ...tool("shell_exec", { shell_id: "dev", command: "TZ=Asia/Tokyo npx vitest run src/utils/date.test.ts" }, {
      ms: 3870,
      exitCode: 1,
      summary: "1 failed, 1 passed",
      shellId: "dev",
      every: 120,
      chunks: [
        prompt("TZ=Asia/Tokyo npx vitest run src/utils/date.test.ts"),
        `\r\n ${bold(cyan("RUN"))} ${cyan("v3.2.4")} ${dim(`/workspace/${dir}`)}\r\n\r\n`,
        ` ${red("❯")} src/utils/date.test.ts ${dim("(2 tests | 1 failed)")} 14${dim("ms")}\r\n`,
        `   ${green("✓")} formatRelative > says today for the same day\r\n`,
        `   ${red("×")} formatRelative > says yesterday for the previous day 4${dim("ms")}\r\n`,
        `\r\n${red(bold(" FAIL "))} src/utils/date.test.ts > formatRelative > says yesterday for the previous day\r\n`,
        `${red("AssertionError")}: expected ${red("'today'")} to be ${green("'yesterday'")} ${dim("// Object.is equality")}\r\n\r\n`,
        `${dim(" ❯ src/utils/date.test.ts:11:27")}\r\n\r\n`,
        ` ${bold("Test Files")}  ${red(bold("1 failed"))} ${dim("(1)")}\r\n`,
        `      ${bold("Tests")}  ${red(bold("1 failed"))} ${dim("|")} ${green(bold("1 passed"))} ${dim("(2)")}\r\n`,
      ],
    }),
    {
      wait: 700,
      say: {
        id: `a2_${sessionId}`,
        text:
          "Reproduced it. `formatRelative` compares calendar days in the runner's local time zone, and the test builds its dates from `Date.now()`, so it fails whenever the runner's zone has already crossed midnight.\n\nThere are two reasonable fixes, and they change the public API differently.",
      },
    },
    {
      wait: 500,
      ask: {
        question_id: `q_tz_${sessionId}`,
        question:
          "Should I make `formatRelative` take an optional time zone (callers and tests pass one explicitly), or just pin the tests to UTC and leave the function alone?",
        choices: ["Make formatRelative take a time zone", "Just pin the tests to UTC"],
      },
    },
  ];

  const afterAnswer = (answer: string): Beat[] => {
    const pinOnly = /pin/i.test(answer) && !/time zone/i.test(answer);
    return [
      {
        wait: 600,
        say: {
          id: `a3_${sessionId}`,
          text: pinOnly
            ? "Pinning the tests: they'll pass a fixed `now`, and I'll add a time zone parameter only where the test needs it."
            : "Adding an optional `timeZone` parameter to `startOfDay` and `formatRelative`. Leaving it out keeps today's behaviour, so no caller changes.",
        },
      },
      ...edit("str_replace", repo, dir, "src/utils/date.ts", DATE_V1, { old_str: "export function startOfDay(d: Date): Date {", new_str: "export function startOfDay(d: Date, timeZone?: string): Date {" }, "Replaced 1 occurrence", "modified"),
      ...edit("str_replace", repo, dir, "src/utils/date.ts", DATE_V2, { old_str: "export function formatRelative(date: Date, now: Date = new Date()): string {" }, "Replaced 1 occurrence", "modified"),
      ...edit("create_file", repo, dir, "src/utils/date.test.ts", TEST_V1, { overwrite: true }, "Wrote 19 lines", "modified"),
      ...edit("create_file", repo, dir, ".changeset/quiet-dates-sleep.md", CHANGESET, {}, "Wrote 5 lines", "created"),
      ...tool("shell_exec", { shell_id: "dev", command: "TZ=Asia/Tokyo npx vitest run src/utils/date.test.ts" }, {
        ms: 2410,
        summary: "3 passed",
        shellId: "dev",
        every: 110,
        chunks: [
          prompt("TZ=Asia/Tokyo npx vitest run src/utils/date.test.ts"),
          `\r\n ${bold(cyan("RUN"))} ${cyan("v3.2.4")} ${dim(`/workspace/${dir}`)}\r\n\r\n`,
          ` ${green("✓")} src/utils/date.test.ts ${dim("(3 tests)")} 9${dim("ms")}\r\n\r\n`,
          ` ${bold("Test Files")}  ${green(bold("1 passed"))} ${dim("(1)")}\r\n`,
          `      ${bold("Tests")}  ${green(bold("3 passed"))} ${dim("(3)")}\r\n`,
        ],
      }),
      { wait: 500, say: { id: `a4_${sessionId}`, text: "Passing in Tokyo time too. Running the full suite before I open the PR." } },
      ...tool("shell_exec", { shell_id: "dev", command: "npm test" }, {
        ms: 21840,
        summary: "186 passed",
        shellId: "dev",
        every: 70,
        chunks: [
          prompt("npm test"),
          `\r\n> @acme/web@4.12.0 test\r\n> vitest run\r\n\r\n ${bold(cyan("RUN"))} ${cyan("v3.2.4")} ${dim(`/workspace/${dir}`)}\r\n\r\n`,
          ...[
            ["src/utils/date.test.ts", 3],
            ["src/utils/money.test.ts", 14],
            ["src/utils/strings.test.ts", 22],
            ["src/hooks/useDebounce.test.ts", 6],
            ["src/components/Timestamp.test.tsx", 5],
            ["src/components/Button.test.tsx", 11],
            ["src/components/Table.test.tsx", 19],
            ["src/pages/settings.test.tsx", 17],
            ["src/pages/billing.test.tsx", 24],
            ["src/api/client.test.ts", 31],
            ["src/api/retry.test.ts", 9],
            ["src/store/session.test.ts", 25],
          ].map(([f, n]) => ` ${green("✓")} ${f} ${dim(`(${n} tests)`)} ${Math.round(Math.random() * 80 + 4)}${dim("ms")}\r\n`),
          `\r\n ${bold("Test Files")}  ${green(bold("24 passed"))} ${dim("(24)")}\r\n`,
          `      ${bold("Tests")}  ${green(bold("186 passed"))} ${dim("(186)")}\r\n`,
          `   ${dim("Duration")}  21.84s\r\n`,
        ],
      }),
      ...tool(
        "create_pull_request",
        { repo, branch, title: "Make formatRelative time-zone aware and fix the flaky date test" },
        {
          ms: 1830,
          summary: `Opened ${repo}#128`,
          between: () => [
            {
              wait: 900,
              emit: e("pr.opened", {
                repo,
                number: 128,
                url: `https://github.com/${repo}/pull/128`,
                title: "Make formatRelative time-zone aware and fix the flaky date test",
                branch,
              }),
            },
          ],
        },
      ),
      {
        wait: 500,
        say: {
          id: `a5_${sessionId}`,
          text: `Opened ${repo}#128.\n\n- \`formatRelative\` and \`startOfDay\` take an optional \`timeZone\`; leaving it out keeps today's behaviour.\n- The tests pin \`now\` and the zone, and a new case covers a runner that is ahead of UTC.\n- The full suite passes (186 tests).\n\nCloses #${issueNo}.`,
        },
      },
      { wait: 400, emit: e("session.status", { status: "completed" }) },
    ];
  };

  return {
    beats,
    afterAnswer,
    baseFiles: { [repo]: { "src/utils/date.ts": DATE_V0, "src/utils/date.test.ts": TEST_V0 } },
  };
}

/** A short reply to a follow-up on a finished session. */
export function followUpBeats(sessionId: string, text: string, n: number): Beat[] {
  return [
    { wait: 300, emit: e("session.status", { status: "running" }) },
    {
      wait: 700,
      say: {
        id: `af${n}_${sessionId}`,
        text: `On it: “${text.length > 80 ? `${text.slice(0, 80)}…` : text}”. In mock mode I only pretend to work, so there is nothing new to push.`,
      },
    },
    { wait: 500, emit: e("session.status", { status: "completed" }) },
  ];
}

/** The agent picking up a steering message mid-run. */
export function steerBeats(sessionId: string, messageId: string, userId: string, text: string, n: number): Beat[] {
  return [
    { wait: 200, emit: e("message.user", { message_id: messageId, user_id: userId, text }) },
    { wait: 500, say: { id: `as${n}_${sessionId}`, text: "Got it, I'll take that into account for the rest of the task." } },
  ];
}

// --- Historical sessions, replayed instantly into finished logs ---

export function eslintBumpScript(ctx: ScriptContext): Script {
  const dir = ctx.repo.split("/")[1] ?? "repo";
  const before = `module.exports = {\n  root: true,\n  extends: ["eslint:recommended"],\n  rules: {\n    "no-unused-vars": "warn",\n  },\n};\n`;
  const after = `import js from "@eslint/js";\n\nexport default [\n  js.configs.recommended,\n  {\n    rules: {\n      "no-unused-vars": "warn",\n    },\n  },\n];\n`;
  return {
    baseFiles: { [ctx.repo]: { ".eslintrc.cjs": before } },
    beats: [
      { wait: 0, emit: e("message.user", { message_id: `m_prompt_${ctx.sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
      { wait: 0, emit: e("session.status", { status: "provisioning" }) },
      { wait: 0, emit: e("session.status", { status: "setting_up" }) },
      { wait: 0, emit: e("shell.created", { shell_id: "setup" }) },
      ...tool("shell_exec", { shell_id: "setup", command: "npm ci" }, { ms: 9800, summary: "added 604 packages", shellId: "setup", chunks: [prompt("npm ci"), "added 604 packages in 10s\r\n"] }),
      { wait: 0, emit: e("session.status", { status: "running" }) },
      { wait: 0, say: { id: `a1_${ctx.sessionId}`, text: "Moving to the flat config and bumping eslint to v9." } },
      ...edit("delete_file", ctx.repo, dir, ".eslintrc.cjs", null, {}, "Deleted", "deleted"),
      ...edit("create_file", ctx.repo, dir, "eslint.config.js", after, {}, "Wrote 10 lines", "created"),
      ...tool("shell_exec", { shell_id: "setup", command: "npx eslint ." }, { ms: 4100, summary: "0 problems", shellId: "setup", chunks: [prompt("npx eslint ."), "\r\n"] }),
      ...tool("create_pull_request", { repo: ctx.repo, title: "Upgrade to eslint 9 flat config" }, {
        ms: 1500,
        summary: `Opened ${ctx.repo}#57`,
        between: () => [{ wait: 0, emit: e("pr.opened", { repo: ctx.repo, number: 57, url: `https://github.com/${ctx.repo}/pull/57`, title: "Upgrade to eslint 9 flat config", branch: `agent/${ctx.sessionId}` }) }],
      }),
      { wait: 0, say: { id: `a2_${ctx.sessionId}`, text: `Opened ${ctx.repo}#57 with the flat config. Lint is clean.` } },
      { wait: 0, emit: e("session.status", { status: "completed" }) },
    ],
  };
}

export function provisionFailScript(ctx: ScriptContext): Script {
  return {
    beats: [
      { wait: 0, emit: e("message.user", { message_id: `m_prompt_${ctx.sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
      { wait: 0, emit: e("session.status", { status: "provisioning" }) },
      {
        wait: 0,
        emit: e("error", {
          code: "sandbox_provision_failed",
          message: "The sandbox didn't start within 120 s (provider timeout). Nothing was changed.",
          retryable: true,
        }),
      },
      { wait: 0, emit: e("session.status", { status: "failed", reason: "Sandbox provisioning failed." }) },
    ],
  };
}

export function cancelledScript(ctx: ScriptContext): Script {
  return {
    beats: [
      { wait: 0, emit: e("message.user", { message_id: `m_prompt_${ctx.sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
      { wait: 0, emit: e("session.status", { status: "provisioning" }) },
      { wait: 0, emit: e("session.status", { status: "running" }) },
      { wait: 0, say: { id: `a1_${ctx.sessionId}`, text: "Reading the current README first." } },
      ...tool("read_file", { path: "/workspace/docs/README.md" }, { ms: 30, summary: "212 lines" }),
      { wait: 0, emit: e("tool_call.started", { call_id: "toolu_cancel1", tool: "search", input: { pattern: "TODO", path: "/workspace/docs" } }) },
      { wait: 0, emit: e("session.status", { status: "cancelled", reason: "Stopped by octocat." }) },
    ],
  };
}

export function noChangesScript(ctx: ScriptContext): Script {
  return {
    beats: [
      { wait: 0, emit: e("message.user", { message_id: `m_prompt_${ctx.sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
      { wait: 0, emit: e("session.status", { status: "provisioning" }) },
      { wait: 0, emit: e("session.status", { status: "running" }) },
      ...tool("search", { pattern: "from [\"']moment[\"']", path: "/workspace" }, { ms: 140, summary: "0 matches", chunks: ["(no matches)\n"] }),
      ...tool("search", { pattern: "require\\([\"']moment[\"']\\)", path: "/workspace" }, { ms: 120, summary: "0 matches" }),
      { wait: 0, say: { id: `a1_${ctx.sessionId}`, text: "Nothing imports `moment` any more in either repo, and it isn't in any `package.json`. No changes needed." } },
      { wait: 0, emit: e("session.status", { status: "no_changes", reason: "Nothing to change: moment.js is no longer used." }) },
    ],
  };
}

export function waitingScript(ctx: ScriptContext): Script {
  return {
    beats: [
      { wait: 0, emit: e("message.user", { message_id: `m_prompt_${ctx.sessionId}`, user_id: ctx.userId, text: ctx.prompt }) },
      { wait: 0, emit: e("session.status", { status: "provisioning" }) },
      { wait: 0, emit: e("session.status", { status: "running" }) },
      ...tool("search", { pattern: "app.use\\(", path: "/workspace/api/src" }, { ms: 80, summary: "9 matches in 2 files" }),
      ...tool("read_file", { path: "/workspace/api/src/server.ts" }, { ms: 25, summary: "88 lines" }),
      { wait: 0, say: { id: `a1_${ctx.sessionId}`, text: "The public routes are all mounted in `src/server.ts`. Before I add a limiter I need one decision from you." } },
      {
        wait: 0,
        ask: {
          question_id: `q_rl_${ctx.sessionId}`,
          question: "What limit should anonymous clients get? Authenticated clients will get 10× that.",
          choices: ["60 requests per minute", "300 requests per minute", "1000 requests per hour"],
        },
      },
    ],
    afterAnswer: (answer) => [
      { wait: 600, say: { id: `a2_${ctx.sessionId}`, text: `Going with ${answer}. In mock mode this session stops here.` } },
      { wait: 500, emit: e("session.status", { status: "completed" }) },
    ],
  };
}
