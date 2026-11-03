// Pure helpers that turn event data into short labels for the session view.

import type { ToolStep } from "@/lib/session/reducer";

/** The first non-empty line of a text, trimmed. */
export function firstLine(text: string): string {
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t) return t;
  }
  return "";
}

/**
 * A sandbox path relative to its repo: /workspace/web/src/a.ts -> src/a.ts.
 * Repos are cloned to /workspace/<name> (docs/architecture.md).
 */
export function repoRelativePath(path: string, repo?: string | null): string {
  const dir = repo?.split("/")[1];
  if (dir && path.startsWith(`/workspace/${dir}/`)) return path.slice(`/workspace/${dir}/`.length);
  const m = /^\/workspace\/[^/]+\/(.+)$/.exec(path);
  return m?.[1] ?? path;
}

function str(input: unknown, key: string): string | null {
  if (typeof input !== "object" || input === null) return null;
  const v = (input as Record<string, unknown>)[key];
  if (typeof v === "string" && v) return v;
  if (typeof v === "number") return String(v);
  return null;
}

/** A one-line description of what a tool call does, from its input. */
export function describeStep(step: Pick<ToolStep, "tool" | "input">): string {
  const { tool, input } = step;
  const path = str(input, "path");
  switch (tool) {
    case "shell_exec":
      return str(input, "command") ?? "";
    case "clone": {
      const repo = str(input, "repo");
      const branch = str(input, "branch");
      return repo && branch ? `${repo} → ${branch}` : (repo ?? "");
    }
    case "create_shell":
      return `open shell ${str(input, "shell_id") ?? ""}`.trim();
    case "shell_read":
      return `read ${str(input, "shell_id") ?? "shell"}`;
    case "shell_interrupt":
      return `interrupt ${str(input, "shell_id") ?? "shell"}`;
    case "search": {
      const pattern = str(input, "pattern") ?? str(input, "query") ?? "";
      const where = str(input, "path") ?? str(input, "glob");
      return where ? `${pattern} in ${repoRelativePath(where)}` : pattern;
    }
    case "read_issue":
      return str(input, "url") ?? str(input, "issue_url") ?? `#${str(input, "number") ?? ""}`;
    case "create_pull_request":
    case "open_pr":
      return str(input, "title") ?? str(input, "repo") ?? "";
    case "message_user":
      return firstLine(str(input, "text") ?? "");
    case "ask_user":
      return firstLine(str(input, "question") ?? "");
  }
  if (path) return repoRelativePath(path, str(input, "repo"));
  // Fall back to the first string argument.
  if (typeof input === "object" && input !== null) {
    for (const v of Object.values(input as Record<string, unknown>)) {
      if (typeof v === "string" && v) return firstLine(v);
    }
  }
  return "";
}

/** Pretty JSON for a tool's input, for the expanded timeline row. */
export function formatInput(input: unknown): string {
  if (input === null || input === undefined) return "";
  try {
    return JSON.stringify(input, null, 2);
  } catch {
    return String(input);
  }
}

/** A stable, URL-safe id for a file's diff anchor. */
export function fileAnchor(repo: string, path: string): string {
  return `diff-${`${repo}-${path}`.replace(/[^A-Za-z0-9_-]+/g, "-")}`;
}

/** Removes ANSI escape sequences, for showing terminal output as plain text. */
export function stripAnsi(s: string): string {
  // CSI sequences, OSC sequences (terminated by BEL or ST), and lone escapes.
  return s.replace(/\u001b\[[0-?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)|\u001b[@-Z\\-_]/g, "");
}
