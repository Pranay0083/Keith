import { parseDiff } from "react-diff-view";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseEvent, type SessionEvent } from "../events";
import { applyEvent, initialSessionState, openQuestion, type SessionState } from "@/lib/session/reducer";
import { fileDiff } from "./unidiff";

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
  vi.resetModules();
});

async function load() {
  return (await import("./index")).mockApi;
}

/** Opens a mock socket and folds everything it sends into reducer state. */
function watch(api: Awaited<ReturnType<typeof load>>, id: string, afterSeq = 0) {
  let state: SessionState = initialSessionState(id);
  const seen: SessionEvent[] = [];
  const sock = api.openEvents(id, afterSeq);
  sock.onmessage = (data) => {
    const e = parseEvent(data);
    if (!e) throw new Error(`bad frame: ${data}`);
    seen.push(e);
    state = applyEvent(state, e);
  };
  return { sock, seen, get state() { return state; }, set state(s) { state = s; } };
}

describe("mock demo session", () => {
  it("plays clone, install, a question, edits, tests and a PR", async () => {
    const api = await load();
    const w = watch(api, "demo");
    await vi.advanceTimersByTimeAsync(60_000);

    expect(w.state.status).toBe("waiting_for_user");
    const q = openQuestion(w.state);
    expect(q?.choices).toHaveLength(2);
    expect(w.state.shells.map((s) => s.id)).toEqual(["setup", "dev"]);
    expect(w.state.steps.some((s) => s.exitCode === 1)).toBe(true);
    expect(w.seen.some((e) => e.type === "message.delta" && e.seq === 0)).toBe(true);

    await expect(api.sendMessage("demo", "hello")).rejects.toMatchObject({ status: 409, code: "answer_required" });
    await expect(api.answer("demo", "nope", "x")).rejects.toMatchObject({ status: 409, code: "question_mismatch" });

    const answered = api.answer("demo", q!.id, q!.choices[0]!);
    await vi.advanceTimersByTimeAsync(1_000);
    await answered;
    await vi.advanceTimersByTimeAsync(120_000);

    expect(w.state.status).toBe("completed");
    expect(openQuestion(w.state)).toBeNull();
    expect(w.state.prs).toHaveLength(1);
    expect(w.state.prs[0]).toMatchObject({ repo: "acme/web", number: 128 });
    expect(w.state.files.map((f) => [f.path.replace("/workspace/web/", ""), f.action])).toEqual([
      ["src/utils/date.ts", "modified"],
      ["src/utils/date.test.ts", "modified"],
      [".changeset/quiet-dates-sleep.md", "created"],
    ]);
    expect(w.state.steps.every((s) => s.status !== "running")).toBe(true);
    expect(w.state.ahead).toEqual([]);

    // Stored seqs are 1..n with no gaps.
    const seqs = w.seen.filter((e) => e.seq > 0).map((e) => e.seq);
    expect(seqs).toEqual(seqs.map((_, i) => i + 1));

    const diffP = api.diff("demo");
    await vi.advanceTimersByTimeAsync(1_000);
    const diff = await diffP;
    const files = parseDiff(diff.repos[0]!.diff);
    expect(files.map((f) => [f.newPath, f.type])).toEqual([
      [".changeset/quiet-dates-sleep.md", "add"],
      ["src/utils/date.test.ts", "modify"],
      ["src/utils/date.ts", "modify"],
    ]);

    const sessionP = api.getSession("demo");
    await vi.advanceTimersByTimeAsync(1_000);
    const session = await sessionP;
    expect(session.status).toBe("completed");
    expect(session.repos[0]?.pr_number).toBe(128);
    expect(session.finished_at).not.toBeNull();
  });

  it("replays only events after after_seq on reconnect", async () => {
    const api = await load();
    const first = watch(api, "s_eslint");
    await vi.advanceTimersByTimeAsync(500);
    const total = first.seen.length;
    expect(total).toBeGreaterThan(5);
    expect(first.state.status).toBe("completed");

    const second = watch(api, "s_eslint", 5);
    await vi.advanceTimersByTimeAsync(500);
    expect(second.seen[0]?.seq).toBe(6);
    expect(second.seen).toHaveLength(total - 5);
  });

  it("stops a running session", async () => {
    const api = await load();
    const w = watch(api, "demo");
    await vi.advanceTimersByTimeAsync(5_000);
    expect(w.state.status).not.toBeNull();
    const stopped = api.stop("demo");
    await vi.advanceTimersByTimeAsync(2_000);
    await stopped;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(w.state.status).toBe("cancelled");
    expect(w.state.steps.every((s) => s.status !== "running")).toBe(true);
    await expect(api.stop("demo")).rejects.toMatchObject({ status: 409, code: "not_running" });
  });

  it("validates new sessions like the API", async () => {
    const api = await load();
    const p = api.createSession("org_acme", { prompt: "", repo_ids: [] }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(await p).toMatchObject({ status: 422, code: "validation_failed" });

    const q = api
      .createSession("org_acme", { prompt: "x", repo_ids: ["r_api"], issue_url: "https://github.com/acme/web/issues/1" })
      .catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(await q).toMatchObject({ status: 422, fields: { issue_url: "issue_repo_not_tagged" } });
  });
});

describe("fileDiff", () => {
  it("writes diffs react-diff-view can parse", () => {
    const before = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n";
    const after = "a\nB\nc\nd\ne\nf\ng\nh\nI\nj\nk\n";
    const text = fileDiff("x.txt", before, after);
    const [file] = parseDiff(text);
    expect(file?.hunks).toHaveLength(1);
    const changes = file!.hunks[0]!.changes;
    expect(changes.filter((c) => c.type === "insert").map((c) => c.content)).toEqual(["B", "I", "k"]);
    expect(changes.filter((c) => c.type === "delete").map((c) => c.content)).toEqual(["b", "i"]);
    expect(fileDiff("x.txt", before, before)).toBe("");
    expect(parseDiff(fileDiff("new.md", null, "hi\n"))[0]?.type).toBe("add");
    expect(parseDiff(fileDiff("old.md", "bye\n", null))[0]?.type).toBe("delete");
  });
});
