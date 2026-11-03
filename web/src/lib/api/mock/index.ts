// Mock mode (NEXT_PUBLIC_API_MOCK=1): the whole Api in the browser, with
// the validation and status codes of docs/api.md, and a scripted worker
// behind the event stream. State lives in memory and resets on reload.
//
// Try /sessions/demo. In the console, window.__keithMock.drop() cuts every
// event socket (to watch the reconnect), and ?mockSpeed=4 plays faster.

import { ApiError, type Api } from "../api";
import { isTerminal } from "../events";
import type { Repo, RepoProfile, Secret, Session } from "../types";
import * as fx from "./fixtures";
import {
  cancelledScript,
  eslintBumpScript,
  flakyDateScript,
  followUpBeats,
  noChangesScript,
  provisionFailScript,
  steerBeats,
  waitingScript,
  type Script,
  type ScriptContext,
} from "./scripts";
import { MockServer, MockSocket, type MockSessionRecord } from "./server";

const server = new MockServer();
const repos = structuredClone(fx.repos);
const secrets = structuredClone(fx.secrets);
const profiles = new Map<string, RepoProfile>();
const orgOf = new Map<string, string>(); // session id -> org id
let idSeq = 0;

for (const list of Object.values(repos)) {
  for (const r of list) {
    if (r.profile) profiles.set(r.id, { repo_id: r.id, ...r.profile });
  }
}

function findRepo(id: string): { orgId: string; repo: Repo } | null {
  for (const [orgId, list] of Object.entries(repos)) {
    const repo = list.find((r) => r.id === id);
    if (repo) return { orgId, repo };
  }
  return null;
}

function newSession(id: string, orgId: string, repoIds: string[], prompt: string, issueUrl: string | null, createdAt: string): Session {
  return {
    id,
    org_id: orgId,
    status: "queued",
    prompt,
    issue_url: issueUrl,
    provider: "anthropic",
    model: "claude-opus-5-5",
    created_by: { id: fx.USER_ID, login: fx.me.user.login, avatar_url: fx.me.user.avatar_url },
    cancel_requested: false,
    created_at: createdAt,
    finished_at: null,
    repos: repoIds.map((rid) => ({
      repo_id: rid,
      full_name: findRepo(rid)?.repo.full_name ?? rid,
      branch: `agent/${id}`,
      pr_number: null,
      pr_url: null,
    })),
  };
}

function ctxFor(s: Session): ScriptContext {
  return { sessionId: s.id, userId: fx.USER_ID, prompt: s.prompt || s.issue_url || "", repo: s.repos[0]?.full_name ?? "acme/web", issueUrl: s.issue_url };
}

// Seed the history, then the live demo (started when first opened).
for (const p of fx.pastSessions) {
  const created = Date.now() - p.age;
  const s = newSession(p.id, fx.ACME, p.repoIds, p.prompt, p.issueUrl, new Date(created).toISOString());
  const ctx = ctxFor(s);
  const script: Script = {
    eslint: eslintBumpScript,
    fail: provisionFailScript,
    cancelled: cancelledScript,
    nochanges: noChangesScript,
    waiting: waitingScript,
  }[p.script](ctx);
  const rec = server.create(s, script);
  orgOf.set(s.id, fx.ACME);
  server.playInstant(rec, created);
}
{
  const issue = "https://github.com/acme/web/issues/42";
  const s = newSession("demo", fx.ACME, ["r_web"], "Fix the flaky date test from issue #42", issue, new Date().toISOString());
  server.create(s, flakyDateScript(ctxFor(s)));
  orgOf.set("demo", fx.ACME);
}

if (typeof window !== "undefined") {
  (window as unknown as { __keithMock: unknown }).__keithMock = {
    server,
    drop: () => server.dropAll(),
    speed: (n: number) => {
      server.speed = n > 0 ? n : 1;
    },
  };
}

// --- helpers ---

const latency = () => new Promise<void>((r) => setTimeout(r, 120 + Math.random() * 180));

async function ok<T>(value: T): Promise<T> {
  await latency();
  return structuredClone(value);
}

function member(orgId: string) {
  const org = fx.me.orgs.find((o) => o.id === orgId);
  if (!org) throw new ApiError(403, "not_a_member", "You're not a member of this organization.");
  return org;
}

function manager(orgId: string) {
  const org = member(orgId);
  if (org.role === "member") throw new ApiError(403, "forbidden_role", "Only owners and admins can do this.");
  return org;
}

function invalid(fields: Record<string, string>): ApiError {
  const [field, msg] = Object.entries(fields)[0] ?? ["", "invalid"];
  return new ApiError(422, "validation_failed", `${field}: ${msg}`, fields);
}

function session(id: string): MockSessionRecord {
  const rec = server.sessions.get(id);
  if (!rec) throw new ApiError(404, "not_found", "No such session.");
  return rec;
}

const SECRET_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const ISSUE = /^https:\/\/github\.com\/([^/]+\/[^/]+)\/issues\/\d+$/;
const messageId = () => `m_${Date.now().toString(36)}${(++idSeq).toString(36)}`;

// --- the Api ---

export const mockApi: Api = {
  me: () => ok(fx.me),
  logout: () => ok(undefined),
  providers: () => ok(fx.providers),

  installations: async (orgId) => {
    member(orgId);
    return ok(fx.installations[orgId] ?? []);
  },
  installUrl: async (orgId) => {
    manager(orgId);
    // Pretend GitHub sent us straight back from a successful install.
    return ok({ url: "/repos?installed=1" });
  },

  repos: async (orgId) => {
    member(orgId);
    return ok(repos[orgId] ?? []);
  },
  getProfile: async (repoId) => {
    if (!findRepo(repoId)) throw new ApiError(404, "not_found", "No such repo.");
    return ok(profiles.get(repoId) ?? fx.emptyProfile(repoId));
  },
  putProfile: async (repoId, input) => {
    const found = findRepo(repoId);
    if (!found) throw new ApiError(404, "not_found", "No such repo.");
    const fields: Record<string, string> = {};
    for (const k of ["install_cmd", "build_cmd", "test_cmd", "run_cmd"] as const) {
      if (input[k].length > 2000) fields[k] = "at most 2,000 characters";
    }
    if (input.knowledge.length > 20000) fields.knowledge = "at most 20,000 characters";
    if (Object.keys(fields).length) throw invalid(fields);
    const saved: RepoProfile = { repo_id: repoId, ...input, updated_at: new Date().toISOString() };
    profiles.set(repoId, saved);
    found.repo.profile = { ...input, updated_at: saved.updated_at! };
    return ok(saved);
  },

  secrets: async (orgId) => {
    member(orgId);
    const list = [...(secrets[orgId] ?? [])].sort(
      (a, b) => a.name.localeCompare(b.name) || Number(a.repo_id !== null) - Number(b.repo_id !== null),
    );
    return ok(list);
  },
  putSecret: async (orgId, name, input) => {
    manager(orgId);
    const fields: Record<string, string> = {};
    if (!SECRET_NAME.test(name) || name.length > 128) fields.name = "letters, digits and _, not starting with a digit";
    if (!input.value) fields.value = "must not be empty";
    const repoId = input.repo_id ?? null;
    const repo = repoId ? (repos[orgId] ?? []).find((r) => r.id === repoId) : null;
    if (repoId && !repo) fields.repo_id = "not a repo of this organization";
    if (Object.keys(fields).length) throw invalid(fields);
    const list = (secrets[orgId] ??= []);
    const now = new Date().toISOString();
    const existing = list.find((s) => s.name === name && s.repo_id === repoId);
    let out: Secret;
    if (existing) {
      existing.updated_at = now;
      out = existing;
    } else {
      out = { name, repo_id: repoId, repo_full_name: repo?.full_name ?? null, created_at: now, updated_at: now };
      list.push(out);
    }
    return ok(out);
  },
  deleteSecret: async (orgId, name, repoId) => {
    manager(orgId);
    const list = secrets[orgId] ?? [];
    const i = list.findIndex((s) => s.name === name && s.repo_id === (repoId ?? null));
    if (i === -1) throw new ApiError(404, "not_found", "No such secret.");
    list.splice(i, 1);
    await latency();
  },

  createSession: async (orgId, input) => {
    member(orgId);
    const fields: Record<string, string> = {};
    const prompt = input.prompt.trim();
    const issueUrl = input.issue_url?.trim() || null;
    const ids = [...new Set(input.repo_ids)];
    const orgRepos = repos[orgId] ?? [];
    if (!prompt && !issueUrl) fields.prompt = "describe the task or give an issue URL";
    if (prompt.length > 32000) fields.prompt = "at most 32,000 characters";
    if (ids.length < 1 || ids.length > 10) fields.repo_ids = "tag between 1 and 10 repos";
    else if (ids.some((id) => !orgRepos.find((r) => r.id === id && r.accessible))) fields.repo_ids = "every repo must be accessible";
    if (issueUrl) {
      const m = ISSUE.exec(issueUrl);
      if (!m) fields.issue_url = "must look like https://github.com/owner/repo/issues/42";
      else if (!ids.some((id) => orgRepos.find((r) => r.id === id)?.full_name.toLowerCase() === m[1]!.toLowerCase())) {
        fields.issue_url = "issue_repo_not_tagged";
      }
    }
    if (input.provider && !fx.providers.providers.some((p) => p.id === input.provider)) {
      throw new ApiError(422, "provider_unavailable", `The ${input.provider} provider isn't configured on this server.`);
    }
    if (Object.keys(fields).length) throw invalid(fields);

    const id = `s_${Date.now().toString(36)}${(++idSeq).toString(36)}`;
    const s = newSession(id, orgId, ids, prompt, issueUrl, new Date().toISOString());
    const provider = input.provider ?? fx.providers.default_provider;
    s.provider = provider;
    s.model = input.model || (fx.providers.providers.find((p) => p.id === provider)?.default_model ?? "");
    const rec = server.create(s, flakyDateScript(ctxFor(s)));
    orgOf.set(id, orgId);
    server.ensureStarted(rec);
    await latency();
    return { id };
  },
  listSessions: async (orgId, cursor, limit = 25) => {
    member(orgId);
    const all = [...server.sessions.values()]
      .filter((r) => orgOf.get(r.session.id) === orgId)
      .map((r) => r.session)
      .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id));
    const start = cursor ? Number(cursor) || 0 : 0;
    const n = Math.min(Math.max(limit, 1), 100);
    const page = all.slice(start, start + n);
    return ok({ sessions: page, next_cursor: start + n < all.length ? String(start + n) : null });
  },
  getSession: async (id) => {
    const rec = session(id);
    return ok(rec.session);
  },
  sendMessage: async (id, text) => {
    const rec = session(id);
    const t = text.trim();
    if (!t || t.length > 16000) throw invalid({ text: "between 1 and 16,000 characters" });
    if (rec.session.status === "waiting_for_user") {
      throw new ApiError(409, "answer_required", "The agent is waiting for an answer to its question.");
    }
    const mid = messageId();
    rec.turns += 1;
    if (isTerminal(rec.session.status)) {
      server.inject(rec, [
        { wait: 600, emit: { type: "message.user", payload: { message_id: mid, user_id: fx.USER_ID, text: t } } },
        ...followUpBeats(id, t, rec.turns),
      ]);
    } else {
      // Picked up at the next beat boundary, like the next model turn.
      server.inject(rec, steerBeats(id, mid, fx.USER_ID, t, rec.turns));
    }
    await latency();
    return { message_id: mid };
  },
  answer: async (id, questionId, text) => {
    const rec = session(id);
    const t = text.trim();
    if (!t || t.length > 16000) throw invalid({ text: "between 1 and 16,000 characters" });
    if (rec.session.status !== "waiting_for_user" || !rec.waitingFor) {
      throw new ApiError(409, "not_waiting", "The session isn't waiting for an answer.");
    }
    if (rec.waitingFor !== questionId) {
      throw new ApiError(409, "question_mismatch", "That question is no longer open.");
    }
    const mid = messageId();
    server.answer(rec, mid, fx.USER_ID, questionId, t);
    await latency();
    return { message_id: mid };
  },
  stop: async (id) => {
    const rec = session(id);
    if (isTerminal(rec.session.status)) throw new ApiError(409, "not_running", "The session has already finished.");
    server.stop(rec);
    await latency();
  },
  diff: async (id) => {
    const rec = session(id);
    return ok({ repos: server.diff(rec) });
  },

  openEvents: (id, afterSeq) => new MockSocket(server, id, afterSeq),
};
