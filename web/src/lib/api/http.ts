// The real API client: fetch with the session cookie, and a WebSocket for
// events. Shapes and statuses follow docs/api.md.

import { ApiError, type Api, type SocketLike } from "./api";
import type { ApiErrorBody } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080").replace(/\/+$/, "");

/** The ws:// or wss:// form of API_URL. */
export function wsBaseUrl(apiUrl: string = API_URL): string {
  return apiUrl.replace(/^http(s?):\/\//, (_m, s: string) => `ws${s}://`);
}

const enc = encodeURIComponent;

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_URL + path, {
      method,
      credentials: "include",
      cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network", "Can't reach the Keith API. Check your connection and try again.");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!res.ok) {
    const err = (data as Partial<ApiErrorBody> | null)?.error;
    throw new ApiError(
      res.status,
      err?.code ?? `http_${res.status}`,
      err?.message ?? (res.statusText || `Request failed with status ${res.status}`),
      err?.fields,
    );
  }
  return data as T;
}

/** Adapts a browser WebSocket to SocketLike. */
export function wrapWebSocket(url: string): SocketLike {
  const ws = new WebSocket(url);
  const sock: SocketLike = {
    onopen: null,
    onmessage: null,
    onclose: null,
    close: (code, reason) => {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
      if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
        ws.close(code ?? 1000, reason);
      }
    },
  };
  ws.onopen = () => sock.onopen?.();
  ws.onmessage = (ev: MessageEvent) => {
    if (typeof ev.data === "string") sock.onmessage?.(ev.data);
  };
  // A failed handshake fires error and then close (1006); close carries it.
  ws.onclose = (ev: CloseEvent) => sock.onclose?.(ev.code, ev.reason);
  return sock;
}

export const httpApi: Api = {
  me: () => request("GET", "/api/me"),
  logout: () => request("POST", "/auth/logout"),
  providers: () => request("GET", "/api/providers"),

  installations: (orgId) => request("GET", `/api/orgs/${enc(orgId)}/github/installations`),
  installUrl: (orgId) => request("GET", `/api/orgs/${enc(orgId)}/github/install-url`),

  repos: (orgId) => request("GET", `/api/orgs/${enc(orgId)}/repos`),
  getProfile: (repoId) => request("GET", `/api/repos/${enc(repoId)}/profile`),
  putProfile: (repoId, input) => request("PUT", `/api/repos/${enc(repoId)}/profile`, input),

  secrets: (orgId) => request("GET", `/api/orgs/${enc(orgId)}/secrets`),
  putSecret: (orgId, name, input) => request("PUT", `/api/orgs/${enc(orgId)}/secrets/${enc(name)}`, input),
  deleteSecret: (orgId, name, repoId) =>
    request(
      "DELETE",
      `/api/orgs/${enc(orgId)}/secrets/${enc(name)}${repoId ? `?repo_id=${enc(repoId)}` : ""}`,
    ),

  createSession: (orgId, input) => request("POST", `/api/orgs/${enc(orgId)}/sessions`, input),
  listSessions: (orgId, cursor, limit) => {
    const q = new URLSearchParams();
    if (cursor) q.set("cursor", cursor);
    if (limit) q.set("limit", String(limit));
    const qs = q.toString();
    return request("GET", `/api/orgs/${enc(orgId)}/sessions${qs ? `?${qs}` : ""}`);
  },
  getSession: (id) => request("GET", `/api/sessions/${enc(id)}`),
  sendMessage: (id, text) => request("POST", `/api/sessions/${enc(id)}/messages`, { text }),
  answer: (id, questionId, text) =>
    request("POST", `/api/sessions/${enc(id)}/answers`, { question_id: questionId, text }),
  stop: async (id) => {
    await request<unknown>("POST", `/api/sessions/${enc(id)}/stop`);
  },
  diff: (id) => request("GET", `/api/sessions/${enc(id)}/diff`),

  openEvents: (id, afterSeq) =>
    wrapWebSocket(`${wsBaseUrl()}/api/sessions/${enc(id)}/events?after_seq=${Math.max(0, Math.floor(afterSeq))}`),
};
