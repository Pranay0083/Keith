// The API surface the UI uses, the error type, and the socket shape the
// event stream needs. http.ts implements it against `keith api`
// (docs/api.md); mock/ implements it in the browser.

import type {
  AcceptedMessage,
  CreateSessionInput,
  CreatedSession,
  InstallUrl,
  Installation,
  Me,
  Providers,
  Repo,
  RepoProfile,
  RepoProfileInput,
  Secret,
  SecretInput,
  Session,
  SessionDiff,
  SessionPage,
} from "./types";

/** A failed API call. `status` is 0 when the API could not be reached. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly fields: Record<string, string>;

  constructor(status: number, code: string, message: string, fields?: Record<string, string>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.fields = fields ?? {};
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

/** The subset of a browser WebSocket that the event stream uses. */
export interface SocketLike {
  onopen: (() => void) | null;
  onmessage: ((data: string) => void) | null;
  onclose: ((code: number, reason: string) => void) | null;
  close(code?: number, reason?: string): void;
}

export interface Api {
  me(): Promise<Me>;
  logout(): Promise<void>;
  providers(): Promise<Providers>;

  installations(orgId: string): Promise<Installation[]>;
  installUrl(orgId: string): Promise<InstallUrl>;

  repos(orgId: string): Promise<Repo[]>;
  getProfile(repoId: string): Promise<RepoProfile>;
  putProfile(repoId: string, input: RepoProfileInput): Promise<RepoProfile>;

  secrets(orgId: string): Promise<Secret[]>;
  putSecret(orgId: string, name: string, input: SecretInput): Promise<Secret>;
  deleteSecret(orgId: string, name: string, repoId: string | null): Promise<void>;

  createSession(orgId: string, input: CreateSessionInput): Promise<CreatedSession>;
  listSessions(orgId: string, cursor?: string | null, limit?: number): Promise<SessionPage>;
  getSession(id: string): Promise<Session>;
  sendMessage(id: string, text: string): Promise<AcceptedMessage>;
  answer(id: string, questionId: string, text: string): Promise<AcceptedMessage>;
  stop(id: string): Promise<void>;
  diff(id: string): Promise<SessionDiff>;

  /** Opens the session's event WebSocket, replaying stored events after afterSeq. */
  openEvents(id: string, afterSeq: number): SocketLike;
}
