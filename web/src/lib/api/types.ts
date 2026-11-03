// REST shapes from docs/api.md. Keep this file in step with that document:
// it is the contract the Go API implements.

import type { SessionStatusName } from "./events";

/** An RFC 3339 timestamp in UTC. */
export type Timestamp = string;

/** The body of every non-2xx response. */
export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    /** Only on 422, keyed by request field name. */
    fields?: Record<string, string>;
  };
}

// --- Auth ---

export type OrgRole = "owner" | "admin" | "member";

export interface User {
  id: string;
  login: string;
  name: string;
  avatar_url: string;
}

export interface Org {
  id: string;
  name: string;
  role: OrgRole;
}

/** GET /api/me */
export interface Me {
  user: User;
  orgs: Org[];
}

export type LLMProviderId = "anthropic" | "gemini";

/** GET /api/providers */
export interface Providers {
  default_provider: LLMProviderId;
  providers: { id: LLMProviderId; default_model: string }[];
}

// --- GitHub ---

export type InstallationStatus = "active" | "suspended" | "deleted";

/** An item of GET /api/orgs/{org_id}/github/installations */
export interface Installation {
  id: string;
  installation_id: number;
  account_login: string;
  status: InstallationStatus;
  created_at: Timestamp;
}

/** GET /api/orgs/{org_id}/github/install-url */
export interface InstallUrl {
  url: string;
}

// --- Repos ---

/** The editable fields of a repo profile (PUT body). */
export interface RepoProfileInput {
  install_cmd: string;
  build_cmd: string;
  test_cmd: string;
  run_cmd: string;
  knowledge: string;
}

/** A profile as embedded in a repo list item. */
export interface RepoProfileSummary extends RepoProfileInput {
  updated_at: Timestamp;
}

/** GET/PUT /api/repos/{repo_id}/profile */
export interface RepoProfile extends RepoProfileInput {
  repo_id: string;
  /** null until a profile has been saved. */
  updated_at: Timestamp | null;
}

/** An item of GET /api/orgs/{org_id}/repos */
export interface Repo {
  id: string;
  full_name: string;
  default_branch: string;
  private: boolean;
  accessible: boolean;
  profile: RepoProfileSummary | null;
}

// --- Secrets ---

/** An item of GET /api/orgs/{org_id}/secrets. Values are never returned. */
export interface Secret {
  name: string;
  repo_id: string | null;
  repo_full_name: string | null;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/** PUT /api/orgs/{org_id}/secrets/{name} */
export interface SecretInput {
  value: string;
  repo_id?: string | null;
}

// --- Sessions ---

export interface SessionRepo {
  repo_id: string;
  full_name: string;
  branch: string;
  pr_number: number | null;
  pr_url: string | null;
}

export interface SessionAuthor {
  id: string;
  login: string;
  avatar_url: string;
}

/** GET /api/sessions/{id}, and the items of the sessions list. */
export interface Session {
  id: string;
  org_id: string;
  status: SessionStatusName;
  prompt: string;
  issue_url: string | null;
  provider: LLMProviderId;
  model: string;
  created_by: SessionAuthor | null;
  cancel_requested: boolean;
  created_at: Timestamp;
  finished_at: Timestamp | null;
  repos: SessionRepo[];
}

/** POST /api/orgs/{org_id}/sessions */
export interface CreateSessionInput {
  prompt: string;
  repo_ids: string[];
  issue_url?: string;
  provider?: LLMProviderId;
  model?: string;
}

/** 201 from POST /api/orgs/{org_id}/sessions */
export interface CreatedSession {
  id: string;
}

/** GET /api/orgs/{org_id}/sessions */
export interface SessionPage {
  sessions: Session[];
  next_cursor: string | null;
}

/** 202 from POST /messages and POST /answers */
export interface AcceptedMessage {
  message_id: string;
}

/** An item of GET /api/sessions/{id}/diff */
export interface RepoDiff {
  full_name: string;
  /** A unified diff; "" before any change. */
  diff: string;
  truncated: boolean;
  updated_at: Timestamp | null;
}

/** GET /api/sessions/{id}/diff */
export interface SessionDiff {
  repos: RepoDiff[];
}
