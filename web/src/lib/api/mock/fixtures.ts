// Seed data for mock mode: one signed-in user in two orgs (an owner in one,
// a plain member in the other, so role-gated UI can be tried).

import type { Installation, Me, Providers, Repo, RepoProfile, Secret } from "../types";

export const USER_ID = "u_octocat";
export const ACME = "org_acme";
export const CONTOSO = "org_contoso";

const ago = (ms: number) => new Date(Date.now() - ms).toISOString();
const H = 3_600_000;
const D = 24 * H;

export const me: Me = {
  user: { id: USER_ID, login: "octocat", name: "Mona Octocat", avatar_url: "" },
  orgs: [
    { id: ACME, name: "acme", role: "owner" },
    { id: CONTOSO, name: "contoso-labs", role: "member" },
  ],
};

export const providers: Providers = {
  default_provider: "anthropic",
  providers: [
    { id: "anthropic", default_model: "claude-opus-5-5" },
    { id: "gemini", default_model: "gemini-3-pro" },
  ],
};

export const installations: Record<string, Installation[]> = {
  [ACME]: [{ id: "inst_acme", installation_id: 51234567, account_login: "acme", status: "active", created_at: ago(40 * D) }],
  [CONTOSO]: [{ id: "inst_contoso", installation_id: 51299881, account_login: "contoso-labs", status: "active", created_at: ago(9 * D) }],
};

function repo(id: string, full_name: string, extra: Partial<Repo> = {}): Repo {
  return { id, full_name, default_branch: "main", private: true, accessible: true, profile: null, ...extra };
}

export const repos: Record<string, Repo[]> = {
  [ACME]: [
    repo("r_web", "acme/web", {
      profile: {
        install_cmd: "npm ci",
        build_cmd: "npm run build",
        test_cmd: "npm test",
        run_cmd: "npm run dev -- --port 3000",
        knowledge: "Vitest for unit tests. Dates go through src/utils/date.ts; never call toLocaleDateString directly.",
        updated_at: ago(12 * D),
      },
    }),
    repo("r_api", "acme/api", {
      profile: {
        install_cmd: "go mod download",
        build_cmd: "go build ./...",
        test_cmd: "go test ./...",
        run_cmd: "go run ./cmd/api",
        knowledge: "Handlers live in internal/http. Integration tests need KEITH_IT=1 and Postgres.",
        updated_at: ago(30 * D),
      },
    }),
    repo("r_ui", "acme/shared-ui"),
    repo("r_docs", "acme/docs", { private: false }),
    repo("r_infra", "acme/infra"),
    repo("r_mobile", "acme/mobile", { default_branch: "develop" }),
    repo("r_pipe", "acme/data-pipeline"),
    repo("r_tokens", "acme/design-tokens", { private: false }),
    repo("r_billing", "acme/legacy-billing", { accessible: false }),
  ],
  [CONTOSO]: [repo("r_lab", "contoso-labs/experiments"), repo("r_site", "contoso-labs/site", { private: false })],
};

export function emptyProfile(repoId: string): RepoProfile {
  return { repo_id: repoId, install_cmd: "", build_cmd: "", test_cmd: "", run_cmd: "", knowledge: "", updated_at: null };
}

export const secrets: Record<string, Secret[]> = {
  [ACME]: [
    { name: "SENTRY_DSN", repo_id: null, repo_full_name: null, created_at: ago(20 * D), updated_at: ago(20 * D) },
    { name: "STRIPE_TEST_KEY", repo_id: null, repo_full_name: null, created_at: ago(18 * D), updated_at: ago(2 * D) },
    { name: "DATABASE_URL", repo_id: "r_api", repo_full_name: "acme/api", created_at: ago(30 * D), updated_at: ago(30 * D) },
  ],
  [CONTOSO]: [{ name: "OPENWEATHER_KEY", repo_id: null, repo_full_name: null, created_at: ago(5 * D), updated_at: ago(5 * D) }],
};

/** Historical sessions: [id, repo ids, prompt, issue url, script name, age]. */
export const pastSessions: Array<{
  id: string;
  repoIds: string[];
  prompt: string;
  issueUrl: string | null;
  script: "eslint" | "fail" | "cancelled" | "nochanges" | "waiting";
  age: number;
}> = [
  { id: "s_ratelimit", repoIds: ["r_api"], prompt: "Add rate limiting to the public API routes", issueUrl: null, script: "waiting", age: 25 * 60_000 },
  { id: "s_moment", repoIds: ["r_web", "r_api"], prompt: "Check whether we still use moment.js anywhere and remove it if so", issueUrl: null, script: "nochanges", age: 5 * H },
  { id: "s_eslint", repoIds: ["r_api"], prompt: "Bump eslint to v9 and move to the flat config", issueUrl: null, script: "eslint", age: 26 * H },
  { id: "s_readme", repoIds: ["r_docs"], prompt: "Rewrite the README intro in a friendlier tone", issueUrl: null, script: "cancelled", age: 2 * D },
  { id: "s_billing", repoIds: ["r_ui"], prompt: "Upgrade the payments SDK to v5 in shared-ui", issueUrl: null, script: "fail", age: 3 * D },
  // On a repo Keith has since lost: a follow-up gets 409 repo_access_lost.
  { id: "s_legacy", repoIds: ["r_billing"], prompt: "Bump eslint to v9 in legacy-billing", issueUrl: null, script: "eslint", age: 6 * D },
];
