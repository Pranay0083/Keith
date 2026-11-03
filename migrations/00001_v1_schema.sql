-- V1 schema: the eleven tables of the cloud coding agent.
-- Chat is not a table: it is the message.* rows in events.
-- Requires Postgres 13+ (gen_random_uuid is built in).

-- +goose Up

-- The workspace that installs the GitHub App and pays.
CREATE TABLE orgs (
    id                   uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    name                 text        NOT NULL,
    -- Per-org default LLM provider for new sessions; NULL means the
    -- service default (DEFAULT_LLM_PROVIDER).
    default_llm_provider text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);

-- People, signed in through GitHub OAuth.
CREATE TABLE users (
    id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    github_user_id bigint      NOT NULL UNIQUE,
    github_login   text        NOT NULL,
    -- GitHub may not expose an email.
    email          text,
    name           text        NOT NULL DEFAULT '',
    avatar_url     text        NOT NULL DEFAULT '',
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- Removing a member stops their access without touching the GitHub
-- connection.
CREATE TABLE org_members (
    org_id     uuid        NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    user_id    uuid        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    role       text        NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, user_id)
);
CREATE INDEX org_members_user_id_idx ON org_members (user_id);

-- One row per App installation, kept in sync by installation webhooks.
CREATE TABLE github_installations (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          uuid        NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    -- GitHub's installation ID, used to mint installation tokens.
    installation_id bigint      NOT NULL UNIQUE,
    account_login   text        NOT NULL,
    status          text        NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'suspended', 'deleted')),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX github_installations_org_id_idx ON github_installations (org_id);

-- One row per repo the App can see.
CREATE TABLE repos (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          uuid        NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    -- GitHub's installation ID (github_installations.installation_id), so
    -- a token can be minted without a join.
    installation_id bigint      NOT NULL
                                REFERENCES github_installations (installation_id) ON DELETE CASCADE,
    github_repo_id  bigint      NOT NULL UNIQUE,
    full_name       text        NOT NULL,
    default_branch  text        NOT NULL DEFAULT 'main',
    private         boolean     NOT NULL DEFAULT true,
    -- False once the repo is removed from the installation or the App is
    -- uninstalled.
    accessible      boolean     NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX repos_org_id_full_name_idx ON repos (org_id, full_name);
CREATE INDEX repos_installation_id_idx ON repos (installation_id);

-- The onboarding answers, plus the ready-made sandbox template.
CREATE TABLE repo_profiles (
    repo_id            uuid        PRIMARY KEY REFERENCES repos (id) ON DELETE CASCADE,
    install_cmd        text        NOT NULL DEFAULT '',
    build_cmd          text        NOT NULL DEFAULT '',
    test_cmd           text        NOT NULL DEFAULT '',
    run_cmd            text        NOT NULL DEFAULT '',
    -- Free-form notes the agent gets in its prompt.
    knowledge          text        NOT NULL DEFAULT '',
    -- Snapshot taken after the first successful setup. Snapshot IDs are
    -- provider-specific; the lockfile hash decides whether a reused
    -- snapshot needs a fresh install.
    snapshot_id        text,
    snapshot_provider  text,
    snapshot_lock_hash text,
    updated_at         timestamptz NOT NULL DEFAULT now()
);

-- One row per task.
CREATE TABLE sessions (
    id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              uuid        NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    -- NULL for sessions no person started (routines, V3).
    created_by          uuid        REFERENCES users (id) ON DELETE SET NULL,
    -- Values match events.SessionStatus.
    status              text        NOT NULL DEFAULT 'queued'
                                    CHECK (status IN ('queued', 'provisioning', 'setting_up', 'running',
                                                      'waiting_for_user', 'completed', 'failed',
                                                      'cancelled', 'no_changes')),
    prompt              text        NOT NULL,
    -- The session is pinned to one LLM provider and model for life.
    provider            text        NOT NULL,
    model               text        NOT NULL,
    sandbox_id          text,
    sandbox_provider    text,
    -- The routine that started this session (V3 adds the routines table
    -- and the foreign key) and what triggered it.
    routine_id          uuid,
    trigger             jsonb,
    -- Set by Stop; the worker checks it between steps.
    cancel_requested_at timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz
);
CREATE INDEX sessions_org_id_created_at_idx ON sessions (org_id, created_at DESC);

-- Lets one session span several repos, each with its own branch and PR.
CREATE TABLE session_repos (
    session_id uuid    NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    repo_id    uuid    NOT NULL REFERENCES repos (id) ON DELETE CASCADE,
    branch     text    NOT NULL,
    pr_number  integer,
    pr_url     text,
    PRIMARY KEY (session_id, repo_id)
);
CREATE INDEX session_repos_repo_id_idx ON session_repos (repo_id);

-- The append-only session log and the UI's source of truth. The primary
-- key is also the (session_id, seq) index that replay reads.
CREATE TABLE events (
    session_id uuid        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    seq        bigint      NOT NULL CHECK (seq > 0),
    type       text        NOT NULL,
    payload    jsonb       NOT NULL DEFAULT '{}',
    ts         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);

-- The model's conversation, saved so a paused session resumes exactly.
CREATE TABLE agent_state (
    session_id   uuid        PRIMARY KEY REFERENCES sessions (id) ON DELETE CASCADE,
    -- Must equal sessions.provider; thinking signatures only replay on the
    -- provider that produced them.
    provider     text        NOT NULL,
    -- []llm.Message
    llm_messages jsonb       NOT NULL DEFAULT '[]',
    -- The providers' raw responses, kept alongside the normalized messages.
    raw_messages jsonb       NOT NULL DEFAULT '[]',
    summary      text        NOT NULL DEFAULT '',
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Org secrets (repo_id NULL) and repo secrets, injected into sandboxes as
-- env vars and never sent in events.
CREATE TABLE secrets (
    id         uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     uuid        NOT NULL REFERENCES orgs (id) ON DELETE CASCADE,
    repo_id    uuid        REFERENCES repos (id) ON DELETE CASCADE,
    -- An environment variable name.
    name       text        NOT NULL CHECK (name ~ '^[A-Za-z_][A-Za-z0-9_]*$'),
    -- Envelope-encrypted value (data key wrapped with SECRETS_KEY); the
    -- format belongs to internal/secrets.
    ciphertext bytea       NOT NULL,
    created_by uuid        REFERENCES users (id) ON DELETE SET NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX secrets_org_name_idx ON secrets (org_id, name) WHERE repo_id IS NULL;
CREATE UNIQUE INDEX secrets_repo_name_idx ON secrets (org_id, repo_id, name) WHERE repo_id IS NOT NULL;

-- +goose Down
DROP TABLE secrets;
DROP TABLE agent_state;
DROP TABLE events;
DROP TABLE session_repos;
DROP TABLE sessions;
DROP TABLE repo_profiles;
DROP TABLE repos;
DROP TABLE github_installations;
DROP TABLE org_members;
DROP TABLE users;
DROP TABLE orgs;
