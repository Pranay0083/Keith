-- User input waiting for the worker, the latest diff per session repo,
-- the session's open question, and the order of a session's repos.
--
-- The API never writes events: the worker holding the session lease is
-- the only writer. Messages and answers therefore go into session_inputs,
-- and the worker emits message.user when it hands one to the model.

-- +goose Up

-- Messages (steering or follow-ups) and answers to question.asked, in the
-- order they arrived. consumed_at is set once the worker has given the
-- input to the model; until then it is pending and never dropped.
CREATE TABLE session_inputs (
    -- "m_" + a time-ordered ID; the message_id the API returns and the
    -- worker puts on message.user.
    id          text        PRIMARY KEY CHECK (id ~ '^m_[0-9A-Za-z]+$'),
    session_id  uuid        NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    kind        text        NOT NULL CHECK (kind IN ('message', 'answer')),
    -- The question.asked this answers; set exactly for answers.
    question_id text,
    -- NULL for input no person sent, or whose author was deleted.
    user_id     uuid        REFERENCES users (id) ON DELETE SET NULL,
    text        text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    consumed_at timestamptz,
    CHECK ((kind = 'answer') = (question_id IS NOT NULL))
);
-- Pending input of one session, oldest first.
CREATE INDEX session_inputs_pending_idx ON session_inputs (session_id, created_at, id)
    WHERE consumed_at IS NULL;

ALTER TABLE session_repos
    -- The latest unified diff the worker saved for the repo (at most
    -- 1 MiB, secrets redacted), against its default branch at clone time.
    ADD COLUMN last_diff           text        NOT NULL DEFAULT '',
    ADD COLUMN last_diff_truncated boolean     NOT NULL DEFAULT false,
    ADD COLUMN last_diff_at        timestamptz,
    -- The repo's index in the session's repo_ids at creation, so the
    -- session lists its repos in the order they were tagged.
    ADD COLUMN position            integer     NOT NULL DEFAULT 0;

-- The question.asked waiting for an answer (its question_id), or NULL.
ALTER TABLE sessions ADD COLUMN open_question_id text;

-- Sessions that hold or are about to hold a sandbox, for the per-org cap.
CREATE INDEX sessions_org_active_idx ON sessions (org_id)
    WHERE status IN ('queued', 'provisioning', 'setting_up', 'running');

-- +goose Down
DROP INDEX sessions_org_active_idx;
ALTER TABLE sessions DROP COLUMN open_question_id;
ALTER TABLE session_repos
    DROP COLUMN position,
    DROP COLUMN last_diff_at,
    DROP COLUMN last_diff_truncated,
    DROP COLUMN last_diff;
DROP TABLE session_inputs;
