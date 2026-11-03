-- Why a session ended up failed, cancelled or no_changes, so lists and
-- the session page can show it without scanning events.

-- +goose Up
ALTER TABLE sessions ADD COLUMN status_reason text NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE sessions DROP COLUMN status_reason;
