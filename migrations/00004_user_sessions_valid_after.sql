-- Logout revokes a user's session cookies. The cookie is stateless (a
-- signed user ID, issue time and expiry), so logout records a cutoff
-- instead: a cookie issued at or before sessions_valid_after is no longer
-- accepted. NULL means no cookie has been revoked.

-- +goose Up
ALTER TABLE users ADD COLUMN sessions_valid_after timestamptz;

-- +goose Down
ALTER TABLE users DROP COLUMN sessions_valid_after;
