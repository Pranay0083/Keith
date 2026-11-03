package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/queue"
)

var deliveryIDRe = regexp.MustCompile(`^[A-Za-z0-9-]{1,128}$`)

// deliveryKey is the dedupe key of a GitHub delivery.
func deliveryKey(id string) string { return "webhook:github:delivery:" + id }

// handleWebhook is POST /hooks/github. No cookie, no Origin check: the
// HMAC signature authenticates it, and it is verified on the raw body
// before anything else is read or parsed. Handled events are deduped on
// X-GitHub-Delivery and enqueued for the worker (202); others are
// ignored (204). Processing happens in internal/hooks.
func (s *Server) handleWebhook(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, github.MaxWebhookBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			s.writeError(w, r, errorf(http.StatusRequestEntityTooLarge, codeTooLarge, "The payload is too large."))
			return
		}
		s.writeError(w, r, errBadRequest("Could not read the body."))
		return
	}
	if err := github.VerifySignature([]byte(s.cfg.GitHubWebhookSecret), body, r.Header.Get(github.SignatureHeader)); err != nil {
		s.log(r).Warn("webhook signature rejected", "err", err)
		s.writeError(w, r, errorf(http.StatusUnauthorized, codeUnauthenticated, "Invalid webhook signature."))
		return
	}

	event := github.EventType(r)
	switch event {
	case github.EventInstallation, github.EventInstallationRepositories:
	default:
		w.WriteHeader(http.StatusNoContent)
		return
	}
	delivery := github.DeliveryID(r)
	if !deliveryIDRe.MatchString(delivery) {
		s.writeError(w, r, errBadRequest("X-GitHub-Delivery is missing or malformed."))
		return
	}

	ctx := r.Context()
	first, err := s.dedupe.FirstSeen(ctx, deliveryKey(delivery), webhookDedupTTL)
	if err != nil {
		s.writeError(w, r, fmt.Errorf("webhook dedupe: %w", err))
		return
	}
	if !first {
		s.log(r).Info("webhook redelivery ignored", "delivery", delivery, "event", event)
		w.WriteHeader(http.StatusAccepted)
		return
	}
	if err := s.queue.EnqueueGitHubWebhook(ctx, queue.GitHubWebhookPayload{
		DeliveryID: delivery, Event: event, Body: body,
	}); err != nil {
		// Let a redelivery through.
		if ferr := s.dedupe.Forget(context.WithoutCancel(ctx), deliveryKey(delivery)); ferr != nil {
			s.log(r).Error("webhook dedupe forget", "delivery", delivery, "err", ferr)
		}
		s.writeError(w, r, fmt.Errorf("enqueue webhook: %w", err))
		return
	}
	s.log(r).Info("webhook queued", "delivery", delivery, "event", event)
	w.WriteHeader(http.StatusAccepted)
}

// RedisDeduper is a Deduper on Redis SET NX with a TTL.
type RedisDeduper struct {
	rdb redis.UniversalClient
}

var _ Deduper = (*RedisDeduper)(nil)

// NewRedisDeduper returns a Deduper on rdb. It does not own rdb.
func NewRedisDeduper(rdb redis.UniversalClient) *RedisDeduper { return &RedisDeduper{rdb: rdb} }

// FirstSeen implements Deduper.
func (d *RedisDeduper) FirstSeen(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	ok, err := d.rdb.SetNX(ctx, key, time.Now().UTC().Format(time.RFC3339), ttl).Result()
	if err != nil {
		return false, fmt.Errorf("api: dedupe %s: %w", key, err)
	}
	return ok, nil
}

// Forget implements Deduper.
func (d *RedisDeduper) Forget(ctx context.Context, key string) error {
	if err := d.rdb.Del(ctx, key).Err(); err != nil {
		return fmt.Errorf("api: forget %s: %w", key, err)
	}
	return nil
}
