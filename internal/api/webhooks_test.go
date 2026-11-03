package api

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Pranay0083/keith/internal/github"
	"github.com/Pranay0083/keith/internal/redistest"
)

func sign(body string) string {
	m := hmac.New(sha256.New, []byte(testHookKey))
	m.Write([]byte(body))
	return "sha256=" + hex.EncodeToString(m.Sum(nil))
}

const installationBody = `{"action":"deleted","installation":{"id":42,"account":{"login":"acme"}}}`

func (h *harness) hook(event, delivery, body, sig string) int {
	opts := []reqOpt{withOrigin(""), withHeader(github.EventHeader, event), withHeader(github.DeliveryHeader, delivery)}
	if sig != "" {
		opts = append(opts, withHeader(github.SignatureHeader, sig))
	}
	return h.do("POST", "/hooks/github", body, opts...).Code
}

func TestWebhookSignature(t *testing.T) {
	h := newHarness(t, nil)
	for name, sig := range map[string]string{
		"missing":     "",
		"wrong key":   "sha256=" + strings.Repeat("ab", 32),
		"other body":  sign(installationBody + " "),
		"no prefix":   strings.TrimPrefix(sign(installationBody), "sha256="),
		"sha1 prefix": "sha1=" + strings.TrimPrefix(sign(installationBody), "sha256="),
	} {
		if code := h.hook("installation", "d-1", installationBody, sig); code != 401 {
			t.Errorf("%s: status %d", name, code)
		}
	}
	if n := len(h.q.Webhooks()); n != 0 {
		t.Fatalf("unsigned deliveries enqueued: %d", n)
	}
	if len(h.dedupe.seen) != 0 {
		t.Fatal("unsigned delivery touched the dedupe store")
	}
}

func TestWebhookQueueAndDedupe(t *testing.T) {
	h := newHarness(t, nil)
	sig := sign(installationBody)
	if code := h.hook("installation", "d-1", installationBody, sig); code != 202 {
		t.Fatalf("status %d", code)
	}
	whs := h.q.Webhooks()
	if len(whs) != 1 || whs[0].DeliveryID != "d-1" || whs[0].Event != "installation" || string(whs[0].Body) != installationBody {
		t.Fatalf("enqueued %+v", whs)
	}
	// A redelivery is accepted and not enqueued again.
	if code := h.hook("installation", "d-1", installationBody, sig); code != 202 {
		t.Fatalf("redelivery status %d", code)
	}
	if n := len(h.q.Webhooks()); n != 1 {
		t.Fatalf("redelivery enqueued: %d", n)
	}
	// installation_repositories is handled too.
	body := `{"action":"removed","installation":{"id":42}}`
	if code := h.hook("installation_repositories", "d-2", body, sign(body)); code != 202 {
		t.Fatalf("status %d", code)
	}
	// Other events are ignored.
	if code := h.hook("push", "d-3", `{}`, sign(`{}`)); code != 204 {
		t.Fatalf("push status %d", code)
	}
	if code := h.hook("ping", "d-4", `{}`, sign(`{}`)); code != 204 {
		t.Fatalf("ping status %d", code)
	}
	// A handled event needs a well-formed delivery ID.
	if code := h.hook("installation", "", installationBody, sig); code != 400 {
		t.Fatalf("no delivery status %d", code)
	}
	if code := h.hook("installation", "bad id!", installationBody, sig); code != 400 {
		t.Fatalf("bad delivery status %d", code)
	}
	if n := len(h.q.Webhooks()); n != 2 {
		t.Fatalf("enqueued %d, want 2", n)
	}

	// The enqueue fails: 500, and the delivery may be retried.
	h.q.SetErr(errors.New("redis down"))
	if code := h.hook("installation", "d-5", installationBody, sig); code != 500 {
		t.Fatalf("enqueue failure status %d", code)
	}
	h.q.SetErr(nil)
	if code := h.hook("installation", "d-5", installationBody, sig); code != 202 || len(h.q.Webhooks()) != 3 {
		t.Fatalf("retry after failure: %d, %d enqueued", code, len(h.q.Webhooks()))
	}
	// The dedupe store fails: 500.
	h.dedupe.err = errors.New("redis down")
	if code := h.hook("installation", "d-6", installationBody, sig); code != 500 {
		t.Fatalf("dedupe failure status %d", code)
	}
}

func TestWebhookTooLarge(t *testing.T) {
	h := newHarness(t, nil)
	body := strings.Repeat("x", github.MaxWebhookBody+1)
	if code := h.hook("installation", "d-big", body, sign(body)); code != 413 {
		t.Fatalf("status %d", code)
	}
}

func TestRedisDeduperIT(t *testing.T) {
	rdb := redistest.Client(t)
	d := NewRedisDeduper(rdb)
	ctx := context.Background()
	key := deliveryKey("it-" + randomString(8))
	t.Cleanup(func() { rdb.Del(context.Background(), key) })
	for i, want := range []bool{true, false} {
		got, err := d.FirstSeen(ctx, key, time.Minute)
		if err != nil || got != want {
			t.Fatalf("call %d: %v %v", i, got, err)
		}
	}
	if ttl := rdb.TTL(ctx, key).Val(); ttl <= 0 || ttl > time.Minute {
		t.Fatalf("ttl %v", ttl)
	}
	if err := d.Forget(ctx, key); err != nil {
		t.Fatal(err)
	}
	if got, _ := d.FirstSeen(ctx, key, time.Minute); !got {
		t.Fatal("not new after Forget")
	}
}
