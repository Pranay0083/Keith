package queue_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"

	"github.com/Pranay0083/keith/internal/queue"
	"github.com/Pranay0083/keith/internal/redistest"
)

func TestRetryPolicy(t *testing.T) {
	busy := fmt.Errorf("lease held: %w", queue.ErrBusy)
	if queue.IsFailure(busy) || !queue.IsFailure(errors.New("boom")) || queue.IsFailure(nil) {
		t.Fatal("IsFailure")
	}
	if d := queue.RetryDelay(3, busy, nil); d != queue.BusyRetryDelay {
		t.Fatalf("busy delay %v", d)
	}
	if d := queue.RetryDelay(1, errors.New("x"), asynq.NewTask("t", nil)); d < time.Second {
		t.Fatalf("default delay %v", d)
	}
}

func TestHandlerDecodesAndValidates(t *testing.T) {
	var got []queue.ResumeSessionPayload
	mux := queue.NewMux(queue.Handlers{
		ResumeSession: func(_ context.Context, p queue.ResumeSessionPayload) error {
			got = append(got, p)
			return nil
		},
	})
	ctx := context.Background()
	ok := asynq.NewTask(queue.TypeResumeSession, []byte(`{"session_id":"s1","reason":"message","input_id":"m_1"}`))
	if err := mux.ProcessTask(ctx, ok); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].SessionID != "s1" || got[0].Reason != queue.ResumeMessage || got[0].InputID != "m_1" {
		t.Fatalf("payload %+v", got)
	}
	for _, body := range []string{`not json`, `{"session_id":"s1","reason":"nope"}`, `{"session_id":"","reason":"cancel"}`,
		`{"session_id":"s","reason":"cancel","input_id":"m_1"}`, `{"session_id":"s","reason":"answer"}`} {
		err := mux.ProcessTask(ctx, asynq.NewTask(queue.TypeResumeSession, []byte(body)))
		if !errors.Is(err, asynq.SkipRetry) {
			t.Errorf("%s: err = %v, want SkipRetry", body, err)
		}
	}
	// Unregistered types fail.
	if err := mux.ProcessTask(ctx, asynq.NewTask(queue.TypeRunSession, []byte(`{"session_id":"s"}`))); err == nil {
		t.Error("unregistered handler succeeded")
	}
}

func TestPayloadJSON(t *testing.T) {
	b, _ := json.Marshal(queue.GitHubWebhookPayload{DeliveryID: "d", Event: "installation", Body: []byte(`{"a":1}`)})
	if !strings.Contains(string(b), `"delivery_id":"d"`) || !strings.Contains(string(b), `"event":"installation"`) {
		t.Fatalf("webhook payload %s", b)
	}
	var back queue.GitHubWebhookPayload
	if err := json.Unmarshal(b, &back); err != nil || string(back.Body) != `{"a":1}` {
		t.Fatalf("round trip %+v %v", back, err)
	}
}

// itRedis returns a client on a Redis database of its own, so these tests
// cannot take tasks other packages' tests enqueue. The database is
// flushed before and after.
func itRedis(t *testing.T) *redis.Client {
	t.Helper()
	base := redistest.Client(t)
	opt := base.Options()
	o := *opt
	o.DB = 13
	rdb := redis.NewClient(&o)
	ctx := context.Background()
	if err := rdb.FlushDB(ctx).Err(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		rdb.FlushDB(context.Background())
		rdb.Close()
	})
	return rdb
}

func TestClientEnqueueIT(t *testing.T) {
	rdb := itRedis(t)
	c := queue.NewClient(rdb)
	defer c.Close()
	ctx := context.Background()
	insp := asynq.NewInspectorFromRedisClient(rdb)

	// run_session: once per session.
	for range 2 {
		if err := c.EnqueueRunSession(ctx, queue.RunSessionPayload{SessionID: "s1"}); err != nil {
			t.Fatal(err)
		}
	}
	info, err := insp.GetTaskInfo(queue.QueueSessions, queue.RunSessionTaskID("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Type != queue.TypeRunSession || info.MaxRetry != queue.SessionMaxRetry || info.Retention != queue.Retention ||
		info.Timeout != queue.SessionTimeout {
		t.Fatalf("run task %+v", info)
	}

	// resume_session: deduped per payload.
	msg := queue.ResumeSessionPayload{SessionID: "s1", Reason: queue.ResumeMessage, InputID: "m_1"}
	for _, p := range []queue.ResumeSessionPayload{
		msg, msg,
		{SessionID: "s1", Reason: queue.ResumeMessage, InputID: "m_2"},
		{SessionID: "s1", Reason: queue.ResumeCancel},
		{SessionID: "s1", Reason: queue.ResumeCancel},
	} {
		if err := c.EnqueueResumeSession(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	pending, err := insp.ListPendingTasks(queue.QueueSessions)
	if err != nil {
		t.Fatal(err)
	}
	resumes := 0
	for _, ti := range pending {
		if ti.Type == queue.TypeResumeSession {
			resumes++
		}
	}
	if resumes != 3 {
		t.Fatalf("resume tasks = %d, want 3", resumes)
	}

	// github_webhook: once per delivery.
	wh := queue.GitHubWebhookPayload{DeliveryID: "d1", Event: "installation", Body: []byte(`{}`)}
	for range 2 {
		if err := c.EnqueueGitHubWebhook(ctx, wh); err != nil {
			t.Fatal(err)
		}
	}
	if whs, _ := insp.ListPendingTasks(queue.QueueWebhooks); len(whs) != 1 {
		t.Fatalf("webhook tasks = %d", len(whs))
	}

	// Invalid payloads are refused before Redis.
	if err := c.EnqueueRunSession(ctx, queue.RunSessionPayload{}); err == nil {
		t.Error("empty run accepted")
	}
	if err := c.EnqueueResumeSession(ctx, queue.ResumeSessionPayload{SessionID: "s", Reason: queue.ResumeAnswer}); err == nil {
		t.Error("answer without input accepted")
	}
	if err := c.EnqueueGitHubWebhook(ctx, queue.GitHubWebhookPayload{Event: "x"}); err == nil {
		t.Error("webhook without delivery accepted")
	}
}

func TestServerRunsTasksIT(t *testing.T) {
	rdb := itRedis(t)
	c := queue.NewClient(rdb)
	defer c.Close()
	ctx := context.Background()

	var mu sync.Mutex
	var runs []string
	busyOnce := true
	done := make(chan struct{}, 4)
	srv := queue.NewServer(rdb, queue.ServerConfig{
		Concurrency: 2,
		Logger:      slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	mux := queue.NewMux(queue.Handlers{
		RunSession: func(_ context.Context, p queue.RunSessionPayload) error {
			mu.Lock()
			defer mu.Unlock()
			if busyOnce {
				busyOnce = false
				return fmt.Errorf("held: %w", queue.ErrBusy)
			}
			runs = append(runs, p.SessionID)
			done <- struct{}{}
			return nil
		},
		GitHubWebhook: func(_ context.Context, p queue.GitHubWebhookPayload) error {
			mu.Lock()
			defer mu.Unlock()
			runs = append(runs, "hook:"+p.DeliveryID+":"+string(p.Body))
			done <- struct{}{}
			return nil
		},
	})
	if err := srv.Start(mux); err != nil {
		t.Fatal(err)
	}
	defer srv.Shutdown()

	if err := c.EnqueueGitHubWebhook(ctx, queue.GitHubWebhookPayload{DeliveryID: "d9", Event: "installation", Body: []byte(`{"x":1}`)}); err != nil {
		t.Fatal(err)
	}
	if err := c.EnqueueRunSession(ctx, queue.RunSessionPayload{SessionID: "s9"}); err != nil {
		t.Fatal(err)
	}
	// The busy run is retried after BusyRetryDelay (plus asynq's 5 s
	// forwarder tick).
	deadline := time.After(30 * time.Second)
	for range 2 {
		select {
		case <-done:
		case <-deadline:
			mu.Lock()
			defer mu.Unlock()
			t.Fatalf("tasks did not finish; ran %v", runs)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	got := strings.Join(runs, ",")
	if !strings.Contains(got, "hook:d9:{\"x\":1}") || !strings.Contains(got, "s9") {
		t.Fatalf("runs = %v", runs)
	}
	info, err := asynq.NewInspectorFromRedisClient(rdb).GetTaskInfo(queue.QueueSessions, queue.RunSessionTaskID("s9"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Retried != 0 {
		t.Fatalf("busy retry counted as failure: retried = %d", info.Retried)
	}
}
