package queue

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/hibiken/asynq"
)

// SessionErrorHandler returns a ServerConfig.ErrorHandler that calls
// onFinal for a run_session or resume_session task whose error ends it:
// its retries are used up, so asynq is about to archive it. (A payload
// error, which wraps asynq.SkipRetry, never ran the session and is
// ignored.) asynq archives an exhausted task even for ErrBusy,
// which otherwise does not count as a failure, so onFinal also sees
// errors wrapping ErrBusy. Nothing would then run the session again;
// onFinal records the failure or enqueues a fresh wakeup.
//
// onFinal runs before the task is archived, on the task's context, which
// may already be done.
func SessionErrorHandler(onFinal func(ctx context.Context, sessionID string, err error)) func(context.Context, *asynq.Task, error) {
	return func(ctx context.Context, t *asynq.Task, err error) {
		id := sessionIDOf(t)
		if id == "" || err == nil {
			return
		}
		retried, _ := asynq.GetRetryCount(ctx)
		max, _ := asynq.GetMaxRetry(ctx)
		if !errors.Is(err, asynq.SkipRetry) && FinalAttempt(retried, max) {
			onFinal(ctx, id, err)
		}
	}
}

// FinalAttempt reports whether a task that failed after retried retries
// (of max) is archived rather than retried, whatever the error.
func FinalAttempt(retried, max int) bool { return retried >= max }

// sessionIDOf returns the session ID of a run_session or resume_session
// task, or "".
func sessionIDOf(t *asynq.Task) string {
	if t == nil || (t.Type() != TypeRunSession && t.Type() != TypeResumeSession) {
		return ""
	}
	var p struct {
		SessionID string `json:"session_id"`
	}
	if json.Unmarshal(t.Payload(), &p) != nil {
		return ""
	}
	return p.SessionID
}
