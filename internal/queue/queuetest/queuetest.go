// Package queuetest is an in-memory queue.Enqueuer for tests.
package queuetest

import (
	"context"
	"sync"

	"github.com/Pranay0083/keith/internal/queue"
)

// Recorder records enqueued tasks. Set Err to make every enqueue fail.
// The zero value is ready to use.
type Recorder struct {
	mu      sync.Mutex
	Err     error
	run     []queue.RunSessionPayload
	resume  []queue.ResumeSessionPayload
	webhook []queue.GitHubWebhookPayload
}

var _ queue.Enqueuer = (*Recorder)(nil)

// SetErr sets the error every later enqueue returns.
func (r *Recorder) SetErr(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Err = err
}

// EnqueueRunSession implements queue.Enqueuer.
func (r *Recorder) EnqueueRunSession(_ context.Context, p queue.RunSessionPayload) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Err != nil {
		return r.Err
	}
	r.run = append(r.run, p)
	return nil
}

// EnqueueResumeSession implements queue.Enqueuer.
func (r *Recorder) EnqueueResumeSession(_ context.Context, p queue.ResumeSessionPayload) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Err != nil {
		return r.Err
	}
	r.resume = append(r.resume, p)
	return nil
}

// EnqueueGitHubWebhook implements queue.Enqueuer.
func (r *Recorder) EnqueueGitHubWebhook(_ context.Context, p queue.GitHubWebhookPayload) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Err != nil {
		return r.Err
	}
	r.webhook = append(r.webhook, p)
	return nil
}

// Runs returns the run_session payloads so far.
func (r *Recorder) Runs() []queue.RunSessionPayload {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]queue.RunSessionPayload(nil), r.run...)
}

// Resumes returns the resume_session payloads so far.
func (r *Recorder) Resumes() []queue.ResumeSessionPayload {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]queue.ResumeSessionPayload(nil), r.resume...)
}

// Webhooks returns the github_webhook payloads so far.
func (r *Recorder) Webhooks() []queue.GitHubWebhookPayload {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]queue.GitHubWebhookPayload(nil), r.webhook...)
}
