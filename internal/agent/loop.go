package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
)

// Input is a user message to hand to the model: the session's prompt, a
// follow-up, a message sent mid-run or the answer to a question.
type Input struct {
	// ID is the message's ID, sent as message.user's message_id. Inputs
	// whose ID was already delivered are skipped.
	ID string
	// UserID is the author's users.id; empty for trigger messages.
	UserID string
	Text   string
	// QuestionID is set when the message answers a question.asked.
	QuestionID string
}

// OutcomeKind says why Runner.Run returned.
type OutcomeKind string

const (
	// OutcomeEndTurn: the model finished its turn and no user input is
	// waiting. Outcome.Text is its final message.
	OutcomeEndTurn OutcomeKind = "end_turn"
	// OutcomePaused: the session waits for the user to answer
	// Outcome.Question (status waiting_for_user).
	OutcomePaused OutcomeKind = "paused"
	// OutcomeCancelled: Config.ShouldStop reported a stop request.
	OutcomeCancelled OutcomeKind = "cancelled"
	// OutcomeStepCap: this run made Config.MaxSteps model calls.
	OutcomeStepCap OutcomeKind = "step_cap"
	// OutcomeBudgetExceeded: the session's usage reached
	// Config.TokenBudget.
	OutcomeBudgetExceeded OutcomeKind = "budget_exceeded"
	// OutcomeFailed: the run could not continue; see Outcome.Reason.
	OutcomeFailed OutcomeKind = "failed"
)

// Failure reasons (Outcome.Reason for OutcomeFailed). They are stable,
// machine-readable codes suitable for sessions.status_reason and the error
// event's code.
const (
	// ReasonContextExhausted: the conversation no longer fits the model's
	// context window (llm.ErrContextTooLong).
	ReasonContextExhausted = "context_exhausted"
	// ReasonLLMError: the model call failed (after retries for
	// retryable errors).
	ReasonLLMError = "llm_error"
	// ReasonRefusal: the model declined to continue. Outcome.Text holds
	// whatever it said.
	ReasonRefusal = "refusal"
	// ReasonToolError: a tool returned a non-nil error (the sandbox is
	// gone, a hook failed).
	ReasonToolError = "tool_error"
	// ReasonInterrupted: ctx was cancelled (worker shutdown, lost lease).
	ReasonInterrupted = "interrupted"
	// ReasonInputsFailed: Config.Inputs returned an error.
	ReasonInputsFailed = "inputs_failed"
	// ReasonCheckpointFailed: Config.Checkpoint returned an error.
	ReasonCheckpointFailed = "checkpoint_failed"
	// ReasonEmitFailed: Run.Emit failed for a stored event.
	ReasonEmitFailed = "emit_failed"
)

// Outcome is the result of Runner.Run. Whatever the kind, the State
// passed to Run is consistent and must be saved: a later Run continues
// from it.
type Outcome struct {
	Kind OutcomeKind
	// Text is the model's final message for OutcomeEndTurn (and what it
	// said before refusing, for ReasonRefusal).
	Text string
	// Question is the open question for OutcomePaused.
	Question *Question
	// Reason is one of the Reason constants for OutcomeFailed.
	Reason string
	// Err is the underlying error for OutcomeFailed.
	Err error
	// Steps is the number of model calls this run made.
	Steps int
	// Usage is the token usage of this run's model calls.
	Usage llm.Usage
}

// RetryPolicy controls retries of retryable model errors
// (llm.IsRetryable).
type RetryPolicy struct {
	// Attempts is the total number of tries per model call (default 3).
	Attempts int
	// Base is the first backoff; each retry doubles it (default 2s).
	Base time.Duration
	// Max caps the doubling backoff (default 30s).
	Max time.Duration
	// MaxRetryAfter caps a provider's RetryAfter hint, which otherwise
	// wins over a shorter backoff (default 2m).
	MaxRetryAfter time.Duration
}

// Config configures a Runner.
type Config struct {
	// Provider and Model are the session's pinned model.
	Provider llm.Provider
	Model    string
	// Tools are the tools the model may call, in the order the model
	// sees them (DefaultTools). The list must not change during a
	// session: it is part of the cached prefix.
	Tools []Tool
	// Effort is llm.Request.Effort, for example llm.EffortHigh.
	Effort string
	// MaxTokens caps each call's output; 0 means the provider default.
	MaxTokens int

	// MaxSteps caps the model calls of one Run (0 means 100). Each Run
	// (each user message or answer) gets a fresh allowance.
	MaxSteps int
	// TokenBudget caps the session's cumulative usage (Aux.Usage, as
	// measured by Cost). Run returns OutcomeBudgetExceeded before a model
	// call once it is reached. 0 means no budget.
	TokenBudget int
	// Cost turns usage into budget units; nil sums all four token counts.
	Cost func(llm.Usage) int

	// Inputs is polled between steps for user messages to hand to the
	// model. Each one is emitted as message.user when handed over. nil
	// means none.
	Inputs func(ctx context.Context) ([]Input, error)
	// ShouldStop is checked between steps and between tool calls; true
	// ends the run with OutcomeCancelled. The worker also interrupts the
	// running shell command so the current step ends promptly.
	ShouldStop func(ctx context.Context) bool
	// Delivered, if set, is called after the message.user events for
	// newly delivered inputs were emitted and before the model sees them.
	// The worker emits session.status running there when a paused
	// session's question was answered (docs/api.md ordering).
	Delivered func(ctx context.Context, inputs []Input) error
	// Checkpoint, if set, is called whenever the State changed in a way
	// worth saving: after each model response, after each batch of tool
	// calls, and after answering tool calls a stopped worker left open.
	Checkpoint func(ctx context.Context, st *State) error

	// Repeat configures repeat detection.
	Repeat RepeatPolicy
	// Retry configures model call retries.
	Retry RetryPolicy
	// Kickoff is the first user message when a session starts with no
	// input (the task is in the system prompt). Default DefaultKickoff.
	Kickoff string

	// Sleep waits between retries; nil uses a timer. Tests use it to
	// avoid real waits.
	Sleep func(ctx context.Context, d time.Duration) error
	// Now is the clock for tool durations; nil means time.Now.
	Now func() time.Time
	// NewID returns a new unique ID with the given prefix (message IDs);
	// nil uses random hex.
	NewID func(prefix string) string
}

// DefaultKickoff is the first user message of a session that starts
// without input.
const DefaultKickoff = "Start working on the task described in the system prompt."

// DefaultMaxSteps is Config.MaxSteps' default.
const DefaultMaxSteps = 100

// Runner runs the agent loop. It holds no per-session state, so one
// Runner can serve many sessions, one Run at a time per session.
type Runner struct {
	cfg   Config
	tools map[string]Tool
	specs []llm.ToolSpec
}

// NewRunner checks cfg and returns a Runner.
func NewRunner(cfg Config) (*Runner, error) {
	if cfg.Provider == nil {
		return nil, errors.New("agent: Config.Provider is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("agent: Config.Model is required")
	}
	r := &Runner{cfg: cfg, tools: map[string]Tool{}}
	for _, t := range cfg.Tools {
		spec := t.Spec()
		if spec.Name == "" {
			return nil, errors.New("agent: a tool has an empty name")
		}
		if _, dup := r.tools[spec.Name]; dup {
			return nil, fmt.Errorf("agent: duplicate tool %q", spec.Name)
		}
		r.tools[spec.Name] = t
		r.specs = append(r.specs, spec)
	}
	if r.cfg.MaxSteps <= 0 {
		r.cfg.MaxSteps = DefaultMaxSteps
	}
	if r.cfg.Cost == nil {
		r.cfg.Cost = func(u llm.Usage) int {
			return u.InputTokens + u.OutputTokens + u.CacheReadTokens + u.CacheWriteTokens
		}
	}
	if r.cfg.Kickoff == "" {
		r.cfg.Kickoff = DefaultKickoff
	}
	if r.cfg.Sleep == nil {
		r.cfg.Sleep = sleep
	}
	if r.cfg.Now == nil {
		r.cfg.Now = time.Now
	}
	if r.cfg.NewID == nil {
		r.cfg.NewID = randomID
	}
	r.cfg.Retry = r.cfg.Retry.withDefaults()
	r.cfg.Repeat = r.cfg.Repeat.withDefaults()
	return r, nil
}

// ToolSpecs returns the tool specs the Runner sends, in order.
func (r *Runner) ToolSpecs() []llm.ToolSpec {
	return append([]llm.ToolSpec(nil), r.specs...)
}

func (p RetryPolicy) withDefaults() RetryPolicy {
	if p.Attempts <= 0 {
		p.Attempts = 3
	}
	if p.Base <= 0 {
		p.Base = 2 * time.Second
	}
	if p.Max <= 0 {
		p.Max = 30 * time.Second
	}
	if p.MaxRetryAfter <= 0 {
		p.MaxRetryAfter = 2 * time.Minute
	}
	return p
}

// backoff is the wait before retry number n (1-based) after err.
func (p RetryPolicy) backoff(n int, err error) time.Duration {
	d := p.Base
	for i := 1; i < n && d < p.Max; i++ {
		d *= 2
	}
	d = min(d, p.Max)
	var le *llm.Error
	if errors.As(err, &le) && le.RetryAfter > d {
		d = min(le.RetryAfter, p.MaxRetryAfter)
	}
	return d
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func randomID(prefix string) string {
	var b [10]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

// errEmptyResponse is a model response with no content blocks. It is
// retried like a retryable error and never appended to the conversation.
var errEmptyResponse = errors.New("agent: model returned an empty response")

// Run runs the loop on st until the model ends its turn with no user input
// waiting, the session pauses, a limit is reached or something fails.
//
// Each step: check ShouldStop, poll Inputs (emitting message.user and
// answering an open question), call the model with the pending user turn,
// append the request's new user message and the response to
// st.Messages, emit message.delta/message.assistant, then run the
// response's tool calls in order (tool_call.started/finished) and collect
// their results, with any new inputs, into the next pending user turn.
//
// Before the first step, Run answers tool calls a stopped worker left open
// (see answerOpenCalls), so the model sees an error result for each.
//
// st.Messages is only ever appended to; see State. run must not be used
// by another Run at the same time.
func (r *Runner) Run(ctx context.Context, run *Run, st *State) (out Outcome) {
	if st.Messages == nil {
		st.Messages = []llm.Message{}
	}
	run.state = st
	steps, usage := 0, llm.Usage{}
	defer func() {
		run.state = nil
		out.Steps, out.Usage = steps, usage
	}()

	if ctx.Err() == nil {
		if err := r.answerOpenCalls(ctx, run, st); err != nil {
			if errors.Is(err, errEmitFailed) {
				return failed(ReasonEmitFailed, err)
			}
			return checkpointFailed(ctx, err)
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			return failed(ReasonInterrupted, err)
		}
		if r.stopRequested(ctx) {
			return Outcome{Kind: OutcomeCancelled}
		}

		inputs, err := r.poll(ctx)
		if err != nil {
			return failed(ReasonInputsFailed, err)
		}
		if err := r.deliver(ctx, run, st, inputs); err != nil {
			if errors.Is(err, errEmitFailed) {
				return failed(ReasonEmitFailed, err)
			}
			return failed(ReasonInputsFailed, err)
		}
		if q := st.OpenQuestion(); q != nil {
			return Outcome{Kind: OutcomePaused, Question: q}
		}

		if st.Aux.Pending.empty() && !st.Aux.ContinueTurn {
			if len(st.Messages) > 0 {
				st.Aux.Pending = nil
				return Outcome{Kind: OutcomeEndTurn, Text: st.LastAssistantText()}
			}
			p := st.pending()
			p.Texts = append(p.Texts, llm.TextBlock(r.cfg.Kickoff))
		}
		if steps >= r.cfg.MaxSteps {
			return Outcome{Kind: OutcomeStepCap}
		}
		if r.cfg.TokenBudget > 0 && r.cfg.Cost(st.Aux.Usage) >= r.cfg.TokenBudget {
			return Outcome{Kind: OutcomeBudgetExceeded}
		}

		resp, err := r.generate(ctx, run, st)
		if err != nil {
			switch {
			case ctx.Err() != nil:
				return failed(ReasonInterrupted, err)
			case isKind(err, llm.ErrContextTooLong):
				return failed(ReasonContextExhausted, err)
			case errors.Is(err, errEmitFailed):
				return failed(ReasonEmitFailed, err)
			}
			return failed(ReasonLLMError, err)
		}
		steps++
		usage.Add(resp.Usage)
		if err := r.checkpoint(ctx, st); err != nil {
			return checkpointFailed(ctx, err)
		}

		if uses := resp.Message.ToolUses(); len(uses) > 0 {
			err := r.runTools(ctx, run, st, uses)
			// question.asked goes out after the whole batch, so the
			// worker's session.status waiting_for_user follows it
			// directly.
			if q := st.OpenQuestion(); q != nil {
				if eerr := run.emit(ctx, events.TypeQuestionAsked, events.QuestionAskedPayload{
					QuestionID: q.ID, Question: q.Text, Choices: q.Choices,
				}); eerr != nil && err == nil {
					err = eerr
				}
			}
			if err != nil {
				if errors.Is(err, errEmitFailed) {
					return failed(ReasonEmitFailed, err)
				}
				if ctx.Err() != nil {
					return failed(ReasonInterrupted, err)
				}
				return failed(ReasonToolError, err)
			}
			if err := r.checkpoint(ctx, st); err != nil {
				return checkpointFailed(ctx, err)
			}
			if q := st.OpenQuestion(); q != nil {
				return Outcome{Kind: OutcomePaused, Question: q}
			}
			continue
		}

		switch resp.StopReason {
		case llm.StopRefusal:
			return Outcome{Kind: OutcomeFailed, Reason: ReasonRefusal, Text: resp.Message.Text(),
				Err: errors.New("agent: the model declined to continue")}
		case llm.StopMaxTokens:
			p := st.pending()
			p.Texts = append(p.Texts, llm.TextBlock(maxTokensNote))
		case llm.StopPauseTurn:
			st.Aux.ContinueTurn = true
		}
	}
}

const maxTokensNote = "(Your last response hit the output token limit and was cut off. Continue where you left off, in shorter pieces.)"

func failed(reason string, err error) Outcome {
	return Outcome{Kind: OutcomeFailed, Reason: reason, Err: err}
}

func checkpointFailed(ctx context.Context, err error) Outcome {
	if ctx.Err() != nil {
		return failed(ReasonInterrupted, err)
	}
	return failed(ReasonCheckpointFailed, err)
}

func isKind(err error, kind llm.ErrorKind) bool {
	var le *llm.Error
	return errors.As(err, &le) && le.Kind == kind
}

func (r *Runner) stopRequested(ctx context.Context) bool {
	return r.cfg.ShouldStop != nil && r.cfg.ShouldStop(ctx)
}

func (r *Runner) poll(ctx context.Context) ([]Input, error) {
	if r.cfg.Inputs == nil {
		return nil, nil
	}
	return r.cfg.Inputs(ctx)
}

func (r *Runner) checkpoint(ctx context.Context, st *State) error {
	if r.cfg.Checkpoint == nil {
		return nil
	}
	return r.cfg.Checkpoint(ctx, st)
}

// deliver hands new inputs to the model: it emits message.user for each,
// answers the open question with the matching input (or the first one)
// and queues the rest as text blocks of the pending user turn.
func (r *Runner) deliver(ctx context.Context, run *Run, st *State, inputs []Input) error {
	var fresh []Input
	for _, in := range inputs {
		if in.ID != "" && st.Aux.delivered(in.ID) {
			continue
		}
		fresh = append(fresh, in)
	}
	if len(fresh) == 0 {
		return nil
	}
	p := st.pending()
	answer := -1
	if q := p.Question; q != nil {
		answer = 0
		for i, in := range fresh {
			if in.QuestionID == q.ID {
				answer = i
				break
			}
		}
	}
	delivered := make([]Input, 0, len(fresh))
	for i, in := range fresh {
		text := storableText(in.Text)
		if text == "" {
			text = "(empty message)"
		}
		qid := in.QuestionID
		if i == answer {
			q := p.Question
			qid = q.ID
			if q.ToolUseID != "" {
				for j, b := range p.Results {
					if b.ToolUseID == q.ToolUseID {
						p.Results[j] = llm.ToolResultBlock(q.ToolUseID, false, llm.TextBlock(text))
					}
				}
			} else {
				p.Texts = append(p.Texts, llm.TextBlock(text))
			}
			p.Question = nil
		} else {
			p.Texts = append(p.Texts, llm.TextBlock(text))
		}
		st.Aux.markDelivered(in.ID)
		if err := run.emit(ctx, events.TypeMessageUser, events.MessageUserPayload{
			MessageID: in.ID, UserID: in.UserID, Text: in.Text, QuestionID: qid,
		}); err != nil {
			return err
		}
		in.QuestionID = qid
		delivered = append(delivered, in)
	}
	if r.cfg.Delivered != nil {
		return r.cfg.Delivered(ctx, delivered)
	}
	return nil
}

// generate calls the model with the conversation plus the pending user
// turn, retrying retryable errors. On success it appends the sent user
// message and the response to st.Messages, records usage and the raw
// response, and emits message.assistant if the response has text.
func (r *Runner) generate(ctx context.Context, run *Run, st *State) (*llm.Response, error) {
	msgs := st.Messages[:len(st.Messages):len(st.Messages)]
	var sent *llm.Message
	if p := st.Aux.Pending; !p.empty() {
		content := make([]llm.Block, 0, len(p.Results)+len(p.Texts))
		content = append(content, p.Results...)
		content = append(content, p.Texts...)
		sent = &llm.Message{Role: llm.RoleUser, Content: content}
		msgs = append(msgs, *sent)
	}
	req := llm.Request{
		Model:               r.cfg.Model,
		System:              st.System,
		Messages:            msgs,
		Tools:               r.specs,
		MaxTokens:           r.cfg.MaxTokens,
		Effort:              r.cfg.Effort,
		CacheSystemAndTools: true,
	}

	var (
		resp  *llm.Response
		msgID string
		err   error
	)
	for attempt := 1; ; attempt++ {
		msgID = r.cfg.NewID("msg")
		id := msgID
		resp, err = r.cfg.Provider.Generate(ctx, req, func(ev llm.StreamEvent) {
			if ev.Type == llm.StreamTextDelta && ev.Text != "" {
				run.emitEphemeral(ctx, events.TypeMessageDelta, events.MessageDeltaPayload{MessageID: id, Text: ev.Text})
			}
		})
		if err == nil && (resp == nil || len(resp.Message.Content) == 0) {
			err = errEmptyResponse
		}
		if err == nil {
			break
		}
		if ctx.Err() != nil || attempt >= r.cfg.Retry.Attempts ||
			!(llm.IsRetryable(err) || errors.Is(err, errEmptyResponse)) {
			return nil, err
		}
		if serr := r.cfg.Sleep(ctx, r.cfg.Retry.backoff(attempt, err)); serr != nil {
			return nil, serr
		}
	}

	msg := resp.Message
	msg.Role = llm.RoleAssistant
	if sent != nil {
		st.Messages = append(st.Messages, *sent)
	}
	st.Messages = append(st.Messages, msg)
	st.Aux.Pending = nil
	st.Aux.ContinueTurn = false
	st.Aux.Steps++
	st.Aux.Usage.Add(resp.Usage)
	if len(resp.Raw) > 0 {
		st.Aux.Raw = append(st.Aux.Raw, RawResponse{Step: st.Aux.Steps, Raw: resp.Raw})
	}
	if text := msg.Text(); text != "" {
		if err := run.emit(ctx, events.TypeMessageAssistant, events.MessageAssistantPayload{MessageID: msgID, Text: text}); err != nil {
			return nil, err
		}
	}
	return resp, nil
}

// lostResultNote is the tool_result of a tool call whose result was lost
// because the worker stopped while it ran.
const lostResultNote = "Interrupted: the worker stopped while this tool call was running, so its result was lost. Check the current state (files, running processes) before relying on it or running it again."

// answerOpenCalls gives every tool_use of the conversation's last message
// exactly one pending tool_result, in tool_use order.
//
// The loop checkpoints after a model call and again after the whole tool
// batch. A worker that crashes or loses its lease while a tool runs
// leaves the first checkpoint: the assistant message with its tool_use
// blocks and no results. Without this, Run would end the turn at once,
// and every later request would carry a tool_use with no tool_result,
// which the providers reject.
//
// A result already held for a call is kept. Each missing one becomes an
// error placeholder, announced with tool_call.finished (summary
// "interrupted"), and the state is checkpointed. Pending texts and the
// open question are left alone. In normal flows (a pause, a finished
// batch, a batch interrupted under a live Run) every call already has a
// result and nothing changes.
func (r *Runner) answerOpenCalls(ctx context.Context, run *Run, st *State) error {
	n := len(st.Messages)
	if n == 0 || st.Messages[n-1].Role != llm.RoleAssistant {
		return nil
	}
	uses := st.Messages[n-1].ToolUses()
	if len(uses) == 0 {
		return nil
	}
	var held []llm.Block
	if st.Aux.Pending != nil {
		held = st.Aux.Pending.Results
	}
	results := make([]llm.Block, 0, len(uses))
	var lost []llm.Block
	for _, u := range uses {
		i := slices.IndexFunc(held, func(b llm.Block) bool { return b.ToolUseID == u.ID })
		if i >= 0 {
			results = append(results, held[i])
			continue
		}
		results = append(results, llm.ToolResultBlock(u.ID, true, llm.TextBlock(lostResultNote)))
		lost = append(lost, u)
	}
	same := len(held) == len(results)
	for i := 0; same && i < len(held); i++ {
		same = held[i].ToolUseID == results[i].ToolUseID
	}
	if same {
		return nil
	}
	// The events go out before the state changes: if one fails, the next
	// Run announces every placeholder again rather than none.
	for _, u := range lost {
		if err := run.emit(ctx, events.TypeToolCallFinished, events.ToolCallFinishedPayload{
			CallID: u.ID, Tool: u.Name, IsError: true, Summary: "interrupted",
		}); err != nil {
			return err
		}
	}
	st.pending().Results = results
	return r.checkpoint(ctx, st)
}

// runTools runs the tool calls of one assistant message in order and adds
// their results to the pending user turn. Every tool_use gets exactly one
// result, even when the run stops early.
func (r *Runner) runTools(ctx context.Context, run *Run, st *State, uses []llm.Block) error {
	p := st.pending()
	notRun := func(rest []llm.Block, why string) {
		for _, u := range rest {
			p.Results = append(p.Results, llm.ToolResultBlock(u.ID, true, llm.TextBlock(why)))
		}
	}
	for i, u := range uses {
		if ctx.Err() != nil {
			notRun(uses[i:], "Not run: the session was interrupted.")
			return nil
		}
		if r.stopRequested(ctx) {
			notRun(uses[i:], "Not run: the user stopped the session.")
			return nil
		}
		if err := r.callTool(ctx, run, st, u); err != nil {
			if ctx.Err() != nil {
				notRun(uses[i:i+1], lostResultNote)
				notRun(uses[i+1:], "Not run: the session was interrupted.")
				return err
			}
			notRun(uses[i:], "Not run: the session stopped because of an internal error.")
			return err
		}
	}
	return nil
}

// finishTimeout bounds emitting tool_call.finished once the run's context
// is done, so the call does not stay running in the timeline.
const finishTimeout = 5 * time.Second

// emitFinished emits tool_call.finished. If ctx is done (the worker is
// shutting down), it uses a short context detached from ctx's
// cancellation; the lease check in the worker's Emit still applies.
func (r *Run) emitFinished(ctx context.Context, p events.ToolCallFinishedPayload) error {
	if ctx.Err() != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.WithoutCancel(ctx), finishTimeout)
		defer cancel()
	}
	return r.emit(ctx, events.TypeToolCallFinished, p)
}

// callTool runs one tool call and appends its result to the pending turn.
func (r *Runner) callTool(ctx context.Context, run *Run, st *State, u llm.Block) error {
	p := st.pending()
	input := u.Input
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	if err := run.emit(ctx, events.TypeToolCallStarted, events.ToolCallStartedPayload{
		CallID: u.ID, Tool: u.Name, Input: input,
	}); err != nil {
		return err
	}
	start := r.cfg.Now()

	tool, ok := r.tools[u.Name]
	var (
		res Result
		err error
	)
	switch {
	case !ok:
		res = errorResult(fmt.Sprintf("Unknown tool %q.", u.Name))
	case p.Question != nil && pauses(tool):
		res = errorResult("Not asked: the session is already waiting for the user to answer another question. Ask this after they reply.")
	default:
		res, err = execute(withCallID(ctx, u.ID), run, tool, input)
	}
	finished := events.ToolCallFinishedPayload{CallID: u.ID, Tool: u.Name}
	if err != nil {
		finished.IsError = true
		finished.Summary = "internal error"
		if ctx.Err() != nil {
			finished.Summary = "interrupted"
		}
		finished.DurationMS = r.cfg.Now().Sub(start).Milliseconds()
		_ = run.emitFinished(ctx, finished)
		return err
	}

	if res.Pause && p.Question == nil {
		q := res.question
		if q == nil {
			q = &Question{ID: u.ID, ToolUseID: u.ID, Text: res.Summary}
		}
		p.Question = q
	}

	switch v := r.cfg.Repeat.observe(&st.Aux.Repeat, u.Name, input, res, run.Repos); {
	case v.pause && p.Question == nil:
		p.Question = &Question{ID: "stuck_" + u.ID, Text: v.question, Auto: true}
		res.Content = append(res.Content, llm.TextBlock(v.note))
	case v.note != "":
		res.Content = append(res.Content, llm.TextBlock(v.note))
	}

	if len(res.Content) == 0 {
		res.Content = []llm.Block{llm.TextBlock("(no output)")}
	}
	res.Content = storableBlocks(res.Content)
	finished.IsError = res.IsError
	finished.ExitCode = res.ExitCode
	finished.TimedOut = res.TimedOut
	finished.Summary = res.Summary
	finished.DurationMS = r.cfg.Now().Sub(start).Milliseconds()
	if err := run.emitFinished(ctx, finished); err != nil {
		return err
	}
	p.Results = append(p.Results, llm.ToolResultBlock(u.ID, res.IsError, res.Content...))
	return nil
}

// execute runs a tool, turning a panic into an error result so one buggy
// tool can't take the worker down.
func execute(ctx context.Context, run *Run, tool Tool, input json.RawMessage) (res Result, err error) {
	defer func() {
		if v := recover(); v != nil {
			res, err = errorResult(fmt.Sprintf("Internal error: the tool crashed (%v).", v)), nil
		}
	}()
	return tool.Execute(ctx, run, input)
}

// pauser is implemented by tools that pause the session (ask_user).
type pauser interface{ pausesSession() }

func pauses(t Tool) bool {
	_, ok := t.(pauser)
	return ok
}

type callIDKey struct{}

func withCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, callIDKey{}, id)
}

// CallID returns the ID of the tool call ctx belongs to (the model's
// tool_use ID), or "" outside a tool call. Tools use it for event
// payloads.
func CallID(ctx context.Context) string {
	id, _ := ctx.Value(callIDKey{}).(string)
	return id
}

// storableText replaces NUL bytes, which Postgres text and jsonb cannot
// hold, so the conversation can always be saved (a binary file read or a
// program printing \0 would otherwise make every checkpoint fail). It is
// applied before content enters the conversation, so what is saved is
// exactly what the model saw.
func storableText(s string) string {
	if strings.IndexByte(s, 0) < 0 {
		return s
	}
	return strings.ReplaceAll(s, "\x00", "\uFFFD")
}

// storableBlocks applies storableText to the text of blocks.
func storableBlocks(bs []llm.Block) []llm.Block {
	for i := range bs {
		bs[i].Text = storableText(bs[i].Text)
	}
	return bs
}

// errEmitFailed wraps a failed Run.Emit.
var errEmitFailed = errors.New("agent: emit failed")

// emit sends a stored event through Run.Emit. A failure to publish an
// event that was stored (events.ErrPublish) is ignored: the event is
// durable and clients see it on replay.
func (r *Run) emit(ctx context.Context, typ string, payload any) error {
	if r.Emit == nil {
		return nil
	}
	err := r.Emit(ctx, typ, payload)
	if err == nil || errors.Is(err, events.ErrPublish) {
		return nil
	}
	return fmt.Errorf("%w: %s: %w", errEmitFailed, typ, err)
}

// emitEphemeral sends an ephemeral event; failures are ignored.
func (r *Run) emitEphemeral(ctx context.Context, typ string, payload any) {
	if r.Emit != nil {
		_ = r.Emit(ctx, typ, payload)
	}
}
