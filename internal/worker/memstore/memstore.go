// Package memstore is an in-memory implementation of worker.Store, for
// the worker's unit tests and `keith run --local` (which runs one session
// without Postgres). It keeps the store package's semantics where the
// worker depends on them: statuses set finished_at, inputs are pending
// until consumed (idempotently), agent state is pinned to the session's
// provider, diffs are capped, repo secrets override org secrets.
package memstore

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/store"
)

// Store is an in-memory worker.Store. It is safe for concurrent use.
type Store struct {
	// Now is the clock for timestamps; nil means time.Now.
	Now func() time.Time

	mu       sync.Mutex
	sessions map[string]*session
	profiles map[string]store.RepoProfile
	secrets  []store.SecretCiphertext
	inputSeq int
}

type session struct {
	s         store.Session
	cancel    *time.Time
	inputs    []store.Input
	state     *store.AgentState
	diffs     map[string]store.SessionDiff
	statusLog []events.SessionStatus
}

// New returns an empty Store.
func New() *Store {
	return &Store{sessions: map[string]*session{}, profiles: map[string]store.RepoProfile{}}
}

func (m *Store) now() time.Time {
	if m.Now != nil {
		return m.Now().UTC()
	}
	return time.Now().UTC()
}

func notFound(op string) error { return fmt.Errorf("memstore: %s: %w", op, store.ErrNotFound) }

// AddSession stores a session with its repos. Empty Status means queued;
// empty repo branches get store.SessionBranch.
func (m *Store) AddSession(s store.Session) store.Session {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s.Status == "" {
		s.Status = events.StatusQueued
	}
	now := m.now()
	if s.CreatedAt.IsZero() {
		s.CreatedAt = now
	}
	s.UpdatedAt = now
	s.Repos = slices.Clone(s.Repos)
	for i := range s.Repos {
		s.Repos[i].SessionID = s.ID
		if s.Repos[i].Branch == "" {
			s.Repos[i].Branch = store.SessionBranch(s.ID)
		}
	}
	m.sessions[s.ID] = &session{s: s, diffs: map[string]store.SessionDiff{}}
	return s
}

func (m *Store) get(id string) (*session, bool) {
	x, ok := m.sessions[id]
	return x, ok
}

func (m *Store) touch(x *session) { x.s.UpdatedAt = m.now() }

// GetSession implements worker.Store.
func (m *Store) GetSession(_ context.Context, id string) (store.Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return store.Session{}, notFound("get session")
	}
	out := x.s
	out.Repos = slices.Clone(x.s.Repos)
	return out, nil
}

// Session returns a copy of a session (nil-safe for tests: zero value
// when unknown).
func (m *Store) Session(id string) store.Session {
	s, _ := m.GetSession(context.Background(), id)
	return s
}

// SetSessionStatus implements worker.Store.
func (m *Store) SetSessionStatus(_ context.Context, id string, status events.SessionStatus, reason string) error {
	if !slices.Contains(events.Statuses, status) {
		return fmt.Errorf("memstore: invalid status %q", status)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return notFound("set session status")
	}
	x.s.Status, x.s.StatusReason = status, reason
	if status.Terminal() {
		t := m.now()
		x.s.FinishedAt = &t
	} else {
		x.s.FinishedAt = nil
	}
	x.statusLog = append(x.statusLog, status)
	m.touch(x)
	return nil
}

// Statuses returns every status a session was set to, in order.
func (m *Store) Statuses(id string) []events.SessionStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	if x, ok := m.get(id); ok {
		return slices.Clone(x.statusLog)
	}
	return nil
}

// SetSessionSandbox implements worker.Store.
func (m *Store) SetSessionSandbox(_ context.Context, id, sandboxID, provider string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return notFound("set session sandbox")
	}
	x.s.SandboxID, x.s.SandboxProvider = sandboxID, provider
	m.touch(x)
	return nil
}

// RequestCancel sets the cancel flag (what Stop does).
func (m *Store) RequestCancel(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return notFound("request cancel")
	}
	if x.cancel == nil {
		t := m.now()
		x.cancel = &t
		x.s.CancelRequestedAt = &t
	}
	m.touch(x)
	return nil
}

// IsCancelRequested implements worker.Store.
func (m *Store) IsCancelRequested(_ context.Context, id string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return false, notFound("is cancel requested")
	}
	return x.cancel != nil, nil
}

// ClearCancelRequest implements worker.Store.
func (m *Store) ClearCancelRequest(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok {
		return notFound("clear cancel request")
	}
	x.cancel, x.s.CancelRequestedAt = nil, nil
	m.touch(x)
	return nil
}

func (x *session) repo(repoID string) *store.SessionRepo {
	for i := range x.s.Repos {
		if x.s.Repos[i].RepoID == repoID {
			return &x.s.Repos[i]
		}
	}
	return nil
}

// SetSessionRepoBranch implements worker.Store.
func (m *Store) SetSessionRepoBranch(_ context.Context, sessionID, repoID, branch string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok || x.repo(repoID) == nil {
		return notFound("set session repo branch")
	}
	x.repo(repoID).Branch = branch
	return nil
}

// SetSessionRepoPR implements worker.Store.
func (m *Store) SetSessionRepoPR(_ context.Context, sessionID, repoID string, number int, url string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok || x.repo(repoID) == nil {
		return notFound("set session repo pr")
	}
	r := x.repo(repoID)
	r.PRNumber, r.PRURL = number, url
	return nil
}

// SetSessionRepoDiff implements worker.Store, capping the diff like the
// store does.
func (m *Store) SetSessionRepoDiff(_ context.Context, sessionID, repoID, diff string, truncated bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok || x.repo(repoID) == nil {
		return notFound("set session repo diff")
	}
	diff = strings.ToValidUTF8(diff, "�")
	if len(diff) > store.MaxDiffBytes {
		n := store.MaxDiffBytes
		for n > 0 && !utf8.RuneStart(diff[n]) {
			n--
		}
		diff, truncated = diff[:n], true
	}
	t := m.now()
	x.diffs[repoID] = store.SessionDiff{RepoID: repoID, FullName: x.repo(repoID).FullName, Diff: diff, Truncated: truncated, UpdatedAt: &t}
	return nil
}

// Diffs returns a session's saved diffs in repo order (one per repo).
func (m *Store) Diffs(sessionID string) []store.SessionDiff {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return nil
	}
	out := make([]store.SessionDiff, 0, len(x.s.Repos))
	for _, r := range x.s.Repos {
		d, ok := x.diffs[r.RepoID]
		if !ok {
			d = store.SessionDiff{RepoID: r.RepoID, FullName: r.FullName}
		}
		out = append(out, d)
	}
	return out
}

// AddInput stores a pending input, as POST /messages or /answers does,
// and returns it. An answer clears the open question, as the API does.
func (m *Store) AddInput(sessionID string, kind store.InputKind, text, questionID, userID string) (store.Input, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return store.Input{}, notFound("create input")
	}
	m.inputSeq++
	in := store.Input{
		ID: fmt.Sprintf("m_%010d", m.inputSeq), SessionID: sessionID, Kind: kind, QuestionID: questionID,
		UserID: userID, Text: text, CreatedAt: m.now(),
	}
	x.inputs = append(x.inputs, in)
	if kind == store.InputAnswer {
		x.s.OpenQuestionID = ""
	}
	return in, nil
}

// ListPendingInputs implements worker.Store.
func (m *Store) ListPendingInputs(_ context.Context, sessionID string) ([]store.Input, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return nil, nil
	}
	var out []store.Input
	for _, in := range x.inputs {
		if in.ConsumedAt == nil {
			out = append(out, in)
		}
	}
	return out, nil
}

// Inputs returns every input of a session, consumed or not.
func (m *Store) Inputs(sessionID string) []store.Input {
	m.mu.Lock()
	defer m.mu.Unlock()
	if x, ok := m.get(sessionID); ok {
		return slices.Clone(x.inputs)
	}
	return nil
}

// MarkInputsConsumed implements worker.Store.
func (m *Store) MarkInputsConsumed(_ context.Context, sessionID string, ids []string) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return 0, nil
	}
	var n int64
	t := m.now()
	for i := range x.inputs {
		if x.inputs[i].ConsumedAt == nil && slices.Contains(ids, x.inputs[i].ID) {
			x.inputs[i].ConsumedAt = &t
			n++
		}
	}
	return n, nil
}

// SetOpenQuestion implements worker.Store.
func (m *Store) SetOpenQuestion(_ context.Context, sessionID, questionID string) error {
	if questionID == "" {
		return fmt.Errorf("memstore: empty question ID")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return notFound("set open question")
	}
	x.s.OpenQuestionID = questionID
	m.touch(x)
	return nil
}

// ClearOpenQuestion implements worker.Store.
func (m *Store) ClearOpenQuestion(_ context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok {
		return notFound("clear open question")
	}
	x.s.OpenQuestionID = ""
	m.touch(x)
	return nil
}

// SaveAgentState implements worker.Store.
func (m *Store) SaveAgentState(_ context.Context, st store.AgentState) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(st.SessionID)
	if !ok {
		return notFound("save agent state")
	}
	if st.Provider != x.s.Provider || (x.state != nil && x.state.Provider != st.Provider) {
		return fmt.Errorf("memstore: save agent state: %w", store.ErrProviderMismatch)
	}
	cp := st
	cp.LLMMessages = append(json.RawMessage(nil), st.LLMMessages...)
	cp.RawMessages = append(json.RawMessage(nil), st.RawMessages...)
	cp.UpdatedAt = m.now()
	x.state = &cp
	return nil
}

// LoadAgentState implements worker.Store.
func (m *Store) LoadAgentState(_ context.Context, sessionID string) (store.AgentState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(sessionID)
	if !ok || x.state == nil {
		return store.AgentState{}, notFound("load agent state")
	}
	return *x.state, nil
}

// SetProfile stores a repo profile.
func (m *Store) SetProfile(p store.RepoProfile) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.profiles[p.RepoID] = p
}

// GetRepoProfile implements worker.Store.
func (m *Store) GetRepoProfile(_ context.Context, repoID string) (store.RepoProfile, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.profiles[repoID]
	if !ok {
		return store.RepoProfile{}, notFound("get repo profile")
	}
	return p, nil
}

// SetRepoSnapshot implements worker.Store.
func (m *Store) SetRepoSnapshot(_ context.Context, repoID string, snap store.Snapshot) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	p := m.profiles[repoID]
	p.RepoID = repoID
	p.Snapshot = snap
	p.UpdatedAt = m.now()
	m.profiles[repoID] = p
	return nil
}

// AddSecret stores a sealed secret.
func (m *Store) AddSecret(c store.SecretCiphertext) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.secrets = append(m.secrets, c)
}

// LoadSecretCiphertexts implements worker.Store: org secrets first, then
// repo secrets, each by name.
func (m *Store) LoadSecretCiphertexts(_ context.Context, orgID string, repoIDs ...string) ([]store.SecretCiphertext, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []store.SecretCiphertext
	for _, c := range m.secrets {
		if c.OrgID == orgID && (c.RepoID == "" || slices.Contains(repoIDs, c.RepoID)) {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if (out[i].RepoID == "") != (out[j].RepoID == "") {
			return out[i].RepoID == ""
		}
		if out[i].RepoID != out[j].RepoID {
			return out[i].RepoID < out[j].RepoID
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

// ListIdleSandboxSessions implements worker.Store.
func (m *Store) ListIdleSandboxSessions(_ context.Context, idleSince time.Time, limit int) ([]store.Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if limit <= 0 {
		limit = 100
	}
	var out []store.Session
	for _, x := range m.sessions {
		parked := x.s.Status.Terminal() || x.s.Status == events.StatusWaitingForUser
		if x.s.SandboxID != "" && parked && x.s.UpdatedAt.Before(idleSince) {
			s := x.s
			s.Repos = nil
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].UpdatedAt.Before(out[j].UpdatedAt) })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// SandboxIDsInUse implements worker.Store.
func (m *Store) SandboxIDsInUse(_ context.Context, ids []string) (map[string]bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := map[string]bool{}
	for _, x := range m.sessions {
		if x.s.SandboxID != "" && slices.Contains(ids, x.s.SandboxID) {
			out[x.s.SandboxID] = true
		}
	}
	return out, nil
}

// Backdate moves a session's updated_at back by d, for reaper tests.
func (m *Store) Backdate(id string, d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if x, ok := m.get(id); ok {
		x.s.UpdatedAt = x.s.UpdatedAt.Add(-d)
	}
}
