package memstore

import (
	"context"
)

// ClearSessionSandbox implements worker.Store: it clears the session's
// sandbox only if it is still sandboxID.
func (m *Store) ClearSessionSandbox(_ context.Context, id, sandboxID string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	x, ok := m.get(id)
	if !ok || sandboxID == "" || x.s.SandboxID != sandboxID {
		return false, nil
	}
	x.s.SandboxID, x.s.SandboxProvider = "", ""
	m.touch(x)
	return true, nil
}
