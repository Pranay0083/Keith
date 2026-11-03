package agent

// RepeatState is repeat detection's counters, kept in Aux.Repeat so they
// survive a pause.
type RepeatState struct {
	// Failures maps a call's key (tool name + canonical input) to its
	// run of identical failures.
	Failures map[string]FailureRun `json:"failures,omitempty"`
	// TestFailures counts consecutive failed runs of a repo's test
	// command, whatever their output; a passing run resets it.
	TestFailures int `json:"test_failures,omitempty"`
	// Seq orders Failures entries for eviction.
	Seq int64 `json:"seq,omitempty"`
}

// FailureRun is a run of identical failures of one call.
type FailureRun struct {
	// ErrHash identifies the error (normalized output) the call failed
	// with.
	ErrHash string `json:"err_hash"`
	Count   int    `json:"count"`
	Seq     int64  `json:"seq"`
}
