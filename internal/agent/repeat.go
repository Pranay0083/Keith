package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/Pranay0083/keith/internal/llm"
)

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

// RepeatPolicy configures repeat detection, which catches an agent stuck
// retrying the same failing thing:
//
//   - The same call (same tool, same input) failing with the same error
//     (output compared with digits and whitespace normalized) counts up;
//     a success or a different error starts over. Other calls in
//     between do not reset it, so "edit, retry, same error" counts.
//   - A failed run of a repo's test command (Profile.TestCmd, matched as
//     a substring of a shell_exec command) counts up whatever its
//     output; a passing run resets it.
//
// When a count reaches NudgeAfter, a note telling the model to change
// approach or use ask_user is appended to the tool result. When it reaches
// PauseAfter, the session pauses with an automatic question.asked
// summarizing the attempts, and the counts start over.
type RepeatPolicy struct {
	// NudgeAfter is the count that adds a nudge (default 3; negative
	// disables).
	NudgeAfter int
	// PauseAfter is the count that pauses the session (default 5;
	// negative disables).
	PauseAfter int
}

func (p RepeatPolicy) withDefaults() RepeatPolicy {
	if p.NudgeAfter == 0 {
		p.NudgeAfter = 3
	}
	if p.PauseAfter == 0 {
		p.PauseAfter = 5
	}
	return p
}

// maxFailureRuns caps RepeatState.Failures.
const maxFailureRuns = 64

// verdict is repeat detection's reaction to one tool call.
type verdict struct {
	// note is appended to the tool result.
	note string
	// pause asks the loop to pause with question.
	pause    bool
	question string
}

// observe records one tool call's outcome in rs and decides whether to
// nudge or pause.
func (p RepeatPolicy) observe(rs *RepeatState, tool string, input json.RawMessage, res Result, repos []RepoContext) verdict {
	key := tool + "\x00" + canonicalJSON(input)
	output := resultText(res)

	same := 0
	if res.IsError {
		if rs.Failures == nil {
			rs.Failures = map[string]FailureRun{}
		}
		h := errHash(output)
		fr, ok := rs.Failures[key]
		if ok && fr.ErrHash == h {
			fr.Count++
		} else {
			fr = FailureRun{ErrHash: h, Count: 1}
		}
		rs.Seq++
		fr.Seq = rs.Seq
		rs.Failures[key] = fr
		same = fr.Count
		evictOldest(rs.Failures)
	} else {
		delete(rs.Failures, key)
	}

	tests, testCmd := 0, ""
	if tool == toolShellExec {
		cmd := commandOf(input)
		if isTestCommand(cmd, repos) {
			testCmd = cmd
			switch {
			case res.TimedOut:
			case res.IsError || res.ExitCode != 0:
				rs.TestFailures++
				tests = rs.TestFailures
			default:
				rs.TestFailures = 0
			}
		}
	}

	n, isTest := same, false
	if tests > same {
		n, isTest = tests, true
	}
	switch {
	case n == 0:
		return verdict{}
	case p.PauseAfter > 0 && n >= p.PauseAfter:
		delete(rs.Failures, key)
		rs.TestFailures = 0
		var q string
		if isTest {
			q = fmt.Sprintf("I seem to be stuck: the test command `%s` has failed %d runs in a row. The latest output ends with:\n\n%s\n\nHow would you like me to proceed?",
				oneLine(testCmd, 200), n, tailString(output, 800))
		} else {
			q = fmt.Sprintf("I seem to be stuck: %s has failed %d times in a row with the same error:\n\n%s\n\nHow would you like me to proceed?",
				describeCall(tool, input), n, tailString(output, 800))
		}
		return verdict{
			pause:    true,
			question: q,
			note:     fmt.Sprintf("[keith] This has now failed %d times in a row, so the session is paused and the user has been asked for help. Their reply will follow.", n),
		}
	case p.NudgeAfter > 0 && n >= p.NudgeAfter:
		if isTest {
			return verdict{note: fmt.Sprintf("[keith] The test command has now failed %d runs in a row. Step back before running it again: re-read the failures and the code under test and try a different approach, or use ask_user if you are blocked (for example, if the failures look unrelated to your change).", n)}
		}
		return verdict{note: fmt.Sprintf("[keith] This exact call has now failed %d times in a row with the same error. Repeating it will not help. Change your approach (re-read the relevant code or output, try a different fix or command), or use ask_user if you are blocked, for example by a missing secret or an unclear requirement.", n)}
	}
	return verdict{}
}

func evictOldest(m map[string]FailureRun) {
	for len(m) > maxFailureRuns {
		oldest, seq := "", int64(-1)
		for k, v := range m {
			if seq < 0 || v.Seq < seq {
				oldest, seq = k, v.Seq
			}
		}
		delete(m, oldest)
	}
}

// canonicalJSON re-encodes v with sorted keys so equal inputs compare
// equal however the model spaced or ordered them.
func canonicalJSON(raw json.RawMessage) string {
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return string(raw)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return string(raw)
	}
	return string(b)
}

var (
	digitsRE = regexp.MustCompile(`[0-9]+`)
	spaceRE  = regexp.MustCompile(`\s+`)
)

// errHash identifies an error output, ignoring numbers (durations, PIDs,
// timestamps) and whitespace.
func errHash(s string) string {
	s = digitsRE.ReplaceAllString(s, "0")
	s = spaceRE.ReplaceAllString(strings.TrimSpace(s), " ")
	return contentHash([]byte(s))[:16]
}

func resultText(res Result) string {
	var parts []string
	for _, b := range res.Content {
		if b.Type == llm.BlockText {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func commandOf(input json.RawMessage) string {
	var in struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal(input, &in)
	return in.Command
}

// isTestCommand reports whether cmd runs one of the repos' test commands.
func isTestCommand(cmd string, repos []RepoContext) bool {
	c := spaceRE.ReplaceAllString(strings.TrimSpace(cmd), " ")
	if c == "" {
		return false
	}
	for _, r := range repos {
		t := spaceRE.ReplaceAllString(strings.TrimSpace(r.Profile.TestCmd), " ")
		if t != "" && strings.Contains(c, t) {
			return true
		}
	}
	return false
}

func describeCall(tool string, input json.RawMessage) string {
	if tool == toolShellExec {
		if cmd := commandOf(input); cmd != "" {
			return fmt.Sprintf("the command `%s`", oneLine(cmd, 200))
		}
	}
	return fmt.Sprintf("%s with input %s", tool, oneLine(canonicalJSON(input), 200))
}
