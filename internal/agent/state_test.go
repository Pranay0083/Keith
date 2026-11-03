package agent

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/Pranay0083/keith/internal/llm"
)

func sampleState() *State {
	st := NewState("system prompt")
	st.Messages = append(st.Messages,
		llm.UserText("fix the bug"),
		llm.Message{Role: llm.RoleAssistant, Content: []llm.Block{
			{Type: llm.BlockThinking, Text: "hmm", Signature: "sig", ProviderRaw: json.RawMessage(`{"x":1}`)},
			llm.TextBlock("Asking."),
			llm.ToolUseBlock("call_1", "ask_user", json.RawMessage(`{"question":"Which DB?"}`)),
		}},
	)
	st.Aux.Raw = []RawResponse{{Step: 1, Raw: json.RawMessage(`{"id":"msg_1"}`)}}
	st.Aux.Edits.push("/workspace/r/a.go", EditRecord{Tool: "edit_file", Existed: true, Prev: []byte{0xff, 0x00, 'a'}, Mode: 0o755, After: "abc"})
	st.Aux.Repeat = RepeatState{Failures: map[string]FailureRun{"k": {ErrHash: "e", Count: 2, Seq: 1}}, TestFailures: 1, Seq: 1}
	st.Aux.Pending = &PendingTurn{
		Results:  []llm.Block{llm.ToolResultBlock("call_1", false)},
		Texts:    []llm.Block{llm.TextBlock("later")},
		Question: &Question{ID: "call_1", ToolUseID: "call_1", Text: "Which DB?", Choices: []string{"pg", "mysql"}},
	}
	st.Aux.Usage = llm.Usage{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 3, CacheWriteTokens: 2}
	st.Aux.Steps = 1
	st.Aux.Delivered = []string{"m1"}
	return st
}

func TestStateColumnsRoundTrip(t *testing.T) {
	st := sampleState()
	msgs, aux, err := st.MarshalColumns()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(msgs, []byte("[")) {
		t.Fatalf("llm_messages is not a JSON array: %s", msgs)
	}
	got, err := UnmarshalState(msgs, aux)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, st) {
		t.Fatalf("round trip mismatch\n got %+v\nwant %+v", got, st)
	}
	if q := got.OpenQuestion(); q == nil || q.ID != "call_1" {
		t.Fatalf("OpenQuestion = %+v", q)
	}
}

func TestStateJSONRoundTrip(t *testing.T) {
	st := sampleState()
	b, err := json.Marshal(st)
	if err != nil {
		t.Fatal(err)
	}
	var got State
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&got, st) {
		t.Fatalf("round trip mismatch\n got %+v\nwant %+v", &got, st)
	}
}

func TestUnmarshalStateDefaults(t *testing.T) {
	for _, in := range []string{"", "null", "[]", " [] "} {
		st, err := UnmarshalState(json.RawMessage(in), json.RawMessage(in))
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if st.System != "" || len(st.Messages) != 0 || st.Messages == nil || st.OpenQuestion() != nil {
			t.Fatalf("%q: got %+v", in, st)
		}
	}
	if _, err := UnmarshalState(json.RawMessage(`{`), nil); err == nil {
		t.Fatal("bad messages: want error")
	}
	if _, err := UnmarshalState(nil, json.RawMessage(`{"aux":1}`)); err == nil {
		t.Fatal("bad aux: want error")
	}
}

func TestEmptyStateMarshalsEmptyArray(t *testing.T) {
	msgs, aux, err := (&State{}).MarshalColumns()
	if err != nil {
		t.Fatal(err)
	}
	if string(msgs) != "[]" {
		t.Fatalf("messages = %s", msgs)
	}
	if !json.Valid(aux) {
		t.Fatalf("aux = %s", aux)
	}
}

func TestEditHistoryLimits(t *testing.T) {
	var h EditHistory
	for i := 0; i < maxEditDepth+5; i++ {
		h.push("/workspace/a", EditRecord{Tool: "write_file", Existed: true, Prev: []byte{byte(i)}})
	}
	if n := len(h.Files["/workspace/a"]); n != maxEditDepth {
		t.Fatalf("depth = %d", n)
	}
	r, ok := h.last("/workspace/a")
	if !ok || r.Prev[0] != byte(maxEditDepth+4) {
		t.Fatalf("last = %+v", r)
	}

	h.push("/workspace/big", EditRecord{Existed: true, Prev: make([]byte, maxEditPrevBytes+1)})
	if r, _ := h.last("/workspace/big"); !r.Dropped || r.Prev != nil {
		t.Fatalf("big record kept content: dropped=%v len=%d", r.Dropped, len(r.Prev))
	}

	// Fill past the total cap: the oldest file's records go first.
	var g EditHistory
	g.push("/workspace/old", EditRecord{Existed: true, Prev: make([]byte, maxEditPrevBytes)})
	for i := 0; i < maxEditBytes/maxEditPrevBytes; i++ {
		g.push("/workspace/new", EditRecord{Existed: true, Prev: make([]byte, maxEditPrevBytes)})
	}
	if _, ok := g.last("/workspace/old"); ok {
		t.Fatal("oldest record survived the total cap")
	}
	g.pop("/workspace/new")
	if n := len(g.Files["/workspace/new"]); n != maxEditBytes/maxEditPrevBytes-1 {
		t.Fatalf("after pop: %d", n)
	}
}

func TestDelivered(t *testing.T) {
	var a Aux
	for i := 0; i < maxDelivered+10; i++ {
		a.markDelivered(string(rune('a' + i%26)) + string(rune(i)))
	}
	if len(a.Delivered) != maxDelivered {
		t.Fatalf("len = %d", len(a.Delivered))
	}
	a.markDelivered("x")
	if !a.delivered("x") || a.delivered("nope") {
		t.Fatal("delivered lookup")
	}
}
