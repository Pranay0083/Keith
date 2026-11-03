package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/Pranay0083/keith/internal/events"
	"github.com/Pranay0083/keith/internal/llm"
)

// maxChoices caps ask_user's choices.
const maxChoices = 10

// messageUserTool posts a progress note to the chat without stopping.
type messageUserTool struct{ newID func(prefix string) string }

func (messageUserTool) Spec() llm.ToolSpec {
	return spec(toolMessageUser,
		"Post a short progress note to the user's chat and keep working. Use it sparingly, for milestones or findings the user would want to see before you finish (for example, what you found to be the root cause). Do not use it to ask questions; use ask_user for that.",
		map[string]any{"text": strProp("The note, in Markdown. Keep it brief.")},
		"text")
}

func (t messageUserTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Text string `json:"text"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	text := strings.TrimSpace(in.Text)
	if text == "" {
		return errorResult("text is empty."), nil
	}
	if err := run.emit(ctx, events.TypeMessageAssistant, events.MessageAssistantPayload{
		MessageID: t.newID("msg"), Text: text,
	}); err != nil {
		return Result{}, err
	}
	return textResult("note: "+oneLine(text, 80), "Posted to the user's chat."), nil
}

// askUserTool asks the user a question and pauses the session until they
// answer. The answer becomes this call's tool_result.
type askUserTool struct{}

func (askUserTool) pausesSession() {}

func (askUserTool) Spec() llm.ToolSpec {
	return spec(toolAskUser,
		"Ask the user a question and pause until they answer. Their answer comes back as this tool's result. Use it when you are blocked: a missing secret or credential, a requirement that is unclear or contradictory, or a decision only the user can make. Ask one clear, self-contained question, include what you already tried or found, and offer choices when the answer is one of a few options. Other tool calls in the same response still run; ask at most one question at a time.",
		map[string]any{
			"question": strProp("The question, in Markdown, with the context the user needs to answer it."),
			"choices":  strArrayProp("Optional short answers the user can pick from (at most 10). The user may still answer in their own words."),
		},
		"question")
}

func (askUserTool) Execute(ctx context.Context, run *Run, input json.RawMessage) (Result, error) {
	var in struct {
		Question string   `json:"question"`
		Choices  []string `json:"choices"`
	}
	if err := decodeInput(input, &in); err != nil {
		return badInput(err), nil
	}
	q := strings.TrimSpace(in.Question)
	if q == "" {
		return badInput(errors.New("question is empty")), nil
	}
	var choices []string
	for _, c := range in.Choices {
		if c = strings.TrimSpace(c); c != "" {
			choices = append(choices, c)
		}
	}
	if len(choices) > maxChoices {
		return badInput(errors.New("at most 10 choices")), nil
	}
	id := CallID(ctx)
	return Result{
		Content:  []llm.Block{llm.TextBlock("(The session is paused until the user answers.)")},
		Summary:  "asked: " + oneLine(q, 100),
		Pause:    true,
		question: &Question{ID: id, ToolUseID: id, Text: q, Choices: choices},
	}, nil
}
