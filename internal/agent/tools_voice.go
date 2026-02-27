package agent

import (
	"fmt"
	"os/exec"
)

// SpeakTextTool allows Keith to synthesize audio out of the Mac's speakers
type SpeakTextTool struct{}

func (t *SpeakTextTool) Name() string { return "speak_text" }
func (t *SpeakTextTool) Description() string {
	return "Synthesizes human speech using macOS native TTS. Use this when the user explicitly asks you to speak, or to alert them. Arguments: 'text' (the string of text to speak), 'voice' (optional: the macOS voice persona to use. Examples: 'Daniel' (UK Male, default), 'Alex' (US Male), 'Rishi' (Indian Male), 'Samantha' (US Female). Use the voice the user specifies)."
}
func (t *SpeakTextTool) Execute(args map[string]string) (string, error) {
	text := args["text"]
	voice := args["voice"]

	if text == "" {
		return "", fmt.Errorf("missing 'text' argument")
	}

	if voice == "" {
		voice = "Daniel" // Default Jarvis-style Male Persona
	}

	err := exec.Command("say", "-v", voice, text).Run()
	if err != nil {
		return "", fmt.Errorf("failed to mechanically synthesize voice '%s': %v", voice, err)
	}

	return fmt.Sprintf("Audio engine physically vocalized payload using '%s' persona: '%s'", voice, text), nil
}
