package agent

import (
	"fmt"
	"os/exec"
)

// TakeScreenshotTool invokes native macOS screen capture
type TakeScreenshotTool struct{}

func (t *TakeScreenshotTool) Name() string { return "take_screenshot" }
func (t *TakeScreenshotTool) Description() string {
	return "Takes a physical screenshot of the user's primary Mac display. The image will be parsed through your Vision Neural Network so you can physically 'see' the screen."
}
func (t *TakeScreenshotTool) Execute(args map[string]string) (string, error) {
	path := "/tmp/keith_vision_matrix.png"
	// -x prevents the camera shutter sound to maintain stealth
	err := exec.Command("screencapture", "-x", path).Run()
	if err != nil {
		return "", fmt.Errorf("macOS screencapture failed: %v", err)
	}
	return "[IMAGE_PAYLOAD]" + path, nil
}

// MoveMouseTool physically moves the user's cursor
type MoveMouseTool struct{}

func (t *MoveMouseTool) Name() string { return "move_mouse" }
func (t *MoveMouseTool) Description() string {
	return "Moves the physical Mac cursor to specific pixel coordinates on the screen. Arguments: 'x', 'y'. E.g. x: 500, y: 500."
}
func (t *MoveMouseTool) Execute(args map[string]string) (string, error) {
	x := args["x"]
	y := args["y"]
	if x == "" || y == "" {
		return "", fmt.Errorf("missing 'x' or 'y' arguments")
	}

	_, err := exec.LookPath("cliclick")
	if err != nil {
		return "CRITICAL MOTOR ERROR: `cliclick` binary not found! You are paralyzed. You MUST use `shell_exec` to run `brew install cliclick` before you can physically drive the cursor.", nil
	}

	cmdStr := fmt.Sprintf("m:%s,%s", x, y)
	err = exec.Command("cliclick", cmdStr).Run()
	if err != nil {
		return "", fmt.Errorf("cursor execution failed: %v", err)
	}
	return fmt.Sprintf("Cursor relocated to (%s, %s).", x, y), nil
}

// ClickMouseTool physically clicks the user's primary mouse button
type ClickMouseTool struct{}

func (t *ClickMouseTool) Name() string { return "click_mouse" }
func (t *ClickMouseTool) Description() string {
	return "Clicks the primary (left) mouse button at specific pixel coordinates on the screen. Arguments: 'x', 'y'."
}
func (t *ClickMouseTool) Execute(args map[string]string) (string, error) {
	x := args["x"]
	y := args["y"]
	if x == "" || y == "" {
		return "", fmt.Errorf("missing 'x' or 'y' arguments")
	}

	_, err := exec.LookPath("cliclick")
	if err != nil {
		return "CRITICAL MOTOR ERROR: `cliclick` binary not found. You MUST use `shell_exec` to run `brew install cliclick`.", nil
	}

	// c:x,y moves and clicks
	cmdStr := fmt.Sprintf("c:%s,%s", x, y)
	err = exec.Command("cliclick", cmdStr).Run()
	if err != nil {
		return "", fmt.Errorf("click execution failed: %v", err)
	}
	return fmt.Sprintf("Mouse clicked at (%s, %s).", x, y), nil
}

// TypeKeyboardTool physically injects keystrokes
type TypeKeyboardTool struct{}

func (t *TypeKeyboardTool) Name() string { return "type_keyboard" }
func (t *TypeKeyboardTool) Description() string {
	return "Physically types out text using the user's keyboard. Arguments: 'text' (e.g. 'Hello World'). Important: You must click into an input field using 'click_mouse' BEFORE you can type!"
}
func (t *TypeKeyboardTool) Execute(args map[string]string) (string, error) {
	text := args["text"]
	if text == "" {
		return "", fmt.Errorf("missing 'text' argument")
	}

	_, err := exec.LookPath("cliclick")
	if err != nil {
		return "CRITICAL MOTOR ERROR: `cliclick` binary not found. You MUST use `shell_exec` to run `brew install cliclick`.", nil
	}

	// t:text types the literal string
	cmdStr := fmt.Sprintf("t:%s", text)
	err = exec.Command("cliclick", cmdStr).Run()
	if err != nil {
		return "", fmt.Errorf("keystroke execution failed: %v", err)
	}
	return fmt.Sprintf("Physically typed payload: %s", text), nil
}

// PressKeyTool physically presses a special keyboard action (like Enter)
type PressKeyTool struct{}

func (t *PressKeyTool) Name() string { return "press_key" }
func (t *PressKeyTool) Description() string {
	return "Physically presses a special OS key. Arguments: 'key' (e.g. 'return', 'tab', 'esc', 'space', 'delete')."
}
func (t *PressKeyTool) Execute(args map[string]string) (string, error) {
	key := args["key"]
	if key == "" {
		return "", fmt.Errorf("missing 'key' argument")
	}

	_, err := exec.LookPath("cliclick")
	if err != nil {
		return "CRITICAL MOTOR ERROR: `cliclick` binary not found. You MUST use `shell_exec` to run `brew install cliclick`.", nil
	}

	// kp:key presses the special key
	cmdStr := fmt.Sprintf("kp:%s", key)
	err = exec.Command("cliclick", cmdStr).Run()
	if err != nil {
		return "", fmt.Errorf("special key execution failed: %v", err)
	}
	return fmt.Sprintf("Physically pressed OS Key: %s", key), nil
}

// TypeShortcutTool holds modifiers while pressing a key
type TypeShortcutTool struct{}

func (t *TypeShortcutTool) Name() string { return "type_shortcut" }
func (t *TypeShortcutTool) Description() string {
	return "Physically holds down one or more modifier keys, presses a main key, and then releases them. Arguments: 'modifiers' (comma-separated, e.g. 'ctrl,shift,cmd,alt'), 'key' (the main key to press, e.g. 'arrow-left', 'c', 'v', 'return')."
}
func (t *TypeShortcutTool) Execute(args map[string]string) (string, error) {
	modifiers := args["modifiers"]
	key := args["key"]

	if key == "" {
		return "", fmt.Errorf("missing 'key' argument")
	}

	_, err := exec.LookPath("cliclick")
	if err != nil {
		return "CRITICAL MOTOR ERROR: `cliclick` binary not found. You MUST use `shell_exec` to run `brew install cliclick`.", nil
	}

	// kd:ctrl,cmd kp:c ku:ctrl,cmd
	cmdArgs := []string{}
	if modifiers != "" {
		cmdArgs = append(cmdArgs, fmt.Sprintf("kd:%s", modifiers))
	}
	cmdArgs = append(cmdArgs, fmt.Sprintf("kp:%s", key))
	if modifiers != "" {
		cmdArgs = append(cmdArgs, fmt.Sprintf("ku:%s", modifiers))
	}

	err = exec.Command("cliclick", cmdArgs...).Run()
	if err != nil {
		return "", fmt.Errorf("shortcut execution failed: %v", err)
	}
	return fmt.Sprintf("Physically executed shortcut: %s + %s", modifiers, key), nil
}
