package agent

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// ControlVolumeTool manages macOS system audio levels
type ControlVolumeTool struct{}

func (t *ControlVolumeTool) Name() string { return "control_volume" }
func (t *ControlVolumeTool) Description() string {
	return "Adjusts the master system volume of the Mac. Arguments: 'level' (integer string from 0 to 100, where 0 is mute and 100 is max)."
}
func (t *ControlVolumeTool) Execute(args map[string]string) (string, error) {
	levelStr := args["level"]
	if levelStr == "" {
		return "", fmt.Errorf("missing 'level' argument")
	}

	level, err := strconv.Atoi(levelStr)
	if err != nil || level < 0 || level > 100 {
		return "", fmt.Errorf("invalid volume level. Must be an integer between 0 and 100")
	}

	script := fmt.Sprintf("set volume output volume %d", level)
	err = exec.Command("osascript", "-e", script).Run()
	if err != nil {
		return "", fmt.Errorf("failed to adjust system volume: %v", err)
	}

	return fmt.Sprintf("Speaker volume fundamentally locked to %d%%.", level), nil
}

// ControlWiFiTool toggles the Wi-Fi Adapter
type ControlWiFiTool struct{}

func (t *ControlWiFiTool) Name() string { return "control_wifi" }
func (t *ControlWiFiTool) Description() string {
	return "Toggles the macOS Wi-Fi adapter. Arguments: 'state' (string: 'on' or 'off')."
}
func (t *ControlWiFiTool) Execute(args map[string]string) (string, error) {
	state := strings.ToLower(args["state"])
	if state != "on" && state != "off" {
		return "", fmt.Errorf("invalid state. Must be 'on' or 'off'")
	}

	// Assuming en0 is standard for Wi-Fi on modern Macs, but parsing automatically is safer
	out, err := exec.Command("sh", "-c", "networksetup -listallhardwareports | awk '/Wi-Fi|AirPort/ {getline; print $NF}'").Output()
	port := strings.TrimSpace(string(out))
	if err != nil || port == "" {
		port = "en0" // Fallback standard
	}

	err = exec.Command("networksetup", "-setairportpower", port, state).Run()
	if err != nil {
		return "", fmt.Errorf("failed to toggle Wi-Fi adapter on %s: %v", port, err)
	}

	return fmt.Sprintf("Network connection physically switched %s on port %s.", strings.ToUpper(state), port), nil
}

// ControlBluetoothTool toggles the Bluetooth Radio
type ControlBluetoothTool struct{}

func (t *ControlBluetoothTool) Name() string { return "control_bluetooth" }
func (t *ControlBluetoothTool) Description() string {
	return "Toggles the macOS Bluetooth radio. Arguments: 'state' (string: 'on' or 'off')."
}
func (t *ControlBluetoothTool) Execute(args map[string]string) (string, error) {
	state := strings.ToLower(args["state"])
	powerInt := "1"
	if state == "off" {
		powerInt = "0"
	} else if state != "on" {
		return "", fmt.Errorf("invalid state. Must be 'on' or 'off'")
	}

	// macOS has locked down native Bluetooth CLI, so we use `blueutil` which we can auto-install via brew
	_, err := exec.LookPath("blueutil")
	if err != nil {
		fmt.Println("Bluetooth CLI missing... Intercepting Brew package manager to install `blueutil` instantly.")
		exec.Command("brew", "install", "blueutil").Run()
	}

	err = exec.Command("blueutil", "-p", powerInt).Run()
	if err != nil {
		return "", fmt.Errorf("failed to command bluetooth daemon: %v", err)
	}

	return fmt.Sprintf("Bluetooth radio broadcast physically switched %s.", strings.ToUpper(state)), nil
}

// ControlDisplayBrightnessTool adjusts the physical monitor backlight via mechanical Key Codes
type ControlDisplayBrightnessTool struct{}

func (t *ControlDisplayBrightnessTool) Name() string { return "control_brightness" }
func (t *ControlDisplayBrightnessTool) Description() string {
	return "Adjusts the physical screen brightness. Arguments: 'level' (integer string from 0 to 100)."
}
func (t *ControlDisplayBrightnessTool) Execute(args map[string]string) (string, error) {
	levelStr := args["level"]
	if levelStr == "" {
		return "", fmt.Errorf("missing 'level' argument")
	}

	level, err := strconv.Atoi(levelStr)
	if err != nil || level < 0 || level > 100 {
		return "", fmt.Errorf("invalid brightness level. Must be an integer between 0 and 100")
	}

	// Apple Silicon locks CLI brightness. We use a zeroing mechanical macro (0 to 16 clicks).
	clicks := int(float64(level) / 100.0 * 16.0)

	script := fmt.Sprintf(`
		tell application "System Events"
			repeat 16 times
				key code 145
			end repeat
			delay 0.1
			repeat %d times
				key code 144
			end repeat
		end tell
	`, clicks)

	err = exec.Command("osascript", "-e", script).Run()
	if err != nil {
		return "", fmt.Errorf("failed to command screen backlight: %v", err)
	}

	return fmt.Sprintf("Screen backlight mechanically calibrated to %d%% (%d clicks).", level, clicks), nil
}
