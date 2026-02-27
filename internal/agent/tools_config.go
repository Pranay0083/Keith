package agent

import (
	"fmt"
	"strings"

	"github.com/Pranay0083/keith/internal/config"
)

// UpdateConfigTool allows Keith to natively update his own operational configurations
type UpdateConfigTool struct{}

func (t *UpdateConfigTool) Name() string { return "update_config" }
func (t *UpdateConfigTool) Description() string {
	return "Saves or updates secure credential settings natively in the agent's core config memory (e.g. Telegram Bot tokens). Use this when the user wants to configure integrations like Telegram. Arguments: 'key' (the config string to update, strictly must be one of: 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_PASSWORD', 'SUPERUSER_NAME'), 'value' (the actual string data to save)."
}
func (t *UpdateConfigTool) Execute(args map[string]string) (string, error) {
	key := strings.ToUpper(args["key"])
	value := args["value"]

	if key == "" || value == "" {
		return "", fmt.Errorf("missing 'key' or 'value' argument")
	}

	accs, err := config.LoadAccounts()
	if err != nil {
		return "", fmt.Errorf("failed to load system credentials ledger: %v", err)
	}

	switch key {
	case "TELEGRAM_BOT_TOKEN":
		accs.TelegramToken = value
	case "TELEGRAM_PASSWORD":
		accs.TelegramPassword = value
	case "SUPERUSER_NAME":
		accs.SuperUser = value
	default:
		return "", fmt.Errorf("security violation: unknown or protected config key requested '%s'", key)
	}

	err = config.SaveAccounts(accs)
	if err != nil {
		return "", fmt.Errorf("failed to physically write changes to file: %v", err)
	}

	return fmt.Sprintf("System Registry Updated. Successfully bound parameter '%s'. Note: If this is a core adapter like Telegram, kindly remind the user to restart the CLI ('Ctrl+C' then 'keith start') for the new integration to spin up.", key), nil
}
