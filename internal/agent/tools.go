package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/memory"
)

type Tool interface {
	Name() string
	Description() string
	Execute(args map[string]string) (string, error)
}

// ShellTool executes a bash command.
type ShellTool struct{}

func (s *ShellTool) Name() string {
	return "shell_exec"
}

func (s *ShellTool) Description() string {
	return "Executes a shell command. Arguments: 'command'."
}

func (s *ShellTool) Execute(args map[string]string) (string, error) {
	cmdStr, ok := args["command"]
	if !ok {
		return "", fmt.Errorf("missing 'command' argument")
	}

	isBgStr, hasBg := args["is_background"]
	isBackground := hasBg && isBgStr == "true"

	cmd := exec.Command("sh", "-c", cmdStr)

	// Default to user home directory
	if homeDir, err := os.UserHomeDir(); err == nil {
		cmd.Dir = homeDir
	}

	if isBackground {
		// Spawn in background and immediately return
		if err := cmd.Start(); err != nil {
			return fmt.Sprintf("Failed to start background process: %v", err), err
		}
		// Make a best-effort attempt to let the OS detach the process
		go func() { cmd.Wait() }()
		return fmt.Sprintf("Process detached. Successfully spawned background process with PID %d.\nCommand: %s", cmd.Process.Pid, cmdStr), nil
	}

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	// Add a timeout to prevent hanging commands
	errc := make(chan error, 1)
	go func() {
		errc <- cmd.Run()
	}()

	select {
	case err := <-errc:
		if err != nil {
			return fmt.Sprintf("Error: %v\nOutput: %s\nStderr: %s", err, out.String(), stderr.String()), err
		}
	case <-time.After(30 * time.Second):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return "", fmt.Errorf("command timed out after 30 seconds")
	}

	result := out.String()
	if result == "" {
		result = stderr.String()
	}
	if result == "" {
		result = "(Command executed successfully with no output)"
	}
	return result, nil
}

// SaveFactTool permanently saves a fact to Keith's SQLite brain.
type SaveFactTool struct{}

func (s *SaveFactTool) Name() string {
	return "save_fact"
}

func (s *SaveFactTool) Description() string {
	return "Saves an important fact, user preference, or project insight to your permanent memory graph. Arguments: 'fact'."
}

func (s *SaveFactTool) Execute(args map[string]string) (string, error) {
	fact, ok := args["fact"]
	if !ok {
		return "", fmt.Errorf("missing 'fact' argument")
	}

	if err := memory.SaveFact(fact); err != nil {
		return fmt.Sprintf("Failed to save memory: %v", err), err
	}
	return "Fact permanently successfully saved to Supermemory.", nil
}

// SearchMemoryTool searches Keith's SQLite FTS5 graph.
type SearchMemoryTool struct{}

func (s *SearchMemoryTool) Name() string {
	return "search_memory"
}

func (s *SearchMemoryTool) Description() string {
	return "Searches your permanent memory graph for past records. Arguments: 'query'."
}

func (s *SearchMemoryTool) Execute(args map[string]string) (string, error) {
	query, ok := args["query"]
	if !ok {
		return "", fmt.Errorf("missing 'query' argument")
	}

	results, err := memory.SearchFacts(query)
	if err != nil {
		return fmt.Sprintf("Search failed: %v", err), err
	}

	if len(results) == 0 {
		return "No relevant memories found.", nil
	}
	return "Found Memories:\n- " + strings.Join(results, "\n- "), nil
}

// SearchCodebaseTool hits the local RAG engine for indexed repository chunks.
type SearchCodebaseTool struct{}

func (s *SearchCodebaseTool) Name() string {
	return "search_codebase"
}

func (s *SearchCodebaseTool) Description() string {
	return "Searches the locally ingested codebase for semantic context. Arguments: 'query'."
}

func (s *SearchCodebaseTool) Execute(args map[string]string) (string, error) {
	query, ok := args["query"]
	if !ok {
		return "", fmt.Errorf("missing 'query' argument")
	}

	results, err := memory.SearchCodebase(query, 5)
	if err != nil {
		return fmt.Sprintf("Codebase search failed: %v", err), err
	}

	if len(results) == 0 {
		return "No codebase chunks found for this query.", nil
	}
	return "Codebase Snippets:\n\n" + strings.Join(results, "\n\n"), nil
}

// WriteFileTool creates or overwrites a file with native Go file I/O safely bypassing shell quote escaping.
type WriteFileTool struct{}

func (s *WriteFileTool) Name() string {
	return "write_file"
}

func (s *WriteFileTool) Description() string {
	return "Creates or overwrites a file natively. Arguments: 'filepath', 'content'."
}

func (s *WriteFileTool) Execute(args map[string]string) (string, error) {
	filepathAuth, ok1 := args["filepath"]
	content, ok2 := args["content"]
	if !ok1 || !ok2 {
		return "", fmt.Errorf("missing 'filepath' or 'content' arguments")
	}

	if err := os.WriteFile(filepathAuth, []byte(content), 0644); err != nil {
		return fmt.Sprintf("Failed to write file: %v", err), err
	}
	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), filepathAuth), nil
}

// ReplaceContentTool replaces a specific string inside a file natively.
type ReplaceContentTool struct{}

func (s *ReplaceContentTool) Name() string {
	return "replace_content"
}

func (s *ReplaceContentTool) Description() string {
	return "Surgically replaces an exact target string in a file with new content without deleting the rest of the file. Arguments: 'filepath', 'target_text', 'replacement_text'."
}

func (s *ReplaceContentTool) Execute(args map[string]string) (string, error) {
	filepathAuth, ok1 := args["filepath"]
	targetText, ok2 := args["target_text"]
	repText, ok3 := args["replacement_text"]
	if !ok1 || !ok2 || !ok3 {
		return "", fmt.Errorf("missing arguments")
	}

	bytes, err := os.ReadFile(filepathAuth)
	if err != nil {
		return fmt.Sprintf("Failed to read file: %v", err), err
	}
	content := string(bytes)

	if !strings.Contains(content, targetText) {
		return "Error: target_text not found in file. Ensure exact match including whitespace/indentation.", nil
	}

	newContent := strings.Replace(content, targetText, repText, 1)
	if err := os.WriteFile(filepathAuth, []byte(newContent), 0644); err != nil {
		return fmt.Sprintf("Failed to write to file: %v", err), err
	}
	return fmt.Sprintf("Successfully replaced target text in %s", filepathAuth), nil
}

// UpdateStateTool manages the persistent short-term agent scratchpad
type UpdateStateTool struct{}

func (s *UpdateStateTool) Name() string {
	return "update_active_state"
}

func (s *UpdateStateTool) Description() string {
	return "Writes to the persistent agent scratchpad (active_state.md). Use this whenever you start a task, achieve a milestone, or need to remember something you are doing across reboots. Completely overwrites previous state."
}

func (s *UpdateStateTool) Execute(args map[string]string) (string, error) {
	content, ok := args["state_markdown"]
	if !ok {
		return "", fmt.Errorf("missing 'state_markdown' argument")
	}

	cfgDir, err := config.GetConfigDir()
	if err != nil {
		return "", err
	}
	statePath := filepath.Join(cfgDir, "active_state.md")

	err = os.WriteFile(statePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write active state: %w", err)
	}
	return "Successfully updated and persisted active_state.md to disk.", nil
}

// DispatchAgentTool provides the schema for Keith to spawn isolated sub-agents. Execution is natively intercepted.
type DispatchAgentTool struct{}

func (s *DispatchAgentTool) Name() string {
	return "dispatch_sub_agent"
}

func (s *DispatchAgentTool) Description() string {
	return "Dispatches an isolated subordinate LLM agent (implementer, reviewer, or ui_agent) to autonomously perform a task. Your execution will natively BLOCK until the sub-agent completes its sub-loop and returns the final verified result. Use this to orchestrate specialized workflows without doing the work yourself."
}

func (s *DispatchAgentTool) Execute(args map[string]string) (string, error) {
	return "Error: This tool fell through to the raw router. The Swarm Orchestrator should have intercepted this natively in loop.go.", nil
}

// SendTelegramMessageTool sends a message to a specific Chat ID.
type SendTelegramMessageTool struct{}

func (s *SendTelegramMessageTool) Name() string { return "send_telegram_message" }

func (s *SendTelegramMessageTool) Description() string {
	return "Sends a Telegram message to a specific Chat ID. Arguments: 'chat_id' (string), 'message' (string)."
}

func (s *SendTelegramMessageTool) Execute(args map[string]string) (string, error) {
	chatID, ok := args["chat_id"]
	if !ok || chatID == "" {
		return "", fmt.Errorf("missing 'chat_id' argument")
	}
	message, ok := args["message"]
	if !ok || message == "" {
		return "", fmt.Errorf("missing 'message' argument")
	}

	accs, err := config.LoadAccounts()
	if err != nil || accs.TelegramToken == "" {
		return "", fmt.Errorf("Telegram is not configured")
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", accs.TelegramToken)
	payload := map[string]interface{}{"chat_id": chatID, "text": message}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("failed to send telegram message, status: %d", resp.StatusCode)
	}

	return fmt.Sprintf("Successfully sent Telegram message natively to Chat ID %s", chatID), nil
}

// SendTelegramPhotoTool physically blasts a PNG binary to a Telegram Chat ID via multipart/form-data
type SendTelegramPhotoTool struct{}

func (s *SendTelegramPhotoTool) Name() string { return "send_telegram_photo" }

func (s *SendTelegramPhotoTool) Description() string {
	return "Sends a physical image file (like a screenshot) directly to a Telegram Chat ID. Arguments: 'chat_id' (string), 'filepath' (string, e.g. /tmp/keith_vision_matrix.png, MUST be an absolute path)."
}

func (s *SendTelegramPhotoTool) Execute(args map[string]string) (string, error) {
	chatID, ok := args["chat_id"]
	if !ok || chatID == "" {
		return "", fmt.Errorf("missing 'chat_id' argument")
	}
	filepathStr, ok := args["filepath"]
	if !ok || filepathStr == "" {
		return "", fmt.Errorf("missing 'filepath' argument")
	}

	accs, err := config.LoadAccounts()
	if err != nil || accs.TelegramToken == "" {
		return "", fmt.Errorf("Telegram is not configured")
	}

	file, err := os.Open(filepathStr)
	if err != nil {
		return "", fmt.Errorf("failed to open physical image file: %v", err)
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	_ = writer.WriteField("chat_id", chatID)

	part, err := writer.CreateFormFile("photo", filepath.Base(filepathStr))
	if err != nil {
		return "", err
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return "", err
	}
	err = writer.Close()
	if err != nil {
		return "", err
	}

	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendPhoto", accs.TelegramToken)
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return "", err
	}
	req.Header.Add("Content-Type", writer.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("multi-part telegram transmission failed with status: %d", resp.StatusCode)
	}

	return fmt.Sprintf("Successfully transmitted Image to Telegram Chat ID %s", chatID), nil
}

// SendUIImageTool teleport an image straight to the user's Web UI chat visualizer
type SendUIImageTool struct{}

func (s *SendUIImageTool) Name() string { return "send_ui_image" }

func (s *SendUIImageTool) Description() string {
	return "Transmits an image explicitly to the user's graphical Web UI chat frame so they can see it alongside your messages! Arguments: 'filepath' (string, absolute local path)."
}

func (s *SendUIImageTool) Execute(args map[string]string) (string, error) {
	filepathStr, ok := args["filepath"]
	if !ok || filepathStr == "" {
		return "", fmt.Errorf("missing 'filepath' argument")
	}

	workDir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("could not resolve working directory: %v", err)
	}

	timestamp := time.Now().UnixNano()
	filename := fmt.Sprintf("keith_vision_%d.png", timestamp)
	targetPath := filepath.Join(workDir, "web", filename)

	out, err := os.Create(targetPath)
	if err != nil {
		return "", err
	}
	defer out.Close()

	file, err := os.Open(filepathStr)
	if err != nil {
		return "", err
	}
	defer file.Close()

	_, err = io.Copy(out, file)
	if err != nil {
		return "", err
	}

	markdownLiteral := fmt.Sprintf("![Desktop Screenshot](/%s)", filename)
	return fmt.Sprintf("Image perfectly duplicated to the static Web Server at %s. YOU MUST NOW output this exact Markdown string in your final conversation response to the user so it physically renders on their screen: %s", targetPath, markdownLiteral), nil
}

// TelegramLogoutTool revokes all Telegram sessions.
type TelegramLogoutTool struct{}

func (s *TelegramLogoutTool) Name() string { return "telegram_logout_all" }

func (s *TelegramLogoutTool) Description() string {
	return "Instantly revokes all active Telegram sessions securely across the network."
}

func (s *TelegramLogoutTool) Execute(args map[string]string) (string, error) {
	accs, err := config.LoadAccounts()
	if err != nil {
		return "", err
	}
	accs.TelegramSessions = make(map[string]int64)
	config.SaveAccounts(accs)
	return "All Telegram users have been forcibly logged out of the Neural Gateway.", nil
}

// ToolRegistry holds available tools.
type ToolRegistry struct {
	tools map[string]Tool
}

func NewToolRegistry() *ToolRegistry {
	r := &ToolRegistry{
		tools: make(map[string]Tool),
	}
	r.Register(&ShellTool{})
	r.Register(&SaveFactTool{})
	r.Register(&SearchMemoryTool{})
	r.Register(&SearchCodebaseTool{})
	r.Register(&WriteFileTool{})
	r.Register(&ReplaceContentTool{})
	r.Register(&UpdateStateTool{})
	r.Register(&DispatchAgentTool{})
	r.Register(&SendTelegramMessageTool{})
	r.Register(&TelegramLogoutTool{})
	r.Register(&SaveGmailCredentialsTool{})
	r.Register(&SendEmailTool{})
	r.Register(&ReadEmailsTool{})
	r.Register(&CreateScheduledTaskTool{})
	r.Register(&ListScheduledTasksTool{})
	r.Register(&DeleteScheduledTaskTool{})
	r.Register(&ReadMacCalendarTool{})
	r.Register(&CreateMacCalendarEventTool{})
	r.Register(&ConnectGoogleCalendarTool{})
	r.Register(&ReadGoogleCalendarTool{})
	r.Register(&CreateGoogleEventTool{})
	r.Register(&TakeScreenshotTool{})
	r.Register(&MoveMouseTool{})
	r.Register(&ClickMouseTool{})
	r.Register(&TypeKeyboardTool{})
	r.Register(&PressKeyTool{})
	r.Register(&TypeShortcutTool{})
	r.Register(&SendTelegramPhotoTool{})
	r.Register(&SendUIImageTool{})
	r.Register(&ControlMusicPlaybackTool{})
	r.Register(&GetCurrentTrackTool{})
	r.Register(&PlaySpecificSongTool{})
	r.Register(&ControlVolumeTool{})
	r.Register(&ControlWiFiTool{})
	r.Register(&ControlBluetoothTool{})
	r.Register(&ControlDisplayBrightnessTool{})
	r.Register(&SpeakTextTool{})
	r.Register(&UpdateConfigTool{})
	return r
}

func (r *ToolRegistry) Register(t Tool) {
	r.tools[t.Name()] = t
}

func (r *ToolRegistry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// ExportToolSchemas returns an array of maps representing Gemini FunctionDeclarations.
func (r *ToolRegistry) ExportToolSchemas() []map[string]interface{} {
	var schemas []map[string]interface{}
	for _, t := range r.tools {
		var params map[string]interface{}

		switch t.Name() {
		case "shell_exec":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"description": "The exact bash command to run.",
					},
					"is_background": map[string]interface{}{
						"type":        "string",
						"description": "Optional: set to 'true' if the command kicks off a long-running process like 'npm run dev' or a server. It will detach the process and return immediately to prevent freezing.",
					},
				},
				"required": []string{"command"},
			}
		case "save_fact":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"fact": map[string]interface{}{
						"type":        "string",
						"description": "The fact, detail, or preference to permanently commit to memory.",
					},
				},
				"required": []string{"fact"},
			}
		case "search_memory":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "The semantic search query used to find related memories.",
					},
				},
				"required": []string{"query"},
			}
		case "search_codebase":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "The semantic natural-language query to find exact source code chunks inside the ingested repository.",
					},
				},
				"required": []string{"query"},
			}
		case "write_file":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"filepath": map[string]interface{}{
						"type":        "string",
						"description": "Absolute or relative path to the file to create or overwrite.",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "The raw string content to write into the file.",
					},
				},
				"required": []string{"filepath", "content"},
			}
		case "replace_content":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"filepath": map[string]interface{}{
						"type":        "string",
						"description": "Absolute or relative path to the file to edit.",
					},
					"target_text": map[string]interface{}{
						"type":        "string",
						"description": "The EXACT string to find and replace. Must match indentation/whitespace perfectly.",
					},
					"replacement_text": map[string]interface{}{
						"type":        "string",
						"description": "The new string to replace the target_text with.",
					},
				},
				"required": []string{"filepath", "target_text", "replacement_text"},
			}
		case "update_active_state":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"state_markdown": map[string]interface{}{
						"type":        "string",
						"description": "A comprehensive markdown string detailing exactly what you are currently working on, any recent milestones, and your very next planned step. Example: 'Working on Vite app in ~/Desktop. Scaffolded it. Next step: npm install.'",
					},
				},
				"required": []string{"state_markdown"},
			}
		case "dispatch_sub_agent":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"role": map[string]interface{}{
						"type":        "string",
						"description": "The persona of the sub-agent. Must be 'implementer', 'reviewer', or 'ui_agent'.",
					},
					"task": map[string]interface{}{
						"type":        "string",
						"description": "The explicit, detailed objective the sub-agent must complete.",
					},
				},
				"required": []string{"role", "task"},
			}
		case "send_telegram_message":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"chat_id": map[string]interface{}{
						"type":        "string",
						"description": "The target Telegram Chat ID to send the textual message to.",
					},
					"message": map[string]interface{}{
						"type":        "string",
						"description": "The complete message payload to send.",
					},
				},
				"required": []string{"chat_id", "message"},
			}
		case "telegram_logout_all":
			params = map[string]interface{}{
				"type": "object",
			}
		case "save_gmail_credentials":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"email": map[string]interface{}{
						"type":        "string",
						"description": "The user's standard Gmail address.",
					},
					"app_password": map[string]interface{}{
						"type":        "string",
						"description": "The 16-character Google App Password generated by the user.",
					},
				},
				"required": []string{"email", "app_password"},
			}
		case "send_email":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"to": map[string]interface{}{
						"type":        "string",
						"description": "The destination email address.",
					},
					"subject": map[string]interface{}{
						"type":        "string",
						"description": "The subject line of the email.",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "The textual body content of the email.",
					},
				},
				"required": []string{"to", "subject", "body"},
			}
		case "read_emails":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"limit": map[string]interface{}{
						"type":        "string",
						"description": "The maximum number of recent emails to read from the Inbox (e.g. '5' or '10'). Defaults to 5.",
					},
				},
			}
		case "create_scheduled_task":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"type": map[string]interface{}{
						"type":        "string",
						"description": "Must be either 'one-off' or 'recurring'.",
					},
					"execute_at_unix": map[string]interface{}{
						"type":        "string",
						"description": "(For one-off only) The exact future Unix Epoch Timestamp (seconds) when this should execute. Calculate mathematically using System Time.",
					},
					"interval_minutes": map[string]interface{}{
						"type":        "string",
						"description": "(For recurring only) The delay in minutes between executions (e.g., 15).",
					},
					"task_prompt": map[string]interface{}{
						"type":        "string",
						"description": "Your exact instructions for what to do when this triggers (e.g., 'Message the user and say drink water').",
					},
					"chat_id": map[string]interface{}{
						"type":        "string",
						"description": "The destination Telegram Chat ID to target.",
					},
				},
				"required": []string{"type", "task_prompt", "chat_id"},
			}
		case "list_scheduled_tasks":
			params = map[string]interface{}{
				"type": "object",
			}
		case "delete_scheduled_task":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"id": map[string]interface{}{
						"type":        "string",
						"description": "The exact ID of the scheduled job to delete.",
					},
				},
				"required": []string{"id"},
			}
		case "read_mac_calendar":
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		case "create_mac_calendar_event":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"calendar_name": map[string]interface{}{
						"type":        "string",
						"description": "Exact name of the calendar to inject the event into (e.g., 'Personal' or 'Work').",
					},
					"title": map[string]interface{}{
						"type":        "string",
						"description": "Event summary.",
					},
					"start_date": map[string]interface{}{
						"type":        "string",
						"description": "Exact string format for AppleScript (e.g., 'February 27, 2026 at 3:00:00 PM').",
					},
					"end_date": map[string]interface{}{
						"type":        "string",
						"description": "Exact string format for AppleScript (e.g., 'February 27, 2026 at 4:00:00 PM').",
					},
				},
				"required": []string{"calendar_name", "title", "start_date", "end_date"},
			}
		case "connect_google_calendar":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"profile": map[string]interface{}{
						"type":        "string",
						"description": "Logical profile name (e.g. 'personal').",
					},
				},
				"required": []string{"profile"},
			}
		case "read_google_calendar":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"profile": map[string]interface{}{"type": "string"},
					"limit":   map[string]interface{}{"type": "string"},
				},
				"required": []string{"profile"},
			}
		case "create_google_event":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"profile":        map[string]interface{}{"type": "string"},
					"title":          map[string]interface{}{"type": "string"},
					"start_time_iso": map[string]interface{}{"type": "string", "description": "RFC3339 format, e.g. 2026-02-27T15:00:00+05:30"},
					"end_time_iso":   map[string]interface{}{"type": "string", "description": "RFC3339 format"},
				},
				"required": []string{"profile", "title", "start_time_iso", "end_time_iso"},
			}
		case "take_screenshot":
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		case "move_mouse":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"x": map[string]interface{}{"type": "string", "description": "Absolute Screen Pixel X"},
					"y": map[string]interface{}{"type": "string", "description": "Absolute Screen Pixel Y"},
				},
				"required": []string{"x", "y"},
			}
		case "click_mouse":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"x": map[string]interface{}{"type": "string", "description": "Absolute Screen Pixel X"},
					"y": map[string]interface{}{"type": "string", "description": "Absolute Screen Pixel Y"},
				},
				"required": []string{"x", "y"},
			}
		case "type_keyboard":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{"type": "string", "description": "Literal string you want to inject"},
				},
				"required": []string{"text"},
			}
		case "press_key":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"key": map[string]interface{}{"type": "string", "description": "Special native keyboard key like 'return', 'tab', 'esc', 'space', 'delete'"},
				},
				"required": []string{"key"},
			}
		case "type_shortcut":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"modifiers": map[string]interface{}{"type": "string", "description": "Comma-separated modifier keys (e.g. 'ctrl,shift,cmd,alt'). Leave empty if none."},
					"key":       map[string]interface{}{"type": "string", "description": "The main key to press (e.g. 'arrow-left', 'c')"},
				},
				"required": []string{"key"},
			}
		case "send_telegram_photo":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"chat_id":  map[string]interface{}{"type": "string", "description": "Telegram user's Chat ID number"},
					"filepath": map[string]interface{}{"type": "string", "description": "Absolute path to the image, e.g. /tmp/keith_vision_matrix.png"},
				},
				"required": []string{"chat_id", "filepath"},
			}
		case "send_ui_image":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"filepath": map[string]interface{}{"type": "string", "description": "Absolute path to the image, e.g. /tmp/keith_vision_matrix.png"},
				},
				"required": []string{"filepath"},
			}
		case "control_music_playback":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"player": map[string]interface{}{"type": "string", "description": "Must be 'spotify' or 'music'"},
					"action": map[string]interface{}{"type": "string", "description": "Must be 'play', 'pause', 'playpause', 'next track', or 'previous track'"},
				},
				"required": []string{"player", "action"},
			}
		case "get_current_track":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"player": map[string]interface{}{"type": "string", "description": "Must be 'spotify' or 'music'"},
				},
				"required": []string{"player"},
			}
		case "play_specific_song":
			params = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"player": map[string]interface{}{"type": "string", "description": "Must be 'spotify' or 'music'"},
					"query":  map[string]interface{}{"type": "string", "description": "e.g., 'Michael Jackson Beat It'"},
				},
				"required": []string{"player", "query"},
			}
		}

		schema := map[string]interface{}{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters":  params,
		}
		schemas = append(schemas, schema)
	}
	return schemas
}
