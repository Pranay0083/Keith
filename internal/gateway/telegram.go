package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/memory"
	"github.com/Pranay0083/keith/internal/models"
	"github.com/Pranay0083/keith/internal/token"
)

type TelegramBot struct {
	token    string
	apiURL   string
	client   *http.Client
	sessions map[int64]*agent.AgentSession
	mu       sync.Mutex
}

func NewTelegramBot(token string) *TelegramBot {
	return &TelegramBot{
		token:    token,
		apiURL:   fmt.Sprintf("https://api.telegram.org/bot%s", token),
		client:   &http.Client{Timeout: 60 * time.Second}, // Long-polling timeout
		sessions: make(map[int64]*agent.AgentSession),
	}
}

func (b *TelegramBot) GetSession(chatID int64) *agent.AgentSession {
	b.mu.Lock()
	defer b.mu.Unlock()
	if session, exists := b.sessions[chatID]; exists {
		return session
	}
	session := agent.NewAgentSession()
	b.sessions[chatID] = session
	return session
}

// StartPolling begins long-polling for Telegram updates.
func (b *TelegramBot) StartPolling() {
	fmt.Println("Starting Telegram Bot adapter...")
	go StartChronoEngine(b)
	offset := 0

	for {
		updates, err := b.getUpdates(offset)
		if err != nil {
			fmt.Printf("Telegram polling error: %v\n", err)
			time.Sleep(5 * time.Second)
			continue
		}

		for _, update := range updates {
			offset = int(update["update_id"].(float64)) + 1
			go b.handleUpdate(update)
		}
	}
}

func (b *TelegramBot) getUpdates(offset int) ([]map[string]interface{}, error) {
	url := fmt.Sprintf("%s/getUpdates?offset=%d&timeout=30", b.apiURL, offset)
	resp, err := b.client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Ok     bool                     `json:"ok"`
		Result []map[string]interface{} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if !result.Ok {
		return nil, fmt.Errorf("telegram API returned ok=false")
	}

	return result.Result, nil
}

func (b *TelegramBot) handleUpdate(update map[string]interface{}) {
	message, ok := update["message"].(map[string]interface{})
	if !ok {
		return
	}

	text, hasText := message["text"].(string)
	if !hasText || text == "" {
		return
	}

	chat, hasChat := message["chat"].(map[string]interface{})
	if !hasChat {
		return
	}

	chatIDFloat, hasID := chat["id"].(float64)
	if !hasID {
		return
	}
	chatID := int64(chatIDFloat)

	fmt.Printf("Received Telegram msg from chat %d: %s\n", chatID, text)

	// Send typing indicator
	b.sendChatAction(chatID, "typing")

	// Phase 4A: Intercept RAG Ingestion Commands
	if strings.HasPrefix(strings.TrimSpace(text), "/ingest ") {
		targetPath := strings.TrimSpace(strings.TrimPrefix(text, "/ingest "))

		b.sendMessage(chatID, fmt.Sprintf("Scanning and vectorizing codebase at: `%s`...\n\n(This might take a minute depending on repository size. The Python matrix daemon is computing embeddings 100%% offline).", targetPath))

		chunks, err := memory.IngestDirectory(targetPath)
		if err != nil {
			b.sendMessage(chatID, fmt.Sprintf("Ingestion Failed: %v", err))
			return
		}

		b.sendMessage(chatID, fmt.Sprintf("[SUCCESS] Successfully ingested %d syntax chunks into the native Codebase RAG Graph!\nYour LLM agent can now instantly query this codebase context.", chunks))
		return
	}

	provider, err := token.GetActiveProvider()
	if err != nil {
		b.sendMessage(chatID, fmt.Sprintf("Auth error: %v. Please run 'keith login' locally.", err))
		return
	}

	model := models.DefaultModel(provider)
	accs, confErr := config.LoadAccounts()
	if confErr == nil && accs.DefaultModel != "" {
		model = accs.DefaultModel
	}

	// Phase 9: Secure Auth Bouncer Middleware
	if confErr == nil && accs.TelegramPassword != "" {
		strChatID := fmt.Sprintf("%d", chatID)
		expiry, exists := accs.TelegramSessions[strChatID]

		// If user has no active 48hr session
		if !exists || time.Now().Unix() > expiry {
			if strings.TrimSpace(text) == accs.TelegramPassword {
				// Authenticate
				if accs.TelegramSessions == nil {
					accs.TelegramSessions = make(map[string]int64)
				}
				accs.TelegramSessions[strChatID] = time.Now().Add(48 * time.Hour).Unix()
				config.SaveAccounts(accs)

				// Spoof Native LLM Prompt
				agentSession := b.GetSession(chatID)
				sysMsg := fmt.Sprintf("SYSTEM: I just provided the correct Master Password and authenticated on Telegram with Chat ID %d. Please use your 'search_memory' tool to check if you already know my Chat ID and name. If you recognise me, welcome me back by name. If you do not know me, greet me natively and ask for my name so you can use 'add_fact' to save my Chat ID to your memory.", chatID)

				resp, err := agentSession.ProcessMessage(sysMsg, model, func(progress string) {
					b.sendMessage(chatID, fmt.Sprintf("> %s", progress))
				})
				if err == nil {
					if len(resp) > 4000 {
						for i := 0; i < len(resp); i += 4000 {
							end := i + 4000
							if end > len(resp) {
								end = len(resp)
							}
							b.sendMessage(chatID, resp[i:end])
						}
					} else {
						b.sendMessage(chatID, resp)
					}
				}
				return
			} else {
				b.sendMessage(chatID, "[SECURITY] Access Denied. I am Keith, an autonomous agent. Please provide the master password to access my terminal.")
				return
			}
		}
	}

	// Get or create session
	agentSession := b.GetSession(chatID)

	// Run Agentic Loop
	resp, err := agentSession.ProcessMessage(text, model, func(progress string) {
		b.sendMessage(chatID, fmt.Sprintf("> %s", progress))
	})
	if err != nil {
		b.sendMessage(chatID, fmt.Sprintf("LLM Error: %v", err))
		return
	}

	// Telegram messages have a 4096 character limit.
	// We should split if needed, but for MVP we just send.
	if len(resp) > 4000 {
		resp = resp[:4000] + "\n...[truncated]"
	}

	err = b.sendMessage(chatID, resp)
	if err != nil {
		fmt.Printf("Failed to send Telegram reply: %v\n", err)
	}
}

func (b *TelegramBot) sendChatAction(chatID int64, action string) error {
	url := fmt.Sprintf("%s/sendChatAction", b.apiURL)
	payload := map[string]interface{}{
		"chat_id": chatID,
		"action":  action,
	}

	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func (b *TelegramBot) sendMessage(chatID int64, text string) error {
	// Escape markdown special characters for Telegram MarkdownV2, or just use HTML.
	// For simplicity in MVP, we strip markdown or send as plain text since Telegram throws 400s easily with malformed MarkdownV2.

	// Basic cleanup
	text = strings.ReplaceAll(text, "\\(", "(")
	text = strings.ReplaceAll(text, "\\)", ")")

	url := fmt.Sprintf("%s/sendMessage", b.apiURL)
	payload := map[string]interface{}{
		"chat_id": chatID,
		"text":    text,
		// "parse_mode": "MarkdownV2", // Removed for stability in MVP
	}

	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("telegram returned status %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}
