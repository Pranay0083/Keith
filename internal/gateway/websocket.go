package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/memory"
	"github.com/Pranay0083/keith/internal/models"
	"github.com/Pranay0083/keith/internal/token"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all for local dev
	},
}

// WebsocketMessage represents the JSON payload back and forth
type WebsocketMessage struct {
	Type    string `json:"type"`            // "user_msg", "bot_chunk", "bot_done", "error"
	Content string `json:"content"`         // Text content
	Model   string `json:"model,omitempty"` // Model to use
}

type Client struct {
	conn *websocket.Conn
	mu   sync.Mutex
}

func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Printf("WebSocket upgrade error: %v\n", err)
		return
	}

	c := &Client{conn: conn}
	defer c.conn.Close()

	fmt.Println("New Web UI client connected")

	for {
		_, msgData, err := c.conn.ReadMessage()
		if err != nil {
			fmt.Printf("Websocket read error (client disconnected): %v\n", err)
			break
		}

		var msg WebsocketMessage
		if err := json.Unmarshal(msgData, &msg); err != nil {
			c.sendError(fmt.Sprintf("Invalid JSON: %v", err))
			continue
		}

		if msg.Type == "user_msg" {
			go s.handleUserMessage(c, msg.Content, msg.Model)
		}
	}
}

func (s *Server) handleUserMessage(c *Client, prompt, model string) {
	fmt.Printf("Received Web prompt: %s\n", prompt)

	provider, err := token.GetActiveProvider()
	if err != nil {
		c.sendError(fmt.Sprintf("Auth error: %v", err))
		return
	}

	if model == "" {
		accs, _ := config.LoadAccounts()
		if accs != nil && accs.DefaultModel != "" {
			model = accs.DefaultModel
		} else {
			model = models.DefaultModel(provider)
		}
	}

	// Phase 4A: Intercept RAG Ingestion Commands
	if strings.HasPrefix(strings.TrimSpace(prompt), "/ingest ") {
		targetPath := strings.TrimSpace(strings.TrimPrefix(prompt, "/ingest "))
		c.sendMessage("bot_chunk", fmt.Sprintf("Scanning and vectorizing codebase at: `%s`...\n\n(This might take a minute depending on repository size. The Python matrix daemon is computing embeddings 100%% offline).", targetPath))

		chunks, err := memory.IngestDirectory(targetPath)
		if err != nil {
			c.sendError(fmt.Sprintf("Ingestion Failed: %v", err))
			return
		}

		c.sendMessage("bot_chunk", fmt.Sprintf("\n\n[SUCCESS] Successfully ingested %d syntax chunks into the native Codebase RAG Graph!\nYour LLM agent can now instantly query this codebase context.", chunks))
		c.sendMessage("bot_done", "")
		return
	}

	fmt.Printf("Routing Web prompt to %s via %s (Agent Loop)\n", model, provider)

	// Get or create session for this connection.
	// We use a static singleton ID for the local Web GUI so that LLM conversation
	// history flawlessly persists even if the React client hot-reloads or drops the socket.
	sessionID := "web_local_admin"
	agentSession := s.GetSession(sessionID)

	// Run the Agentic Loop!
	// This will recursively call tools (like shell) until it gets a final answer.
	resp, err := agentSession.ProcessMessage(prompt, model, func(progress string) {
		c.sendMessage("progress", progress)
	})
	if err != nil {
		c.sendError(err.Error())
		return
	}

	// Send whole response back as a single chunk since we haven't refactored the client package for intercepts yet.
	c.sendMessage("bot_chunk", resp)
	c.sendMessage("bot_done", "")
}

func (c *Client) sendError(errMsg string) {
	fmt.Printf("Sending Web error: %s\n", errMsg)
	c.sendMessage("error", errMsg)
}

func (c *Client) sendMessage(msgType, content string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	msg := WebsocketMessage{
		Type:    msgType,
		Content: content,
	}
	data, _ := json.Marshal(msg)
	c.conn.WriteMessage(websocket.TextMessage, data)
}
