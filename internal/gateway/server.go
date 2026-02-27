package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/Pranay0083/keith/internal/agent"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/memory"
)

type Server struct {
	port     int
	sessions map[string]*agent.AgentSession
	mu       sync.Mutex
}

func NewServer(port int) *Server {
	return &Server{
		port:     port,
		sessions: make(map[string]*agent.AgentSession),
	}
}

func (s *Server) GetSession(id string) *agent.AgentSession {
	s.mu.Lock()
	defer s.mu.Unlock()
	if session, exists := s.sessions[id]; exists {
		return session
	}
	session := agent.NewAgentSession()
	s.sessions[id] = session
	return session
}

func (s *Server) Start() error {
	// Start Telegram Bot if token is present
	botToken := os.Getenv("TELEGRAM_BOT_TOKEN")
	if botToken != "" {
		bot := NewTelegramBot(botToken)
		go bot.StartPolling()
	} else {
		fmt.Println("TELEGRAM_BOT_TOKEN not set. Telegram integration disabled.")
	}

	mux := http.NewServeMux()

	// 1. Serve static files from the 'web' directory
	// Try multiple locations: CWD (dev), next to executable (installed), ~/.config/keith/web (fallback)
	webDir := ""
	candidates := []string{}
	if workDir, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(workDir, "web"))
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "web"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".config", "keith", "web"))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			webDir = c
			break
		}
	}
	if webDir == "" {
		fmt.Println("Warning: Could not locate 'web' directory. Web UI will not be available.")
	}

	fs := http.FileServer(http.Dir(webDir))
	mux.Handle("/", fs)

	// 2. Setup WebSocket endpoint
	mux.HandleFunc("/ws", s.handleWebSocket)

	// Phase 4B: Supermemory REST API
	mux.HandleFunc("/api/memories", s.handleGetMemories)
	mux.HandleFunc("/api/memories/delete", s.handleDeleteMemory)
	mux.HandleFunc("/api/jobs", s.handleGetJobs)
	mux.HandleFunc("/oauth2callback", s.handleOAuthCallback)

	addr := fmt.Sprintf(":%d", s.port)
	fmt.Printf("Starting Keith Gateway on http://localhost%s\n", addr)

	return http.ListenAndServe(addr, mux)
}

// Phase 4B: REST Handlers
func (s *Server) handleGetMemories(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	facts, err := memory.GetAllFacts()
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%v"}`, err), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(facts)
}

func (s *Server) handleDeleteMemory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method != http.MethodPost && r.Method != http.MethodDelete {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ID int `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"Invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if err := memory.DeleteFact(req.ID); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%v"}`, err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"deleted"}`))
}

func (s *Server) handleGetJobs(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	jobs, err := config.LoadJobs()
	if err != nil {
		http.Error(w, "Failed to load jobs", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jobs)
}

func (s *Server) handleOAuthCallback(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	code := r.URL.Query().Get("code")

	if state == "" || code == "" {
		http.Error(w, "Critical Error: Missing state or code from Google Auth pipeline", http.StatusBadRequest)
		return
	}

	oauthConfig, err := config.GetOAuthConfig()
	if err != nil {
		http.Error(w, "OAuth Database Matrix Error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	token, err := oauthConfig.Exchange(context.Background(), code)
	if err != nil {
		http.Error(w, "Natively failed to broker token exchange: "+err.Error(), http.StatusInternalServerError)
		return
	}

	err = config.SaveToken(state, token)
	if err != nil {
		http.Error(w, "Filesystem serialization failed for active Token: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(fmt.Sprintf(`<html><body style="background-color: #0f172a; color: #f8fafc; font-family: monospace; text-align: center; padding: 50px;">
		<h2 style="color: #38bdf8;">Keith Native Google Matrix</h2>
		<p>Authentication completely synthesized for logic profile: <b>%s</b></p>
		<p style="color: #94a3b8;">Physical bindings established to ~/.config/keith/gcal_tokens/</p>
		<p>You can securely close this browser context and command Keith directly in the Terminal UI.</p>
	</body></html>`, state)))
}
