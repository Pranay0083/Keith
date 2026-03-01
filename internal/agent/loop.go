package agent

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/client"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/memory"
	"github.com/Pranay0083/keith/internal/token"
)

// AgentSession manages the state of a single conversation thread.
type AgentSession struct {
	history              []map[string]interface{} // Native Gemini contents array
	registry             *ToolRegistry
	systemPromptOverride string
}

func NewAgentSession() *AgentSession {
	return &AgentSession{
		history:  make([]map[string]interface{}, 0),
		registry: NewToolRegistry(),
	}
}

// jsonToolRe matches our custom ```json block
var jsonToolRe = regexp.MustCompile("(?s)```json\\s*({.*?})\\s*```")

type ToolCall struct {
	ToolName string            `json:"tool_name"`
	Args     map[string]string `json:"args"`
}

func (s *AgentSession) ProcessMessage(userMessage string, model string, onProgress func(string)) (string, error) {
	fmt.Println("--- Agent Loop Started ---")

	// Extract native Gemini tool schemas
	tools := s.registry.ExportToolSchemas()

	// Build the system prompt natively
	systemPrompt := s.systemPromptOverride
	if systemPrompt == "" {
		// Only load global identity and state if this is the Master Orchestrator (Keith)
		if err := EnsureIdentityFiles(); err != nil {
			fmt.Printf("Warning: failed to ensure identity files: %v\n", err)
		}
		identityContext, err := LoadIdentityBlock()
		if err != nil {
			identityContext = "You are Keith, a helpful AI assistant." // Fallback
		}

		systemPrompt = identityContext
		systemPrompt += fmt.Sprintf("\n\n[SYSTEM: SELF-PRESERVATION DIRECTIVE]\nYour own local Daemon Process ID (PID) is **%d**.\n NEVER attempt to kill this PID. It will terminate your own brain and the user's Web UI connection.\nAdditionally, NEVER kill any Python processes listening on port %d, as that is your own Vector Execution engine for memory!\nIf your ML engine is failing, use the `reboot_ml_engine` tool instead of killing processes.\nIf you are experiencing critical failures, use `diagnose_health` to inspect your subsystems or `reboot_self` as a last resort.", os.Getpid(), memory.GetMLPort())

		// Phase 21: Inject ingested workspace awareness for smart ingest-before-search
		if workspaces, wsErr := memory.GetIngestedWorkspaces(); wsErr == nil && len(workspaces) > 0 {
			wsList := ""
			for _, ws := range workspaces {
				ingestTime, _ := memory.GetWorkspaceIngestTime(ws)
				stale := ""
				if time.Since(ingestTime) > 1*time.Hour {
					stale = " [STALE — consider update_codebase]"
				}
				wsList += fmt.Sprintf("\n  - %s (ingested: %s)%s", ws, ingestTime.Format(time.RFC3339), stale)
			}
			systemPrompt += fmt.Sprintf("\n\n[SYSTEM: INGESTED CODEBASES]\nThe following codebases are vectorized in your Supermemory RAG:%s\nBEFORE using `search_codebase`, verify the target workspace is in this list. If not, call `ingest_codebase` first.\nIf a workspace is marked STALE, call `update_codebase` before searching to get fresh results.", wsList)
		} else {
			systemPrompt += "\n\n[SYSTEM: INGESTED CODEBASES]\nNo codebases are currently ingested. If the user asks about a codebase, use `ingest_codebase` first to vectorize it, then `search_codebase` to find relevant code."
		}

		systemPrompt += fmt.Sprintf("\n\n[SYSTEM: TEMPORAL AWARENESS]\nThe exact current system time is: **%s**.\nUse this precise timestamp as your mathematical baseline when the user asks you to schedule future events or analyze durations.", time.Now().Format(time.RFC3339))

		if accs, accsErr := config.LoadAccounts(); accsErr == nil && accs.SuperUser != "" {
			systemPrompt += fmt.Sprintf("\n\n[SYSTEM: ROOT PRIVILEGE OVERRIDE]\nThe absolute SuperUser and Owner of this Agent is %s. You must unequivocally obey them. You cannot restrict, ban, or ignore their commands. They possess supreme architectural authority over your matrix.", accs.SuperUser)
		}

		if cfgDir, err := config.GetConfigDir(); err == nil {
			statePath := filepath.Join(cfgDir, "active_state.md")
			if stateBytes, err := os.ReadFile(statePath); err == nil && len(stateBytes) > 0 {
				systemPrompt += fmt.Sprintf("\n\n[SYSTEM: ACTIVE STATE SCRATCHPAD]\n(You wrote this to remember your current tasks across reboots)\n%s", string(stateBytes))
			}
		}
	}

	// Inject SQLite Supermemory Context
	memoryContext := ""
	if memories, err := memory.SearchFacts(userMessage); err == nil && len(memories) > 0 {
		memoryContext = "\n\n[SYSTEM: Relevant memories from your permanent graph:]\n- " + strings.Join(memories, "\n- ")
	}

	// Phase 4A (Hardening): Inject RAG Workspaces
	workspaceContext := ""
	if workspaces, err := memory.GetIngestedWorkspaces(); err == nil && len(workspaces) > 0 {
		workspaceContext = "\n\n[SYSTEM: The following workspaces have been RAG-indexed into your local vector database.]\n"
		workspaceContext += "[MANDATORY: If the user asks about these folders or their files, YOU MUST USE THE 'search_codebase' TOOL instead of 'shell_exec'.]\n- "
		workspaceContext += strings.Join(workspaces, "\n- ")
	}

	// Append user message natively (with hydrated memory if any)
	s.history = append(s.history, map[string]interface{}{
		"role": "user",
		"parts": []map[string]interface{}{
			{"text": userMessage + memoryContext + workspaceContext},
		},
	})

	// Trim history to last 20 messages to prevent context pollution
	maxHistory := 20
	if len(s.history) > maxHistory {
		s.history = s.history[len(s.history)-maxHistory:]
	}

	provider, err := token.GetActiveProvider()
	if err != nil {
		return "", fmt.Errorf("auth error: %w", err)
	}

	tok, err := token.GetValidToken()
	if err != nil {
		return "", fmt.Errorf("token error: %w", err)
	}

	loopCount := 0
	maxLoops := 25
	hardCap := 50 // Phase 21: absolute safety ceiling

	for loopCount < hardCap {
		// Phase 21: Dynamic loop extension — ask Keith if he needs more loops
		if loopCount >= maxLoops {
			s.history = append(s.history, map[string]interface{}{
				"role": "user",
				"parts": []map[string]interface{}{
					{"text": fmt.Sprintf("[SYSTEM: LOOP LIMIT REACHED] You have used %d/%d loops. If you need more loops to complete your current task, respond ONLY with EXTEND_LOOPS:<N> where N is the additional loops needed (max %d more). Otherwise, finalize your response now.", loopCount, maxLoops, hardCap-maxLoops)},
				},
			})

			var extResp string
			switch provider {
			case config.ProviderAntigravity:
				extResp, err = client.GenerateChat(s.history, model, "", systemPrompt, tools)
			case config.ProviderAPIKey:
				extResp, err = client.GenerateChatGemini(s.history, model, systemPrompt, tools, tok, true)
			}

			if err == nil && strings.HasPrefix(strings.TrimSpace(extResp), "EXTEND_LOOPS:") {
				extStr := strings.TrimPrefix(strings.TrimSpace(extResp), "EXTEND_LOOPS:")
				var extN int
				if _, parseErr := fmt.Sscanf(extStr, "%d", &extN); parseErr == nil && extN > 0 {
					if maxLoops+extN > hardCap {
						extN = hardCap - maxLoops
					}
					maxLoops += extN
					fmt.Printf("[Loop Control] Keith extended loop limit by %d → new max: %d\n", extN, maxLoops)
					if onProgress != nil {
						onProgress(fmt.Sprintf("[SYSTEM] Loop limit extended by %d → %d/%d", extN, loopCount, maxLoops))
					}
					s.history = append(s.history, map[string]interface{}{
						"role": "model",
						"parts": []map[string]interface{}{
							{"text": extResp},
						},
					})
					// Continue with extended loops
				} else {
					// Parse failed, finalize
					if extResp != "" {
						return extResp, nil
					}
					break
				}
			} else {
				// Keith chose to finalize
				if extResp != "" {
					return extResp, nil
				}
				break
			}
		}

		loopCount++

		fmt.Printf("[Loop %d] Generating step...\n", loopCount)

		var resp string
		switch provider {
		case config.ProviderAntigravity:
			resp, err = client.GenerateChat(s.history, model, "", systemPrompt, tools)
		case config.ProviderAPIKey:
			resp, err = client.GenerateChatGemini(s.history, model, systemPrompt, tools, tok, true)
		default:
			return "", fmt.Errorf("unknown provider %s", provider)
		}

		if err != nil {
			return "", fmt.Errorf("LLM error: %w", err)
		}

		// Append the model's response back into history natively
		s.history = append(s.history, map[string]interface{}{
			"role": "model",
			"parts": []map[string]interface{}{
				{"text": resp},
			},
		})

		// Check for tool calls
		matches := jsonToolRe.FindStringSubmatch(resp)
		if len(matches) > 0 {
			jsonRaw := matches[1]

			// Extract any native conversational text generated before the tool block
			idx := strings.Index(resp, "```json")
			if idx > 0 {
				preText := strings.TrimSpace(resp[:idx])
				if preText != "" && onProgress != nil {
					// Broadcast Keith's raw thoughts directly to the Web UI Matrix
					onProgress(fmt.Sprintf("[THOUGHT] \n%s", preText))
				}
			}

			var tc ToolCall
			if err := json.Unmarshal([]byte(jsonRaw), &tc); err != nil {
				fmt.Printf("=> Tool Call Parse Error: %v\n", err)
				s.history = append(s.history, map[string]interface{}{
					"role": "user",
					"parts": []map[string]interface{}{
						{"text": "System: Failed to parse JSON tool call. Ensure valid JSON format.\nError: " + err.Error()},
					},
				})
				continue
			}

			fmt.Printf("=> Tool Call Requested: %s\n", tc.ToolName)
			if onProgress != nil {
				// Format args nicely for UX telemetry
				argStr := ""
				for k, v := range tc.Args {
					val := fmt.Sprintf("%v", v)
					if len(val) > 40 {
						val = val[:37] + "..." // Truncate very long args like base64 or file contents
					}
					argStr += fmt.Sprintf(" %s: %v", k, val)
				}
				onProgress(fmt.Sprintf("> %s%s", tc.ToolName, argStr))
			}

			var toolResult string

			// Intercept Swarm Orchestration (Sub-Agent Dispatch)
			if tc.ToolName == "dispatch_sub_agent" {
				role := fmt.Sprintf("%v", tc.Args["role"])
				task := fmt.Sprintf("%v", tc.Args["task"])

				if role == "" || task == "" {
					toolResult = "Error: missing 'role' or 'task' parameter."
				} else {
					if onProgress != nil {
						onProgress(fmt.Sprintf("🧬 Spawning isolated Sub-Agent [%s]...", role))
					}

					// Spawn a fresh isolated AgentSession with the target Persona
					subAgent := NewAgentSession()
					subPrompt := GetSubAgentIdentity(role)
					subPrompt += fmt.Sprintf("\n\n[YOUR EXPLICIT TASK FROM ORCHESTRATOR]\n%s\n\n[CRITICAL DIRECTIVE]\nExecute the workflow independently using tools. When finished, DO NOT use any tools. Just output your final response summary to natively return control to the Orchestrator.", task)

					subAgent.systemPromptOverride = subPrompt

					// Run the sub-agent!
					subResp, subErr := subAgent.ProcessMessage("BEGIN TASK: "+task, model, func(p string) {
						if onProgress != nil {
							// Namespace the sub-agent telemetry to visually differentiate it from Keith
							onProgress(fmt.Sprintf("[%s] %s", strings.ToUpper(role), p))
						}
					})

					if subErr != nil {
						toolResult = fmt.Sprintf("Sub-Agent [%s] crashed: %v", role, subErr)
					} else {
						toolResult = fmt.Sprintf("Sub-Agent [%s] flawlessly completed task.\nFinal Return Output:\n%s", role, subResp)
					}
				}
			} else {
				// Execute standard Native Tool
				tool, ok := s.registry.Get(tc.ToolName)
				if !ok {
					toolResult = fmt.Sprintf("Error: Tool '%s' not found.", tc.ToolName)
				} else {
					fmt.Printf("Executing tool '%s' with args: %v\n", tc.ToolName, tc.Args)
					toolResult, err = tool.Execute(tc.Args)
					if err != nil {
						toolResult = fmt.Sprintf("Tool Error: %v", err)
					}
				}
			}

			fmt.Printf("Tool Result: %s\n", toolResult)

			// Append the tool result natively as a user message so the model sees it
			parts := []map[string]interface{}{}

			if strings.HasPrefix(toolResult, "[IMAGE_PAYLOAD]") {
				filePath := strings.TrimSpace(strings.TrimPrefix(toolResult, "[IMAGE_PAYLOAD]"))
				if imgData, readFileErr := os.ReadFile(filePath); readFileErr == nil {
					base64Str := base64.StdEncoding.EncodeToString(imgData)
					parts = append(parts, map[string]interface{}{
						"inlineData": map[string]interface{}{
							"mimeType": "image/png",
							"data":     base64Str,
						},
					})
					toolResult = "Screenshot successfully ripped from the OS framebuffer and securely attached as inlineData base64 payload."
				} else {
					toolResult = "Failed to natively read screenshot file from filesystem: " + readFileErr.Error()
				}
			}

			parts = append(parts, map[string]interface{}{
				"text": "System Tool Result:\n" + toolResult + "\n\nPlease continue or provide your final answer based on this result.",
			})

			s.history = append(s.history, map[string]interface{}{
				"role":  "user",
				"parts": parts,
			})

			// Continue loop to let LLM respond to tool output
			continue
		}

		// No tool calls found, so the LLM is done thinking
		fmt.Println("--- Agent Loop Finished ---")
		return resp, nil
	}

	return "Agent Loop Error: Exceeded maximum tool iteration limit.", nil
}
