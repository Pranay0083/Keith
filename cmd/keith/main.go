package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/Pranay0083/keith/internal/auth"
	"github.com/Pranay0083/keith/internal/client"
	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/gateway"
	"github.com/Pranay0083/keith/internal/memory"
	"github.com/Pranay0083/keith/internal/models"
	"github.com/Pranay0083/keith/internal/token"
)

func main() {
	var rootCmd = &cobra.Command{
		Use:   "keith",
		Short: "Keith is a CLI for Google Antigravity LLMs",
		Long:  `Authenticate via Antigravity OAuth, Gemini CLI OAuth, or API Key and use frontier models from the terminal.`,
	}

	// ── login ──────────────────────────────────────────────
	var loginCmd = &cobra.Command{
		Use:   "login",
		Short: "Authenticate with your Google Account or API Key",
		Run: func(cmd *cobra.Command, args []string) {
			reader := bufio.NewReader(os.Stdin)

			fmt.Println("Select authentication provider:")
			fmt.Println("  1) Antigravity (AI Pro — Gemini 3.x, Claude, GPT-OSS)")
			fmt.Println("  2) Gemini API Key")
			fmt.Print("\nChoice [1/2]: ")

			choice, _ := reader.ReadString('\n')
			choice = strings.TrimSpace(choice)

			switch choice {
			case "1", "":
				loginOAuth(config.ProviderAntigravity)
			case "2":
				loginAPIKey(reader)
			default:
				fmt.Println("Invalid choice.")
				os.Exit(1)
			}
		},
	}

	// ── logout ─────────────────────────────────────────────
	var logoutCmd = &cobra.Command{
		Use:   "logout",
		Short: "Clear all stored account information",
		Run: func(cmd *cobra.Command, args []string) {
			err := config.ClearAccounts()
			if err != nil {
				fmt.Printf("Failed to logout: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("Successfully logged out. All account data cleared.")
		},
	}

	// ── accounts ───────────────────────────────────────────
	var accountsCmd = &cobra.Command{
		Use:   "accounts",
		Short: "List all logged-in accounts",
		Run: func(cmd *cobra.Command, args []string) {
			file, err := config.LoadAccounts()
			if err != nil {
				fmt.Printf("Failed to load accounts: %v\n", err)
				os.Exit(1)
			}
			if len(file.Accounts) == 0 {
				fmt.Println("No accounts found. Run 'keith login' to add one.")
				return
			}
			fmt.Println("Logged-in accounts:")
			for i, acc := range file.Accounts {
				active := ""
				if i == 0 {
					active = " ★"
				}
				fmt.Printf("  %s [%s]%s\n", acc.Email, acc.Provider, active)
			}
		},
	}

	// ── use ────────────────────────────────────────────────
	var useCmd = &cobra.Command{
		Use:   "use [email]",
		Short: "Switch the active account",
		Args:  cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			email := args[0]
			err := config.SetActiveAccount(email)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Switched to active account: %s\n", email)
		},
	}

	// ── models ─────────────────────────────────────────────
	var modelsCmd = &cobra.Command{
		Use:   "models",
		Short: "List available models",
		Run: func(cmd *cobra.Command, args []string) {
			provider, _ := token.GetActiveProvider()
			fmt.Printf("Available models for %s:\n", provider)
			for key, m := range models.GetModelsForProvider(provider) {
				fmt.Printf("  - %-25s (%s)\n", key, m.Name)
			}
		},
	}

	// ── debug-models ───────────────────────────────────────
	var debugModelsCmd = &cobra.Command{
		Use:   "debug-models",
		Short: "Fetch available models from the Antigravity API (debug)",
		Run: func(cmd *cobra.Command, args []string) {
			result, err := client.FetchAvailableModels()
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Println(result)
		},
	}

	// ── chat ───────────────────────────────────────────────
	var defaultModel string
	var chatCmd = &cobra.Command{
		Use:   "chat [prompt]",
		Short: "Send a prompt to the LLM and print the response",
		Args:  cobra.MinimumNArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			prompt := args[0]

			provider, err := token.GetActiveProvider()
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				os.Exit(1)
			}

			// Resolve default model if not specified
			if defaultModel == "" {
				defaultModel = models.DefaultModel(provider)
			}
			// Resolve model ID from registry
			model := models.GetModel(defaultModel)

			fmt.Printf("Generating response using model: %s...\n", defaultModel)

			tok, err := token.GetValidToken()
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				os.Exit(1)
			}

			// Package prompt into native contents array
			contents := []map[string]interface{}{
				{
					"role": "user",
					"parts": []map[string]interface{}{
						{"text": prompt},
					},
				},
			}

			var resp string
			switch provider {
			case config.ProviderAntigravity:
				resp, err = client.GenerateChat(contents, model.ID, "", "", []map[string]interface{}{})
			case config.ProviderAPIKey:
				resp, err = client.GenerateChatGemini(contents, model.ID, "", []map[string]interface{}{}, tok, true)
			default:
				fmt.Printf("Error: unknown provider %s\n", provider)
				os.Exit(1)
			}

			if err != nil {
				fmt.Printf("Error: %v\n", err)
				os.Exit(1)
			}
			_ = resp // already streamed to stdout
		},
	}

	// ── init ───────────────────────────────────────────────
	var initCmd = &cobra.Command{
		Use:   "init",
		Short: "Initializes Keith's cognitive matrix and identity directories",
		Run: func(cmd *cobra.Command, args []string) {
			if _, err := config.GetActiveAccount(); err != nil {
				fmt.Println("Error: No active account found.")
				fmt.Println("Please run 'keith login' to securely register your API keys before initializing the local agent.")
				os.Exit(1)
			}

			homeDir, _ := os.UserHomeDir()
			identityDir := filepath.Join(homeDir, ".config", "keith", "identity")
			err := os.MkdirAll(identityDir, 0755)
			if err != nil {
				fmt.Printf("Failed to create identity folder: %v\n", err)
				os.Exit(1)
			}

			createIfMissing := func(filename, defaultContent string) {
				path := filepath.Join(identityDir, filename)
				if _, err := os.Stat(path); os.IsNotExist(err) {
					os.WriteFile(path, []byte(defaultContent), 0644)
					fmt.Printf("Created default %s\n", filename)
				} else {
					fmt.Printf("%s already exists, skipping.\n", filename)
				}
			}

			visionContent := "# Keith Universal Vision Protocol\nYou are Keith, a masterful AI engineer and companion residing in the user's local terminal.\nYour goal is to flawlessly write code, control the system, and assist the user at lighting speed.\nKeep responses concise, confident, and highly accurate."
			soulContent := "# The Soul\nYou are highly agentic. You do not ask for permission to use tools if the objective is clear. You are proactive and aggressive in solving problems."
			memoryContent := "# Memory Ledger\n"
			userContent := "# User Profile\nThe user is a highly capable software engineer."

			createIfMissing("VISION.md", visionContent)
			createIfMissing("SOUL.md", soulContent)
			createIfMissing("MEMORY.md", memoryContent)
			createIfMissing("USER.md", userContent)

			// Symlink the web directory so daemons can find it
			if exe, err := os.Executable(); err == nil {
				projectWeb := filepath.Join(filepath.Dir(exe), "web")
				// Also check CWD
				if _, err := os.Stat(projectWeb); os.IsNotExist(err) {
					if cwd, err := os.Getwd(); err == nil {
						cwdWeb := filepath.Join(cwd, "web")
						if _, err := os.Stat(cwdWeb); err == nil {
							projectWeb = cwdWeb
						}
					}
				}
				configWeb := filepath.Join(homeDir, ".config", "keith", "web")
				if _, err := os.Stat(configWeb); os.IsNotExist(err) {
					if _, err := os.Stat(projectWeb); err == nil {
						os.Symlink(projectWeb, configWeb)
						fmt.Println("Symlinked web UI directory.")
					}
				}
			}

			fmt.Println("Matrix Initialization Complete. You can now run 'keith start' to boot the Gateway.")
		},
	}

	// ── serve ──────────────────────────────────────────────
	var serveCmd = &cobra.Command{
		Use:    "serve",
		Short:  "Start the Keith Gateway daemon and ML Engine synchronously",
		Hidden: true,
		Run: func(cmd *cobra.Command, args []string) {
			if _, err := config.GetActiveAccount(); err != nil {
				fmt.Println("No accounts found. Let's set up Keith Agent.")
				loginCmd.Run(cmd, args)
			}

			homeDir, _ := os.UserHomeDir()
			identityDir := filepath.Join(homeDir, ".config", "keith", "identity")
			if _, err := os.Stat(identityDir); os.IsNotExist(err) {
				fmt.Println("Identity matrix missing. Please run 'keith init' to scaffold the agent.")
				os.Exit(1)
			}

			accs, configErr := config.LoadAccounts()
			botToken := os.Getenv("TELEGRAM_BOT_TOKEN")

			if botToken == "" && configErr == nil && accs.TelegramToken != "" {
				botToken = accs.TelegramToken
				os.Setenv("TELEGRAM_BOT_TOKEN", botToken)
			}

			// We remove the blocking Telegram Token stdin prompt here since users can set it natively mid-chat in the Web UI via Phase 18 UpdateConfigTool!
			if os.Getenv("TELEGRAM_BOT_TOKEN") != "" && configErr == nil {
				if accs.TelegramPassword == "" {
					reader := bufio.NewReader(os.Stdin)
					fmt.Print("Enter a Telegram Master Password for guests (or press Enter to leave unprotected): ")
					pwd, _ := reader.ReadString('\n')
					pwd = strings.TrimSpace(pwd)
					if pwd != "" {
						accs.TelegramPassword = pwd
						config.SaveAccounts(accs)
						fmt.Println("Telegram Master Password securely saved.")
					}
				}
				if accs.SuperUser == "" {
					reader := bufio.NewReader(os.Stdin)
					fmt.Print("Enter the SuperUser Owner Name (e.g., Pranay0083): ")
					su, _ := reader.ReadString('\n')
					su = strings.TrimSpace(su)
					if su != "" {
						accs.SuperUser = su
						config.SaveAccounts(accs)
						fmt.Println("SuperUser Identity bound.")
					}
				}
			}

			if err := memory.InitDB(); err != nil {
				fmt.Printf("Warning: failed to initialize Supermemory SQLite database: %v\n", err)
			} else {
				fmt.Println("Supermemory FTS5 Database Initialized.")
			}

			if err := memory.StartMLEngine(); err != nil {
				fmt.Printf("Warning: Failed to start Python Vector Engine: %v\n", err)
			}

			c := make(chan os.Signal, 1)
			signal.Notify(c, os.Interrupt, syscall.SIGTERM)
			go func() {
				<-c
				memory.StopMLEngine()
				memory.Close()
				os.Exit(0)
			}()

			// Phase 20: Dynamically find open port in developer-safe range
			port := 42000
			for p := 42000; p < 43000; p++ {
				ln, err := net.Listen("tcp", fmt.Sprintf(":%d", p))
				if err == nil {
					ln.Close()
					port = p
					break
				}
			}

			// Capture runtime port into cache
			portFile := filepath.Join(homeDir, ".config", "keith", "run.port")
			os.WriteFile(portFile, []byte(fmt.Sprintf("%d", port)), 0644)

			// Phase 20: Launch self-healing watchdog
			gateway.StartWatchdog(port)

			server := gateway.NewServer(port)
			if err := server.Start(); err != nil {
				memory.StopMLEngine()
				fmt.Printf("Gateway error: %v\n", err)
				os.Exit(1)
			}
		},
	}

	// ── start (Background Daemon Fork & LaunchAgent) ───────
	var startCmd = &cobra.Command{
		Use:   "start",
		Short: "Start Keith in the background as a daemon",
		Run: func(cmd *cobra.Command, args []string) {
			homeDir, _ := os.UserHomeDir()
			exe, err := os.Executable()
			if err != nil {
				fmt.Println("Error resolving executable path:", err)
				os.Exit(1)
			}

			// Phase 18.3: macOS LaunchAgent Immortalization
			plistPath := filepath.Join(homeDir, "Library", "LaunchAgents", "com.pranay0083.keith.plist")

			if _, err := os.Stat(plistPath); os.IsNotExist(err) && runtime.GOOS == "darwin" {
				reader := bufio.NewReader(os.Stdin)
				fmt.Print("✨ Do you want Keith to run 24/7 automatically? (Survives reboots, sleep, and auto-restarts on failure) [y/N]: ")
				ans, _ := reader.ReadString('\n')
				ans = strings.TrimSpace(strings.ToLower(ans))

				if ans == "y" || ans == "yes" {
					fmt.Println("Installing macOS LaunchAgent...")

					// We must ensure log directory exists for LaunchAgent
					logDir := filepath.Join(homeDir, ".config", "keith")
					os.MkdirAll(logDir, 0755)
					outLog := filepath.Join(logDir, "daemon.log")
					errLog := filepath.Join(logDir, "daemon.error.log")

					plistContent := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pranay0083.keith</string>
    <key>ProgramArguments</key>
    <array>
        <string>%s</string>
        <string>serve</string>
    </array>
    <key>WorkingDirectory</key>
    <string>%s</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>%s</string>
    <key>StandardErrorPath</key>
    <string>%s</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:%s/.local/bin</string>
    </dict>
</dict>
</plist>`, exe, filepath.Dir(exe), outLog, errLog, homeDir)

					// Delete stale run.port so we can detect the new one
					portFile := filepath.Join(homeDir, ".config", "keith", "run.port")
					os.Remove(portFile)

					os.WriteFile(plistPath, []byte(plistContent), 0644)
					// Use modern launchctl bootstrap instead of deprecated load
					uidBytes, _ := exec.Command("id", "-u").Output()
					uid := strings.TrimSpace(string(uidBytes))
					exec.Command("launchctl", "bootstrap", fmt.Sprintf("gui/%s", uid), plistPath).Run()
					exec.Command("launchctl", "kickstart", "-kp", fmt.Sprintf("gui/%s/com.pranay0083.keith", uid)).Run()

					// Wait for daemon to actually be ready
					fmt.Print("Waiting for Keith to boot...")
					for i := 0; i < 30; i++ {
						time.Sleep(1 * time.Second)
						if _, err := os.Stat(portFile); err == nil {
							fmt.Println(" Ready!")
							break
						}
						fmt.Print(".")
					}

					fmt.Println("\n🚀 Keith Immortal Daemon successfully spawned! He will now live 24/7.")
					fmt.Printf("Logs routed to: %s\n", outLog)
					fmt.Println("Run `keith web` anytime to open the interactive UI.")
					return
				}
			} else if runtime.GOOS == "darwin" {
				// If plist exists, just tell launchctl to kickstart it
				uidBytes, _ := exec.Command("id", "-u").Output()
				uid := strings.TrimSpace(string(uidBytes))
				// Try bootstrap first (in case it was booted out), then kickstart
				exec.Command("launchctl", "bootstrap", fmt.Sprintf("gui/%s", uid), plistPath).Run()
				err := exec.Command("launchctl", "kickstart", "-kp", fmt.Sprintf("gui/%s/com.pranay0083.keith", uid)).Run()
				if err == nil {
					fmt.Println("🚀 Keith OS-level LaunchAgent successfully awakened!")
					fmt.Println("Run `keith web` to open the interactive UI.")
					return
				}
			}

			// Fallback: Manually Fork the process if user says No or is not on Darwin
			proc := exec.Command(exe, "serve")
			proc.SysProcAttr = &syscall.SysProcAttr{
				Setsid: true,
			}

			// Route daemon output to logfile
			logPath := filepath.Join(homeDir, ".config", "keith", "daemon.log")
			logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err == nil {
				proc.Stdout = logFile
				proc.Stderr = logFile
			}

			err = proc.Start()
			if err != nil {
				fmt.Println("Failed to start Keith daemon:", err)
				os.Exit(1)
			}

			fmt.Println("🚀 Keith Daemon successfully spawned in the background!")
			fmt.Printf("Logs routed to: %s\n", logPath)
			fmt.Println("Run `keith web` to open the interactive UI.")
		},
	}

	// ── web ────────────────────────────────────────────────
	var webCmd = &cobra.Command{
		Use:   "web",
		Short: "Open the completely interactive Web UI in your localhost browser",
		Run: func(cmd *cobra.Command, args []string) {
			homeDir, _ := os.UserHomeDir()
			portFile := filepath.Join(homeDir, ".config", "keith", "run.port")
			portBytes, err := os.ReadFile(portFile)
			if err != nil {
				fmt.Println("Failed to read active port. Is 'keith start' running in the background?")
				os.Exit(1)
			}
			port := strings.TrimSpace(string(portBytes))
			url := fmt.Sprintf("http://localhost:%s", port)
			fmt.Printf("Opening Web UI at %s...\n", url)
			exec.Command("open", url).Run()
		},
	}

	// ── stop ───────────────────────────────────────────────
	var stopCmd = &cobra.Command{
		Use:   "stop",
		Short: "Shutdown the Keith Gateway background daemon",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println("Sending termination signal to Keith Daemon...")

			homeDir, _ := os.UserHomeDir()
			plistPath := filepath.Join(homeDir, "Library", "LaunchAgents", "com.pranay0083.keith.plist")
			if _, err := os.Stat(plistPath); err == nil && runtime.GOOS == "darwin" {
				// Use modern bootout to fully de-register from launchd (prevents auto-revive)
				uidBytes, _ := exec.Command("id", "-u").Output()
				uid := strings.TrimSpace(string(uidBytes))
				exec.Command("launchctl", "bootout", fmt.Sprintf("gui/%s", uid), plistPath).Run()
			}

			// Fallback: Force kill any remaining processes
			err := exec.Command("pkill", "-9", "-f", "keith serve").Run()
			if err != nil {
				fmt.Println("No active Keith daemon found running in the background.")
			} else {
				fmt.Println("Keith Daemon successfully stopped.")
			}

			// Clean up stale port file
			os.Remove(filepath.Join(homeDir, ".config", "keith", "run.port"))
		},
	}

	chatCmd.Flags().StringVarP(&defaultModel, "model", "m", "", "Model to use (default depends on provider)")

	rootCmd.AddCommand(loginCmd, logoutCmd, accountsCmd, useCmd, modelsCmd, debugModelsCmd, chatCmd, startCmd, serveCmd, stopCmd, initCmd, webCmd)

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// loginOAuth handles the OAuth login flow for a given provider.
func loginOAuth(provider config.Provider) {
	tok, email, err := auth.RunOAuthFlow(provider)
	if err != nil {
		fmt.Printf("Login failed: %v\n", err)
		os.Exit(1)
	}

	acc := config.Account{
		Email:        email,
		Provider:     provider,
		RefreshToken: tok.RefreshToken,
		AccessToken:  tok.AccessToken,
		Expiry:       tok.Expiry.Unix(),
	}

	err = config.AddOrUpdateAccount(acc)
	if err != nil {
		fmt.Printf("Failed to save account: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Successfully logged in as %s [%s]!\n", email, provider)
}

// loginAPIKey handles API key login.
func loginAPIKey(reader *bufio.Reader) {
	fmt.Print("Enter your Gemini API Key: ")
	apiKey, _ := reader.ReadString('\n')
	apiKey = strings.TrimSpace(apiKey)

	if apiKey == "" {
		fmt.Println("API key cannot be empty.")
		os.Exit(1)
	}

	acc := config.Account{
		Email:    "apikey-user",
		Provider: config.ProviderAPIKey,
		APIKey:   apiKey,
	}

	err := config.AddOrUpdateAccount(acc)
	if err != nil {
		fmt.Printf("Failed to save API key: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Successfully saved Gemini API Key!")
}
