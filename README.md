# 👑 Keith - The Autonomous Developer Companion

**Keith** is a hyper-agentic, 24/7 terminal-native AI engineer. Built completely from the ground up to orchestrate complex RAG (Retrieval-Augmented Generation), control native macOS hardware, and write code entirely autonomously alongside you.

Keith lives natively in your CLI, detaches into an immortal background OS daemon, and interacts through a stunning, real-time React Web Deck. He learns your API keys, your aesthetic preferences, and your codebase autonomously.

---

## ⚡ Core Dependencies

To run Keith from source or build the binary, you need the following host dependencies:

1. **[Go](https://go.dev/doc/install)** (1.21+ recommended) - *Core CLI, Daemon Router, Gateway, and Agent Loop.*
2. **[Python 3](https://www.python.org/downloads/)** (`python3` + `pip3`) - *Zero-latency local Semantic Vector Engine.*
3. **macOS** (Apple Silicon or Intel) - *Required for native `launchctl` daemon persistence and hardware controls.*
4. **Gemini API Key** - *The primary cognitive LLM backend.*

*(Note: Keith autonomously downloads huggingface models and generates his own virtual environments on boot. No further Python setup is required).*

---

## 🚀 Quickstart & Setup

### 1. Build and Install Globally
Pull the repository and compile the native Unix binary into your `~/.local/bin`. Ensure `~/.local/bin` is in your `$PATH`.

```bash
git clone git.com:Pranay0083/Keith.git
cd keith
go build -o keith ./cmd/keith
cp keith ~/.local/bin/keith
```

### 2. Login & Authenticate
Securely bind your Gemini API Key into Keith's global config (`~/.config/keith/config.yaml`).
```bash
keith login
```

### 3. Initialize the RAG Identity Matrix
Scaffold Keith's initial identity blocks (`VISION.md`, `SOUL.md`, `MEMORY.md`, `USER.md`).
```bash
keith init
```

### 4. Spawn the Immortal Daemon
Start Keith. He will ask to install a true macOS `launchd` daemon so he survives reboots, sleep states, and panics perfectly.
```bash
keith start
```

### 5. Open the Web Interface
Once Keith is humming in the background, pop open the command deck in your native browser instantly.
```bash
keith web
```
*(You can also securely drop in a Telegram Bot Token via the Web UI chat to talk to Keith from your phone).*

---

## 🛑 Operations

To physically terminate the background Daemon process (useful for updates or heavy resets), run:
```bash
keith stop
```

---

## 🧠 Architecture Overview

- **`cmd/keith`**: The Cobra CLI router that maps User inputs into Go sub-functions.
- **`internal/agent`**: The terrifyingly potent Cognitive LLM Loop that continuously feeds Context, Tools, and Terminal output to Gemini to physically execute coding objectives.
- **`internal/memory`**: The Supermemory pipeline utilizing SQLite Full-Text Search and a purely local `sentence-transformers` Python REST API on a dynamic `40000+` port matrix.
- **`internal/system`**: Low-level macOS hardware bindings for Keyboard Brightness (`osascript`), System Volume, Wi-Fi, and Voice Synthesis (`say`).
- **`web/`**: The dynamic, glassmorphism-powered React Frontend UI bridging Server-Sent Events to raw LLM pipelines.

## License
MIT
