package memory

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/config"
)

var embedderCmd *exec.Cmd
var mlPort int

const pythonServerScript = `
import os
import sys
from flask import Flask, request, jsonify

# Prevent huggingface from dialing home tracking
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("FATAL: sentence-transformers not installed")
    sys.exit(1)

app = Flask(__name__)

# Load the lightning-fast L6 model directly into memory
print("Loading all-MiniLM-L6-v2 directly into RAM...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Ready. Booting localhost server on port 8046.")

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text'}), 400
    
    # Generate the 384-dimensional mathematical vector instantaneously
    embedding = model.encode(data['text'])
    
    # Convert numpy array to standard python list for JSON serialization
    return jsonify({'embedding': embedding.tolist()})

if __name__ == '__main__':
    # Run the internal matrix math server tightly locked to localhost
    app.run(host='127.0.0.1', port=%d)
`

// StartMLEngine orchestrates the local zero-dependency Python Vector daemon.
func StartMLEngine() error {
	fmt.Println("[Phase 3D] Orchestrating Local Python Vector Embedder...")

	cfgDir, err := config.GetConfigDir()
	if err != nil {
		return err
	}

	embedderDir := filepath.Join(cfgDir, "embedder")
	if err := os.MkdirAll(embedderDir, 0755); err != nil {
		return err
	}

	// 1. Acquire dynamic OS execution port (40000+)
	mlPort = 40000
	for p := 40000; p < 41000; p++ {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", p))
		if err == nil {
			ln.Close()
			mlPort = p
			break
		}
	}

	// 2. Write the Python logic dynamically to disk with the injected port
	serverPath := filepath.Join(embedderDir, "server.py")
	scriptPayload := fmt.Sprintf(pythonServerScript, mlPort)
	if err := os.WriteFile(serverPath, []byte(scriptPayload), 0644); err != nil {
		return err
	}

	// 2. Locate the system Python interpreter
	pythonBin := "python3"
	if runtime.GOOS == "windows" {
		pythonBin = "python"
	}

	venvDir := filepath.Join(embedderDir, "venv")

	// Create Virtual Environment if it doesn't exist
	if _, err := os.Stat(venvDir); os.IsNotExist(err) {
		fmt.Println("[Phase 3D] Initializing isolated Python venv for Semantic Search...")
		cmd := exec.Command(pythonBin, "-m", "venv", venvDir)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to create python venv (is python installed?): %v", err)
		}
	}

	// Determine isolated pip/python binaries
	pipBin := filepath.Join(venvDir, "bin", "pip")
	pyBin := filepath.Join(venvDir, "bin", "python")
	if runtime.GOOS == "windows" {
		pipBin = filepath.Join(venvDir, "Scripts", "pip.exe")
		pyBin = filepath.Join(venvDir, "Scripts", "python.exe")
	}

	// 3. Ensure ML dependencies are correctly installed
	fmt.Println("[Phase 3D] Validating ML dependencies (`sentence-transformers`, `flask`)...")
	pipCheck := exec.Command(pipBin, "show", "sentence-transformers")
	if err := pipCheck.Run(); err != nil {
		fmt.Println("[Phase 3D] Downloading huggingface packages (first boot only - this takes a minute)...")
		installCmd := exec.Command(pipBin, "install", "sentence-transformers", "flask")
		installCmd.Stdout = os.Stdout
		installCmd.Stderr = os.Stderr
		if err := installCmd.Run(); err != nil {
			return fmt.Errorf("failed to install pip dependencies: %v", err)
		}
	}

	// 4. Spawn the Python Subprocess Daemon
	fmt.Println("[Phase 3D] Spawning Vector Matrix Engine...")
	embedderCmd = exec.Command(pyBin, serverPath)
	// We map the Python engine's outputs to Go so the user can see model loading
	embedderCmd.Stdout = os.Stdout
	embedderCmd.Stderr = os.Stderr

	if err := embedderCmd.Start(); err != nil {
		return fmt.Errorf("failed to start python embedder: %v", err)
	}

	// 5. Robust polling until the dynamic port is responding to HTTP requests
	client := &http.Client{Timeout: 1 * time.Second}
	ready := false
	for i := 0; i < 30; i++ {
		resp, err := client.Post(fmt.Sprintf("http://127.0.0.1:%d/embed", mlPort), "application/json", strings.NewReader(`{"text":"warmup"}`))
		if err == nil {
			resp.Body.Close()
			ready = true
			break
		}
		time.Sleep(1 * time.Second)
	}

	if !ready {
		StopMLEngine()
		return fmt.Errorf("python vector engine failed to bind to port %d in time", mlPort)
	}

	fmt.Printf("[Phase 3D] Local ML Vector Embedded Graph initialized on localhost:%d!\n", mlPort)
	return nil
}

// StopMLEngine strictly reaps the Python subprocess to prevent zombie daemons.
func StopMLEngine() {
	if embedderCmd != nil && embedderCmd.Process != nil {
		fmt.Println("[Phase 3D] Terminating Python ML Subprocess...")
		embedderCmd.Process.Kill()
		embedderCmd = nil
	}
}

// GetLocalEmbedding hits the pure-local ML pipeline to vectorize semantic intent.
func GetLocalEmbedding(text string) ([]float32, error) {
	reqBody := map[string]string{"text": text}
	jsonVal, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request payload: %w", err)
	}

	resp, err := http.Post(fmt.Sprintf("http://127.0.0.1:%d/embed", mlPort), "application/json", bytes.NewBuffer(jsonVal))
	if err != nil {
		return nil, fmt.Errorf("python ML engine unreachable: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding error %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Embedding, nil
}
