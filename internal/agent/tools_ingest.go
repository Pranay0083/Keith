package agent

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/memory"
)

// Shared state for background ingestion tracking
var (
	ingestionStatus = make(map[string]string) // path -> "running" | "done: N chunks" | "error: ..."
	ingestionMu     sync.Mutex
)

// GetIngestionStatus returns the current status of a background ingestion job.
func GetIngestionStatus(path string) string {
	ingestionMu.Lock()
	defer ingestionMu.Unlock()
	if s, ok := ingestionStatus[path]; ok {
		return s
	}
	return "not_started"
}

// --- Ingest Codebase Tool ---
type IngestCodebaseTool struct{}

func (t *IngestCodebaseTool) Name() string { return "ingest_codebase" }
func (t *IngestCodebaseTool) Description() string {
	return "Ingest and vectorize an entire codebase directory into Keith's Supermemory RAG. This runs in the background for large repos. Arguments: 'path' (absolute path to directory). After ingesting, you can use search_codebase to find relevant code."
}
func (t *IngestCodebaseTool) Execute(args map[string]string) (string, error) {
	path, ok := args["path"]
	if !ok || path == "" {
		return "", fmt.Errorf("missing 'path' argument")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}

	// Check if already running
	ingestionMu.Lock()
	if ingestionStatus[absPath] == "running" {
		ingestionMu.Unlock()
		return fmt.Sprintf("Ingestion for %s is already running in the background. Use search_codebase once it completes.", absPath), nil
	}
	ingestionStatus[absPath] = "running"
	ingestionMu.Unlock()

	// Run ingestion in background goroutine
	go func() {
		startTime := time.Now()
		totalChunks, err := memory.IngestDirectory(absPath)
		ingestionMu.Lock()
		defer ingestionMu.Unlock()
		if err != nil {
			ingestionStatus[absPath] = fmt.Sprintf("error: %v", err)
			fmt.Printf("[Ingest] Background ingestion failed for %s: %v\n", absPath, err)
		} else {
			elapsed := time.Since(startTime).Round(time.Second)
			ingestionStatus[absPath] = fmt.Sprintf("done: %d chunks in %s", totalChunks, elapsed)
			fmt.Printf("[Ingest] Background ingestion completed for %s: %d chunks in %s\n", absPath, totalChunks, elapsed)
		}
	}()

	return fmt.Sprintf("Ingestion started for %s in the background. This may take several minutes for large codebases. You can continue working — use search_codebase once ingestion completes. Check status with diagnose_health.", absPath), nil
}

// --- Update Codebase Tool ---
type UpdateCodebaseTool struct{}

func (t *UpdateCodebaseTool) Name() string { return "update_codebase" }
func (t *UpdateCodebaseTool) Description() string {
	return "Incrementally update an already-ingested codebase. Only re-indexes files that changed since last ingest, making it fast. Arguments: 'path' (absolute path to previously ingested directory)."
}
func (t *UpdateCodebaseTool) Execute(args map[string]string) (string, error) {
	path, ok := args["path"]
	if !ok || path == "" {
		return "", fmt.Errorf("missing 'path' argument")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}

	// Get last ingest time
	lastIngest, err := memory.GetWorkspaceIngestTime(absPath)
	if err != nil {
		return fmt.Sprintf("Workspace %s has not been ingested yet. Use ingest_codebase first.", absPath), nil
	}

	// Check if already running
	ingestionMu.Lock()
	if ingestionStatus[absPath] == "running" {
		ingestionMu.Unlock()
		return fmt.Sprintf("An ingestion job for %s is already running.", absPath), nil
	}
	ingestionStatus[absPath] = "running"
	ingestionMu.Unlock()

	// Run incremental update in background
	go func() {
		startTime := time.Now()
		totalChunks, err := memory.IngestDirectoryIncremental(absPath, lastIngest)
		ingestionMu.Lock()
		defer ingestionMu.Unlock()
		if err != nil {
			ingestionStatus[absPath] = fmt.Sprintf("error: %v", err)
			fmt.Printf("[Ingest] Incremental update failed for %s: %v\n", absPath, err)
		} else {
			elapsed := time.Since(startTime).Round(time.Second)
			ingestionStatus[absPath] = fmt.Sprintf("updated: %d chunks in %s", totalChunks, elapsed)
			fmt.Printf("[Ingest] Incremental update completed for %s: %d new/updated chunks in %s\n", absPath, totalChunks, elapsed)
		}
	}()

	return fmt.Sprintf("Incremental update started for %s in background. Only files modified after %s will be re-indexed.", absPath, lastIngest.Format(time.RFC3339)), nil
}

// --- Clean Codebase Tool ---
type CleanCodebaseTool struct{}

func (t *CleanCodebaseTool) Name() string { return "clean_codebase" }
func (t *CleanCodebaseTool) Description() string {
	return "Remove a previously ingested codebase from Keith's Supermemory. Deletes all vectorized chunks and workspace tracking. Arguments: 'path' (absolute path to the workspace to clean)."
}
func (t *CleanCodebaseTool) Execute(args map[string]string) (string, error) {
	path, ok := args["path"]
	if !ok || path == "" {
		return "", fmt.Errorf("missing 'path' argument")
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}

	deletedChunks, err := memory.DeleteChunksForWorkspace(absPath)
	if err != nil {
		return "", fmt.Errorf("failed to delete chunks: %w", err)
	}

	if err := memory.RemoveWorkspace(absPath); err != nil {
		return "", fmt.Errorf("failed to remove workspace tracking: %w", err)
	}

	// Clear ingestion status
	ingestionMu.Lock()
	delete(ingestionStatus, absPath)
	ingestionMu.Unlock()

	return fmt.Sprintf("Successfully cleaned workspace %s. Removed %d vectorized chunks and workspace tracking entry.", absPath, deletedChunks), nil
}
