package memory

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Supported text and source code extensions
var supportedExtensions = map[string]bool{
	".go":    true,
	".py":    true,
	".js":    true,
	".ts":    true,
	".tsx":   true,
	".jsx":   true,
	".md":    true,
	".html":  true,
	".css":   true,
	".json":  true,
	".cpp":   true,
	".c":     true,
	".h":     true,
	".txt":   true,
	".sh":    true,
	".rs":    true,
	".mod":   true,
	".sum":   true,
	".swift": true,
	".zig":   true,
}

var ignoredDirectories = map[string]bool{
	".git":         true,
	"node_modules": true,
	"venv":         true,
	"env":          true,
	"__pycache__":  true,
	"dist":         true,
	"build":        true,
	"vendor":       true,
	".vscode":      true,
	".idea":        true,
	"testdata":     true,
}

// IngestDirectory recursively walks a directory, chunks supported files, and vectorizes them into SQLite.
func IngestDirectory(rootPath string) (int, error) {
	absPath, err := filepath.Abs(rootPath)
	if err != nil {
		return 0, fmt.Errorf("failed to resolve absolute path: %w", err)
	}

	info, err := os.Stat(absPath)
	if err != nil || !info.IsDir() {
		return 0, fmt.Errorf("invalid directory path: %s", absPath)
	}

	filePaths := []string{}
	err = filepath.WalkDir(absPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip unreadable paths
		}

		if d.IsDir() {
			if ignoredDirectories[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if supportedExtensions[ext] {
			filePaths = append(filePaths, path)
		}
		return nil
	})

	if err != nil {
		return 0, fmt.Errorf("error walking directory: %w", err)
	}

	totalChunks := 0
	for _, fp := range filePaths {
		chunks, err := chunkFile(fp)
		if err != nil {
			fmt.Printf("Warning: failed to chunk file %s: %v\n", fp, err)
			continue
		}

		for i, chunkText := range chunks {
			err := saveCodeChunk(fp, i, chunkText)
			if err != nil {
				fmt.Printf("Warning: failed to vector-index chunk %d of %s: %v\n", i, fp, err)
				continue
			}
			totalChunks++
		}
	}

	if totalChunks > 0 {
		if err := TrackWorkspace(absPath); err != nil {
			fmt.Printf("Warning: failed to track workspace path: %v\n", err)
		}
	}

	return totalChunks, nil
}

// IngestDirectoryIncremental only re-chunks files modified after the given cutoff time.
// This makes workspace updates dramatically faster for large codebases.
func IngestDirectoryIncremental(rootPath string, since time.Time) (int, error) {
	absPath, err := filepath.Abs(rootPath)
	if err != nil {
		return 0, fmt.Errorf("failed to resolve absolute path: %w", err)
	}

	info, err := os.Stat(absPath)
	if err != nil || !info.IsDir() {
		return 0, fmt.Errorf("invalid directory path: %s", absPath)
	}

	filePaths := []string{}
	err = filepath.WalkDir(absPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if ignoredDirectories[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !supportedExtensions[ext] {
			return nil
		}
		// Only include files modified after the cutoff
		if fi, err := d.Info(); err == nil && fi.ModTime().After(since) {
			filePaths = append(filePaths, path)
		}
		return nil
	})
	if err != nil {
		return 0, fmt.Errorf("error walking directory: %w", err)
	}

	totalChunks := 0
	for _, fp := range filePaths {
		// Delete old chunks for this specific file before re-ingesting
		if db != nil {
			db.Exec("DELETE FROM ingested_files WHERE filepath = ?", fp)
		}

		chunks, err := chunkFile(fp)
		if err != nil {
			fmt.Printf("Warning: failed to chunk file %s: %v\n", fp, err)
			continue
		}
		for i, chunkText := range chunks {
			if err := saveCodeChunk(fp, i, chunkText); err != nil {
				fmt.Printf("Warning: failed to vector-index chunk %d of %s: %v\n", i, fp, err)
				continue
			}
			totalChunks++
		}
	}

	// Update the workspace timestamp
	if totalChunks > 0 || len(filePaths) == 0 {
		TrackWorkspace(absPath)
	}

	return totalChunks, nil
}

// Optimized for all-MiniLM 384-dimensional matrices (approx 500 chars).
func chunkFile(filePath string) ([]string, error) {
	bytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	content := string(bytes)
	if len(strings.TrimSpace(content)) == 0 {
		return nil, nil
	}

	const chunkSize = 500
	const overlap = 150

	var chunks []string
	contentLen := len(content)

	for i := 0; i < contentLen; i += (chunkSize - overlap) {
		end := i + chunkSize
		if end > contentLen {
			end = contentLen
		}
		chunks = append(chunks, content[i:end])
		if end == contentLen {
			break
		}
	}

	return chunks, nil
}
