package memory

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	_ "modernc.org/sqlite"
)

var db *sql.DB

func InitDB() error {
	cfgDir, err := config.GetConfigDir()
	if err != nil {
		return err
	}

	dbPath := filepath.Join(cfgDir, "memory.db")
	db, err = sql.Open("sqlite", dbPath)
	if err != nil {
		return fmt.Errorf("failed to open sqlite database: %w", err)
	}

	// Create physical facts table
	createFactsTable := `
	CREATE TABLE IF NOT EXISTS facts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		content TEXT NOT NULL,
		embedding BLOB,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`
	if _, err := db.Exec(createFactsTable); err != nil {
		return fmt.Errorf("failed to create facts table: %w", err)
	}

	// Make sure embedding column exists (skip if it does)
	db.Exec("ALTER TABLE facts ADD COLUMN embedding BLOB;")

	// Phase 4A: Codebase RAG Table
	createRAGTable := `
	CREATE TABLE IF NOT EXISTS ingested_files (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		filepath TEXT NOT NULL,
		chunk_index INTEGER NOT NULL,
		content TEXT NOT NULL,
		embedding BLOB NOT NULL,
		ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`
	if _, err := db.Exec(createRAGTable); err != nil {
		return fmt.Errorf("failed to create ingested_files table: %w", err)
	}

	// Phase 4A (Hardening): Track explicitly vectorized workspaces
	createWorkspacesTable := `
	CREATE TABLE IF NOT EXISTS ingested_workspaces (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		path TEXT UNIQUE NOT NULL,
		ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`
	if _, err := db.Exec(createWorkspacesTable); err != nil {
		return fmt.Errorf("failed to create ingested_workspaces table: %w", err)
	}

	return nil
}

// TrackWorkspace saves the bounding root folder to inform the LLM of its RAG availability
func TrackWorkspace(path string) error {
	if db == nil {
		return fmt.Errorf("database not initialized")
	}
	_, err := db.Exec("INSERT OR IGNORE INTO ingested_workspaces (path, ingested_at) VALUES (?, ?)", path, time.Now())
	return err
}

// GetIngestedWorkspaces returns a list of all actively RAG-indexed folders
func GetIngestedWorkspaces() ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database not initialized")
	}
	rows, err := db.Query("SELECT path FROM ingested_workspaces")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var workspaces []string
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err == nil {
			workspaces = append(workspaces, path)
		}
	}
	return workspaces, nil
}

// saveCodeChunk handles persisting a file chunk into the Workspace Vector database
func saveCodeChunk(filePath string, chunkIndex int, content string) error {
	if db == nil {
		return fmt.Errorf("database not initialized")
	}

	embedding, err := GetLocalEmbedding(content)
	if err != nil {
		return fmt.Errorf("local ml engine vectorization failed for chunk: %w", err)
	}

	embBytes, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("failed to serialize chunk embedding: %w", err)
	}

	_, err = db.Exec("INSERT INTO ingested_files (filepath, chunk_index, content, embedding, ingested_at) VALUES (?, ?, ?, ?, ?)", filePath, chunkIndex, content, embBytes, time.Now())
	if err != nil {
		return fmt.Errorf("error inserting ingested workspace vector: %w", err)
	}
	return nil
}

// SearchCodebase hits the RAG Workspace Knowledge Graph natively
func SearchCodebase(query string, limit int) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database not initialized")
	}

	queryEmb, err := GetLocalEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize codebase query: %w", err)
	}

	rows, err := db.Query("SELECT filepath, chunk_index, content, embedding FROM ingested_files")
	if err != nil {
		return nil, fmt.Errorf("failed to query codebase chunks: %w", err)
	}
	defer rows.Close()

	type scoredChunk struct {
		filepath string
		chunkIdx int
		content  string
		score    float64
	}
	var scoredChunks []scoredChunk

	for rows.Next() {
		var filepath string
		var chunkIdx int
		var content string
		var embBytes []byte
		if err := rows.Scan(&filepath, &chunkIdx, &content, &embBytes); err != nil {
			continue
		}

		var factEmb []float32
		if err := json.Unmarshal(embBytes, &factEmb); err != nil {
			continue
		}

		// Local Matrix Mathematics
		score := cosineSimilarity(queryEmb, factEmb)

		// For Codebase RAG, we don't use absolute thresholds because English questions
		// ("What does this do?") naturally have wide vector distances from raw Go/Py syntax.
		// We simply store all scores and bubble sort the strongest matches!
		scoredChunks = append(scoredChunks, scoredChunk{
			filepath: filepath,
			chunkIdx: chunkIdx,
			content:  content,
			score:    score,
		})
	}

	// Bubble up the strongest mathematical alignments seamlessly
	sort.Slice(scoredChunks, func(i, j int) bool {
		return scoredChunks[i].score > scoredChunks[j].score
	})

	var results []string
	for i, sc := range scoredChunks {
		if i >= limit {
			break
		}
		formatted := fmt.Sprintf("[%s (Chunk %d)]: %s", sc.filepath, sc.chunkIdx, sc.content)
		results = append(results, formatted)
	}

	return results, nil
}

func cosineSimilarity(a, b []float32) float64 {
	var dotProduct, normA, normB float64
	for i := 0; i < len(a) && i < len(b); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func SaveFact(content string) error {
	if db == nil {
		return fmt.Errorf("database not initialized")
	}

	// Generate 384-dimensional mathematical Vector locally on CPU
	embedding, err := GetLocalEmbedding(content)
	if err != nil {
		return fmt.Errorf("local ml engine vectorization failed: %w", err)
	}

	embBytes, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	_, err = db.Exec("INSERT INTO facts (content, embedding, created_at) VALUES (?, ?, ?)", content, embBytes, time.Now())
	if err != nil {
		return fmt.Errorf("error inserting vector fact: %w", err)
	}
	return nil
}

func SearchFacts(query string) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database not initialized")
	}

	// Mathematically align user conversational query internally via Flask server
	queryEmb, err := GetLocalEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize query: %w", err)
	}

	rows, err := db.Query("SELECT content, embedding FROM facts WHERE embedding IS NOT NULL")
	if err != nil {
		return nil, fmt.Errorf("failed to query facts: %w", err)
	}
	defer rows.Close()

	type scoredFact struct {
		content string
		score   float64
	}
	var scoredFacts []scoredFact

	for rows.Next() {
		var content string
		var embBytes []byte
		if err := rows.Scan(&content, &embBytes); err != nil {
			continue
		}

		var factEmb []float32
		if err := json.Unmarshal(embBytes, &factEmb); err != nil {
			continue
		}

		// Compute semantic mathematical Cosine Similarity dynamically
		score := cosineSimilarity(queryEmb, factEmb)

		// L6 model operates best at a tighter tolerance threshold
		if score > 0.35 {
			scoredFacts = append(scoredFacts, scoredFact{content: content, score: score})
		}
	}

	// Sort dynamically by semantic alignment
	sort.Slice(scoredFacts, func(i, j int) bool {
		return scoredFacts[i].score > scoredFacts[j].score
	})

	var results []string
	for i, sf := range scoredFacts {
		if i >= 5 {
			break
		}
		results = append(results, sf.content)
	}

	return results, nil
}

// Fact representations for the REST API
type Fact struct {
	ID        int       `json:"id"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

// GetAllFacts dumps the conversational Supermemory straight to the frontend bypassing ML matrices
func GetAllFacts() ([]Fact, error) {
	if db == nil {
		return nil, fmt.Errorf("database not initialized")
	}

	rows, err := db.Query("SELECT id, content, created_at FROM facts ORDER BY created_at DESC")
	if err != nil {
		return nil, fmt.Errorf("failed to load facts from supermemory: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		var f Fact
		if err := rows.Scan(&f.ID, &f.Content, &f.CreatedAt); err == nil {
			facts = append(facts, f)
		}
	}
	return facts, nil
}

// DeleteFact allows manual pruning of the factual Graph
func DeleteFact(id int) error {
	if db == nil {
		return fmt.Errorf("database not initialized")
	}
	_, err := db.Exec("DELETE FROM facts WHERE id = ?", id)
	return err
}

func Close() error {
	if db != nil {
		return db.Close()
	}
	return nil
}
