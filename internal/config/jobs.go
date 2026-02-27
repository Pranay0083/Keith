package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

type ScheduledJob struct {
	ID              string `json:"id"`
	Type            string `json:"type"` // "one-off" or "recurring"
	ExecuteAtUnix   int64  `json:"execute_at_unix"`
	IntervalMinutes int    `json:"interval_minutes"`
	LastExecuted    int64  `json:"last_executed"`
	TaskPrompt      string `json:"task_prompt"`
	ChatID          string `json:"chat_id"`
}

var jobsMutex sync.Mutex

func GetJobsFile() (string, error) {
	dir, err := GetConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "jobs.json"), nil
}

func LoadJobs() ([]ScheduledJob, error) {
	jobsMutex.Lock()
	defer jobsMutex.Unlock()

	path, err := GetJobsFile()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return []ScheduledJob{}, nil
	}
	if err != nil {
		return nil, err
	}

	var jobs []ScheduledJob
	err = json.Unmarshal(data, &jobs)
	if err != nil {
		return []ScheduledJob{}, err
	}
	return jobs, nil
}

func SaveJobs(jobs []ScheduledJob) error {
	jobsMutex.Lock()
	defer jobsMutex.Unlock()

	path, err := GetJobsFile()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(jobs, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}
