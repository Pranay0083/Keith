package agent

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strconv"
	"time"

	"github.com/Pranay0083/keith/internal/config"
)

func generateJobID() string {
	b := make([]byte, 4)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// CreateScheduledTaskTool securely injects a task into the Go Chrono runtime
type CreateScheduledTaskTool struct{}

func (s *CreateScheduledTaskTool) Name() string { return "create_scheduled_task" }
func (s *CreateScheduledTaskTool) Description() string {
	return "Schedules a native task for you to execute autonomously in the future. " +
		"Arguments:\n" +
		"- 'type': MUST be either 'one-off' or 'recurring'\n" +
		"- 'execute_at_unix': (For one-off only) The exact Unix Epoch Timestamp (seconds) when this should execute. Calculate this mathematically using the Current System Time provided in your prompt.\n" +
		"- 'interval_minutes': (For recurring only) The delay in minutes between executions (e.g., 15 for every 15 minutes).\n" +
		"- 'task_prompt': Your exact instructions for what to do when this triggers (e.g., 'Message the user and say drink water').\n" +
		"- 'chat_id': The destination Telegram Chat ID to target."
}
func (s *CreateScheduledTaskTool) Execute(args map[string]string) (string, error) {
	jobType, ok := args["type"]
	if !ok || (jobType != "one-off" && jobType != "recurring") {
		return "", fmt.Errorf("invalid 'type': must be 'one-off' or 'recurring'")
	}

	taskPrompt, ok := args["task_prompt"]
	if !ok || taskPrompt == "" {
		return "", fmt.Errorf("missing 'task_prompt' argument")
	}

	chatID, ok := args["chat_id"]
	if !ok || chatID == "" {
		return "", fmt.Errorf("missing 'chat_id' argument")
	}

	job := config.ScheduledJob{
		ID:         "job_" + generateJobID(),
		Type:       jobType,
		TaskPrompt: taskPrompt,
		ChatID:     chatID,
	}

	if jobType == "one-off" {
		unixStr, _ := args["execute_at_unix"]
		unixTime, err := strconv.ParseInt(unixStr, 10, 64)
		if err != nil || unixTime < time.Now().Unix() {
			return "", fmt.Errorf("invalid 'execute_at_unix': must be a valid future Unix timestamp")
		}
		job.ExecuteAtUnix = unixTime
	} else {
		intStr, _ := args["interval_minutes"]
		interval, err := strconv.Atoi(intStr)
		if err != nil || interval <= 0 {
			return "", fmt.Errorf("invalid 'interval_minutes': must be a positive integer")
		}
		job.IntervalMinutes = interval
		job.LastExecuted = time.Now().Unix() // Start the timer now
	}

	jobs, _ := config.LoadJobs()
	jobs = append(jobs, job)
	err := config.SaveJobs(jobs)
	if err != nil {
		return "", fmt.Errorf("failed to save job to matrix: %v", err)
	}

	return fmt.Sprintf("Chrono Job [%s] successfully scheduled securely in the daemon database.", job.ID), nil
}

// ListScheduledTasksTool allows Keith to inspect his timeline
type ListScheduledTasksTool struct{}

func (s *ListScheduledTasksTool) Name() string { return "list_scheduled_tasks" }
func (s *ListScheduledTasksTool) Description() string {
	return "Returns an array of all active scheduled jobs currently running in the background daemon."
}
func (s *ListScheduledTasksTool) Execute(args map[string]string) (string, error) {
	jobs, err := config.LoadJobs()
	if err != nil {
		return "", err
	}
	if len(jobs) == 0 {
		return "Timeline empty. No scheduled jobs found.", nil
	}

	output := "Active Chrono-Matrix Jobs:\n"
	for _, j := range jobs {
		output += fmt.Sprintf("- ID: %s | Type: %s | Prompt: %s\n", j.ID, j.Type, j.TaskPrompt)
	}
	return output, nil
}

// DeleteScheduledTaskTool grants Keith authority to cancel future events
type DeleteScheduledTaskTool struct{}

func (s *DeleteScheduledTaskTool) Name() string { return "delete_scheduled_task" }
func (s *DeleteScheduledTaskTool) Description() string {
	return "Permanently cancels and deletes an active scheduled job. Arguments: 'id' (the job ID)."
}
func (s *DeleteScheduledTaskTool) Execute(args map[string]string) (string, error) {
	targetID, ok := args["id"]
	if !ok || targetID == "" {
		return "", fmt.Errorf("missing 'id' argument")
	}

	jobs, err := config.LoadJobs()
	if err != nil {
		return "", err
	}

	var newJobs []config.ScheduledJob
	found := false
	for _, j := range jobs {
		if j.ID == targetID {
			found = true
			continue
		}
		newJobs = append(newJobs, j)
	}

	if !found {
		return fmt.Sprintf("Job ID %s not found in timeline.", targetID), nil
	}

	config.SaveJobs(newJobs)
	return fmt.Sprintf("Eradicated Job ID %s from the Chrono-Matrix.", targetID), nil
}
