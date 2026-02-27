package gateway

import (
	"fmt"
	"strconv"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/models"
	"github.com/Pranay0083/keith/internal/token"
)

// StartChronoEngine permanently runs as a background Goroutine ticker, monitoring config/jobs.json
func StartChronoEngine(bot *TelegramBot) {
	fmt.Println("🕰️  Chrono-Matrix Engine initialized. Sweeping background schedules...")
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		<-ticker.C

		jobs, err := config.LoadJobs()
		if err != nil || len(jobs) == 0 {
			continue
		}

		now := time.Now().Unix()
		var updatedJobs []config.ScheduledJob
		var jobsChanged bool

		for _, job := range jobs {
			trigger := false

			if job.Type == "one-off" {
				if now >= job.ExecuteAtUnix {
					trigger = true
				} else {
					updatedJobs = append(updatedJobs, job) // keep it in the matrix
				}
			} else if job.Type == "recurring" {
				if now >= job.LastExecuted+(int64(job.IntervalMinutes)*60) {
					trigger = true
					job.LastExecuted = now
				}
				updatedJobs = append(updatedJobs, job) // retain in database
			} else {
				// Corrupted job type, drop it
				jobsChanged = true
			}

			if trigger {
				jobsChanged = true
				fmt.Printf("🕰️  [CHRONO-TRIGGER] Routing autonomous execution matrix for task: %s\n", job.ID)

				// Spawn asynchronously to avoid blocking the sweeping loop
				go func(triggeredJob config.ScheduledJob) {
					provider, err := token.GetActiveProvider()
					if err != nil {
						fmt.Printf("Chrono auth error: %v\n", err)
						return
					}

					model := models.DefaultModel(provider)
					accs, confErr := config.LoadAccounts()
					if confErr == nil && accs.DefaultModel != "" {
						model = accs.DefaultModel
					}

					chatID, _ := strconv.ParseInt(triggeredJob.ChatID, 10, 64)
					agentSession := bot.GetSession(chatID)
					sysMsg := fmt.Sprintf("SYSTEM CHRONO-TRIGGER: You scheduled this autonomous task to execute right now. Execute it immediately without asking for user permission: '%s'", triggeredJob.TaskPrompt)

					resp, err := agentSession.ProcessMessage(sysMsg, model, func(progress string) {
						bot.sendMessage(chatID, fmt.Sprintf("> %s", progress))
					})

					if err == nil {
						if len(resp) > 4000 {
							for i := 0; i < len(resp); i += 4000 {
								end := i + 4000
								if end > len(resp) {
									end = len(resp)
								}
								bot.sendMessage(chatID, resp[i:end])
							}
						} else {
							bot.sendMessage(chatID, resp)
						}
					}
				}(job)
			}
		}

		if jobsChanged {
			config.SaveJobs(updatedJobs)
		}
	}
}
