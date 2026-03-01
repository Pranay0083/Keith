package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Pranay0083/keith/internal/memory"
)

// StartWatchdog launches a background goroutine that continuously monitors
// Keith's vital subsystems and auto-heals them when they fail.
func StartWatchdog(gatewayPort int) {
	fmt.Println("[Watchdog] Self-healing monitor activated.")

	go func() {
		client := &http.Client{Timeout: 5 * time.Second}
		mlFailCount := 0
		const maxMLFailures = 3

		for {
			time.Sleep(60 * time.Second)

			// 1. ML Engine Health Check
			mlPort := memory.GetMLPort()
			if mlPort > 0 {
				resp, err := client.Post(
					fmt.Sprintf("http://127.0.0.1:%d/embed", mlPort),
					"application/json",
					strings.NewReader(`{"text":"watchdog_ping"}`),
				)
				if err != nil || resp.StatusCode != http.StatusOK {
					mlFailCount++
					fmt.Printf("[Watchdog] ML Engine health check failed (%d/%d)\n", mlFailCount, maxMLFailures)
					if resp != nil {
						resp.Body.Close()
					}
					if mlFailCount >= maxMLFailures {
						fmt.Println("[Watchdog] ML Engine unresponsive. Initiating auto-restart...")
						memory.StopMLEngine()
						time.Sleep(2 * time.Second)
						if err := memory.StartMLEngine(); err != nil {
							fmt.Printf("[Watchdog] ML Engine restart failed: %v\n", err)
						} else {
							fmt.Println("[Watchdog] ML Engine successfully restarted!")
						}
						mlFailCount = 0
					}
				} else {
					resp.Body.Close()
					mlFailCount = 0
				}
			}

			// 2. Gateway Self-Ping
			resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/", gatewayPort))
			if err != nil {
				fmt.Printf("[Watchdog] Gateway self-ping failed: %v\n", err)
			} else {
				resp.Body.Close()
			}
		}
	}()
}
