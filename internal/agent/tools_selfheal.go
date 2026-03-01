package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/Pranay0083/keith/internal/memory"
)

// --- Reboot ML Engine Tool ---
type RebootMLEngineTool struct{}

func (r *RebootMLEngineTool) Name() string { return "reboot_ml_engine" }
func (r *RebootMLEngineTool) Description() string {
	return "Reboot the local Python Vector ML Engine. Use this when embedding calls are failing or the engine is unresponsive. No arguments needed."
}
func (r *RebootMLEngineTool) Execute(args map[string]string) (string, error) {
	fmt.Println("[Self-Heal] Rebooting ML Engine...")
	memory.StopMLEngine()
	time.Sleep(2 * time.Second)
	if err := memory.StartMLEngine(); err != nil {
		return fmt.Sprintf("ML Engine restart failed: %v", err), err
	}
	return "ML Engine successfully rebooted and is now operational.", nil
}

// --- Reboot Self Tool ---
type RebootSelfTool struct{}

func (r *RebootSelfTool) Name() string { return "reboot_self" }
func (r *RebootSelfTool) Description() string {
	return "Initiate a full graceful restart of the Keith Daemon. This will cleanly shutdown everything and relaunch. Use as a last resort when Keith is experiencing critical failures. No arguments needed."
}
func (r *RebootSelfTool) Execute(args map[string]string) (string, error) {
	fmt.Println("[Self-Heal] Initiating full daemon reboot...")
	exe, err := os.Executable()
	if err != nil {
		return "Cannot resolve own executable path for reboot.", err
	}

	// Spawn new instance before dying
	cmd := exec.Command(exe, "serve")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Sprintf("Failed to spawn new instance: %v", err), err
	}

	// Gracefully tear down the current instance
	go func() {
		time.Sleep(2 * time.Second)
		memory.StopMLEngine()
		memory.Close()
		os.Exit(0)
	}()

	return fmt.Sprintf("Reboot initiated. New process spawned with PID %d. Current instance will terminate in 2 seconds.", cmd.Process.Pid), nil
}

// --- Diagnose Health Tool ---
type DiagnoseHealthTool struct {
	startTime time.Time
}

func NewDiagnoseHealthTool() *DiagnoseHealthTool {
	return &DiagnoseHealthTool{startTime: time.Now()}
}

func (d *DiagnoseHealthTool) Name() string { return "diagnose_health" }
func (d *DiagnoseHealthTool) Description() string {
	return "Returns a comprehensive JSON health report of Keith's subsystems: ML engine status, memory usage, uptime, ports, OS info. No arguments needed."
}
func (d *DiagnoseHealthTool) Execute(args map[string]string) (string, error) {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	mlPort := memory.GetMLPort()
	mlStatus := "DOWN"
	if mlPort > 0 {
		mlStatus = fmt.Sprintf("UP (port %d)", mlPort)
	}

	report := map[string]interface{}{
		"uptime_seconds":   int(time.Since(d.startTime).Seconds()),
		"uptime_human":     time.Since(d.startTime).Round(time.Second).String(),
		"ml_engine_status": mlStatus,
		"ml_engine_port":   mlPort,
		"pid":              os.Getpid(),
		"go_version":       runtime.Version(),
		"os":               runtime.GOOS,
		"arch":             runtime.GOARCH,
		"num_goroutines":   runtime.NumGoroutine(),
		"memory_alloc_mb":  fmt.Sprintf("%.2f", float64(memStats.Alloc)/1024/1024),
		"memory_sys_mb":    fmt.Sprintf("%.2f", float64(memStats.Sys)/1024/1024),
		"num_gc_cycles":    memStats.NumGC,
	}

	jsonBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "Failed to generate health report.", err
	}
	return string(jsonBytes), nil
}
