package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/providers"
	"github.com/Pranay0083/keith/internal/sandbox/tmuxshell"
)

// benchOptions configures one sandbox-bench run.
type benchOptions struct {
	N        int
	Parallel int
	Template string
	CPUs     int
	MemoryMB int
	Shells   bool
	JSON     bool
	Labels   map[string]string
}

// Bench phases, in the order they run.
const (
	phaseCreate      = "create"
	phaseFirstExec   = "first_exec"
	phaseShellCreate = "shell_create"
	phaseShellExec   = "shell_exec"
	phasePause       = "pause"
	phaseResume      = "resume"
	phaseExecResumed = "exec_after_resume"
	phaseDelete      = "delete"
)

func (o benchOptions) phases() []string {
	ph := []string{phaseCreate, phaseFirstExec}
	if o.Shells {
		ph = append(ph, phaseShellCreate, phaseShellExec)
	}
	return append(ph, phasePause, phaseResume, phaseExecResumed, phaseDelete)
}

type benchRun struct {
	Run       int                      `json:"run"`
	SandboxID string                   `json:"sandbox_id,omitempty"`
	Phases    map[string]time.Duration `json:"-"`
	PhasesMS  map[string]float64       `json:"phases_ms"`
	Error     string                   `json:"error,omitempty"`
}

type benchStat struct {
	N      int     `json:"n"`
	MinMS  float64 `json:"min_ms"`
	MedMS  float64 `json:"median_ms"`
	MeanMS float64 `json:"mean_ms"`
	MaxMS  float64 `json:"max_ms"`
}

type benchReport struct {
	Provider string               `json:"provider"`
	Template string               `json:"template,omitempty"`
	N        int                  `json:"n"`
	Parallel int                  `json:"parallel"`
	WallMS   float64              `json:"wall_ms"`
	Runs     []benchRun           `json:"runs"`
	Summary  map[string]benchStat `json:"summary"`
}

func newSandboxBenchCmd() *cobra.Command {
	var o benchOptions
	cmd := &cobra.Command{
		Use:   "sandbox-bench",
		Short: "Time sandbox cold start, first exec, pause, resume and delete",
		Long: `sandbox-bench creates N sandboxes on the configured provider (SANDBOX_PROVIDER,
plus DAYTONA_API_KEY/DAYTONA_API_URL for daytona) and reports how long each
step takes: create (cold start), first exec, pause, resume, an exec after
resume, and delete. With --shells it also times creating a tmux shell and
running a command in it. Every sandbox is deleted at the end, including on
Ctrl-C, and carries the label keith.bench=1.

Use it for the provider spike: run the same template on each provider and
compare the summaries (--json for machine-readable output).`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}
			p, err := providers.New(cfg)
			if err != nil {
				return err
			}
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			return runSandboxBench(ctx, cmd.OutOrStdout(), cmd.ErrOrStderr(), p, o)
		},
	}
	f := cmd.Flags()
	f.IntVarP(&o.N, "n", "n", 3, "number of sandboxes to create")
	f.IntVarP(&o.Parallel, "parallel", "p", 1, "how many sandboxes to run at once")
	f.StringVar(&o.Template, "template", "", "image or provider template (default: the provider's base image)")
	f.IntVar(&o.CPUs, "cpus", 0, "vCPUs per sandbox (0: provider default)")
	f.IntVar(&o.MemoryMB, "memory-mb", 0, "memory per sandbox in MB (0: provider default)")
	f.BoolVar(&o.Shells, "shells", false, "also time a tmux shell (needs tmux and bash in the image)")
	f.BoolVar(&o.JSON, "json", false, "print the report as JSON")
	f.StringToStringVar(&o.Labels, "label", nil, "extra sandbox label key=value (repeatable)")
	return cmd
}

// runSandboxBench runs the benchmark and writes the report to out and
// progress to progress.
func runSandboxBench(ctx context.Context, out, progress io.Writer, p sandbox.Provider, o benchOptions) error {
	if o.N < 1 {
		return errors.New("sandbox-bench: --n must be at least 1")
	}
	o.Parallel = min(max(o.Parallel, 1), o.N)

	runs := make([]benchRun, o.N)
	sem := make(chan struct{}, o.Parallel)
	var wg sync.WaitGroup
	var logMu sync.Mutex
	start := time.Now()
	for i := range runs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				runs[i] = benchRun{Run: i + 1, Error: ctx.Err().Error()}
				return
			}
			defer func() { <-sem }()
			runs[i] = benchOne(ctx, p, o, i+1)
			logMu.Lock()
			if r := runs[i]; r.Error != "" {
				fmt.Fprintf(progress, "run %d: %s: error: %s\n", r.Run, r.SandboxID, r.Error)
			} else {
				fmt.Fprintf(progress, "run %d: %s: create %v\n", r.Run, r.SandboxID, r.Phases[phaseCreate].Round(time.Millisecond))
			}
			logMu.Unlock()
		}()
	}
	wg.Wait()

	rep := benchReport{
		Provider: p.Name(),
		Template: o.Template,
		N:        o.N,
		Parallel: o.Parallel,
		WallMS:   ms(time.Since(start)),
		Runs:     runs,
		Summary:  map[string]benchStat{},
	}
	for _, ph := range o.phases() {
		var ds []time.Duration
		for _, r := range runs {
			if d, ok := r.Phases[ph]; ok {
				ds = append(ds, d)
			}
		}
		if len(ds) > 0 {
			rep.Summary[ph] = stat(ds)
		}
	}
	for i := range rep.Runs {
		rep.Runs[i].PhasesMS = map[string]float64{}
		for ph, d := range rep.Runs[i].Phases {
			rep.Runs[i].PhasesMS[ph] = ms(d)
		}
	}

	if o.JSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			return err
		}
	} else {
		writeBenchTable(out, rep, o.phases())
	}

	var failed int
	for _, r := range runs {
		if r.Error != "" {
			failed++
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if failed > 0 {
		return fmt.Errorf("sandbox-bench: %d of %d runs failed", failed, o.N)
	}
	return nil
}

type benchStep struct {
	phase string
	fn    func() error
}

// benchOne times one sandbox's lifecycle. The sandbox is always deleted.
func benchOne(ctx context.Context, p sandbox.Provider, o benchOptions, n int) (r benchRun) {
	r = benchRun{Run: n, Phases: map[string]time.Duration{}}
	timed := func(phase string, fn func() error) error {
		t := time.Now()
		err := fn()
		if err != nil {
			return fmt.Errorf("%s: %w", phase, err)
		}
		r.Phases[phase] = time.Since(t)
		return nil
	}
	execOK := func(sb sandbox.Sandbox) func() error {
		return func() error {
			res, err := sb.Exec(ctx, sandbox.ExecRequest{Cmd: "echo ok", Timeout: 2 * time.Minute}, nil)
			if err != nil {
				return err
			}
			if res.ExitCode != 0 || strings.TrimSpace(res.Stdout) != "ok" {
				return fmt.Errorf("exit %d, stdout %q, stderr %q", res.ExitCode, res.Stdout, res.Stderr)
			}
			return nil
		}
	}

	var sb sandbox.Sandbox
	err := timed(phaseCreate, func() error {
		var err error
		labels := maps.Clone(o.Labels)
		if labels == nil {
			labels = map[string]string{}
		}
		labels["keith.bench"] = "1"
		labels["keith.bench.run"] = fmt.Sprint(n)
		sb, err = p.Create(ctx, sandbox.Spec{Template: o.Template, CPUs: o.CPUs, MemoryMB: o.MemoryMB, Labels: labels})
		return err
	})
	if err != nil {
		r.Error = err.Error()
		return r
	}
	r.SandboxID = sb.ID()
	defer func() {
		// Delete even after a failure or Ctrl-C.
		dctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Minute)
		defer cancel()
		t := time.Now()
		if err := sb.Delete(dctx); err != nil {
			if r.Error == "" {
				r.Error = "delete: " + err.Error()
			}
			return
		}
		if r.Error == "" {
			r.Phases[phaseDelete] = time.Since(t)
		}
	}()

	steps := []benchStep{{phaseFirstExec, execOK(sb)}}
	if o.Shells {
		sh := tmuxshell.New(sb)
		steps = append(steps,
			benchStep{phaseShellCreate, func() error { return sh.Create(ctx, "bench") }},
			benchStep{phaseShellExec, func() error {
				res, err := sh.Exec(ctx, "bench", "echo ok", time.Minute)
				if err != nil {
					return err
				}
				if res.ExitCode != 0 || res.OutputTail != "ok\n" {
					return fmt.Errorf("exit %d, output %q", res.ExitCode, res.OutputTail)
				}
				return nil
			}})
	}
	steps = append(steps,
		benchStep{phasePause, func() error { return sb.Pause(ctx) }},
		benchStep{phaseResume, func() error { return sb.Resume(ctx) }},
		benchStep{phaseExecResumed, execOK(sb)})
	for _, s := range steps {
		if err := timed(s.phase, s.fn); err != nil {
			r.Error = err.Error()
			return r
		}
	}
	return r
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func stat(ds []time.Duration) benchStat {
	slices.Sort(ds)
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	med := ds[len(ds)/2]
	if len(ds)%2 == 0 {
		med = (ds[len(ds)/2-1] + ds[len(ds)/2]) / 2
	}
	return benchStat{
		N:      len(ds),
		MinMS:  ms(ds[0]),
		MedMS:  ms(med),
		MeanMS: ms(sum / time.Duration(len(ds))),
		MaxMS:  ms(ds[len(ds)-1]),
	}
}

func writeBenchTable(out io.Writer, rep benchReport, phases []string) {
	tmpl := rep.Template
	if tmpl == "" {
		tmpl = "(provider default)"
	}
	fmt.Fprintf(out, "provider %s, template %s, %d sandboxes, %d at a time, wall %s\n\n",
		rep.Provider, tmpl, rep.N, rep.Parallel, time.Duration(rep.WallMS*float64(time.Millisecond)).Round(time.Millisecond))
	tw := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "run\tsandbox\t%s\terror\n", strings.Join(phases, "\t"))
	for _, r := range rep.Runs {
		cells := []string{fmt.Sprint(r.Run), r.SandboxID}
		for _, ph := range phases {
			if d, ok := r.Phases[ph]; ok {
				cells = append(cells, d.Round(time.Millisecond).String())
			} else {
				cells = append(cells, "-")
			}
		}
		cells = append(cells, r.Error)
		fmt.Fprintln(tw, strings.Join(cells, "\t"))
	}
	tw.Flush()

	fmt.Fprintln(out)
	tw = tabwriter.NewWriter(out, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "phase\tn\tmin\tmedian\tmean\tmax\t")
	for _, ph := range phases {
		s, ok := rep.Summary[ph]
		if !ok {
			continue
		}
		f := func(v float64) string {
			return time.Duration(v * float64(time.Millisecond)).Round(time.Millisecond).String()
		}
		fmt.Fprintf(tw, "%s\t%d\t%s\t%s\t%s\t%s\t\n", ph, s.N, f(s.MinMS), f(s.MedMS), f(s.MeanMS), f(s.MaxMS))
	}
	tw.Flush()
}
