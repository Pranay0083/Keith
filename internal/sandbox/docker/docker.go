// Package docker is a sandbox.Provider that runs each sandbox as a local
// Docker container, for local development and end-to-end tests. It drives
// the docker CLI with os/exec, so it needs `docker` on PATH and a running
// daemon, and nothing else.
//
// A sandbox is a long-lived container ("keith-sb-<random>") running an
// idle loop as the image's non-root user, with CPU, memory and PID limits,
// the label keith.sandbox=1 and a fixed set of container ports published
// on random 127.0.0.1 host ports (Options.Ports), which PortURL reports.
// Exec is `docker exec`; files move through exec with stdin and stdout, so
// they are owned by the sandbox user and large files stream. Pause and
// Resume are `docker pause`/`unpause` (the processes freeze and keep their
// memory), Snapshot is `docker commit`, Delete is `docker rm -f`.
//
// Not enforced locally: Spec.EgressAllowlist. Containers get Docker's
// default bridge network with unrestricted egress; Create logs a warning
// when an allowlist is given. Use a cloud provider for the hardened setup.
package docker

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"maps"
	"net"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// Name is the provider's name, matching config.SandboxProviderDocker.
const Name = "docker"

// Defaults for Options.
const (
	DefaultImage     = "keith-sandbox:latest"
	DefaultUser      = "agent"
	DefaultCPUs      = 2
	DefaultMemoryMB  = 4096
	DefaultPidsLimit = 4096
	namePrefix       = "keith-sb-"
	snapshotRepo     = "keith-snapshot"
)

// Labels set on every sandbox container and snapshot image.
const (
	LabelSandbox  = "keith.sandbox"
	LabelSnapshot = "keith.snapshot"
)

// DefaultPorts are the container ports published by default: the usual
// dev-server ports.
var DefaultPorts = []int{3000, 3001, 4000, 4200, 5000, 5173, 8000, 8080, 8888}

// Options configures the Provider. Zero values mean the defaults.
type Options struct {
	// Image is used when Spec.Template and Spec.SnapshotID are empty
	// (default DefaultImage, built from sandbox/base.Dockerfile).
	Image string
	// User is the container's default user (default DefaultUser). Set
	// "-" to use the image's own USER.
	User string
	// CPUs and MemoryMB apply when the Spec leaves them 0.
	CPUs     int
	MemoryMB int
	// Ports are the container ports published on random localhost ports
	// (default DefaultPorts). PortURL works only for these.
	Ports []int
	// Labels are added to every container and snapshot image (for
	// example to tag test runs for cleanup).
	Labels map[string]string
	// ExtraHosts are added to every container's /etc/hosts
	// (docker run --add-host), for example
	// "host.docker.internal:host-gateway" so a sandbox on Linux can reach
	// a server on the host.
	ExtraHosts []string
	// Docker is the docker CLI binary (default "docker" on PATH).
	Docker string
	// Logger receives warnings (default slog.Default()).
	Logger *slog.Logger
}

// Provider implements sandbox.Provider with local Docker containers.
type Provider struct {
	opts Options
	log  *slog.Logger
}

var _ sandbox.Provider = (*Provider)(nil)

// New returns a Provider. It does not contact the daemon.
func New(opts Options) *Provider {
	if opts.Image == "" {
		opts.Image = DefaultImage
	}
	if opts.User == "" {
		opts.User = DefaultUser
	}
	if opts.CPUs <= 0 {
		opts.CPUs = DefaultCPUs
	}
	if opts.MemoryMB <= 0 {
		opts.MemoryMB = DefaultMemoryMB
	}
	if opts.Ports == nil {
		opts.Ports = DefaultPorts
	}
	if opts.Docker == "" {
		opts.Docker = "docker"
	}
	opts.Labels = maps.Clone(opts.Labels)
	p := &Provider{opts: opts, log: opts.Logger}
	if p.log == nil {
		p.log = slog.Default()
	}
	return p
}

// Name implements sandbox.Provider.
func (p *Provider) Name() string { return Name }

// cliError is a failed docker CLI call.
type cliError struct {
	args   []string
	code   int
	stderr string
}

func (e *cliError) Error() string {
	return fmt.Sprintf("docker %s: exit %d: %s", e.args[0], e.code, e.stderr)
}

var notFoundRE = regexp.MustCompile(`(?i)no such (container|object|image)|unable to find image`)

// wrap maps a CLI error about a missing object to sandbox.ErrNotFound.
func wrap(err error) error {
	var ce *cliError
	if errors.As(err, &ce) && notFoundRE.MatchString(ce.stderr) {
		return fmt.Errorf("%w (%s)", sandbox.ErrNotFound, err)
	}
	return err
}

// run executes a docker CLI command and returns its stdout.
func (p *Provider) run(ctx context.Context, stdin io.Reader, env []string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, p.opts.Docker, args...)
	cmd.Stdin = stdin
	if env != nil {
		cmd.Env = append(os.Environ(), env...)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return stdout.Bytes(), &cliError{args: args, code: ee.ExitCode(), stderr: strings.TrimSpace(stderr.String())}
		}
		return nil, fmt.Errorf("docker %s: %w", args[0], err)
	}
	return stdout.Bytes(), nil
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

var envKeyRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// envArgs turns env into `-e KEY` flags plus KEY=VALUE entries for the
// CLI's own environment, so values (org secrets) never appear in argv.
func envArgs(env map[string]string) (args, vars []string, err error) {
	keys := slices.Sorted(maps.Keys(env))
	for _, k := range keys {
		if !envKeyRE.MatchString(k) {
			return nil, nil, fmt.Errorf("docker: invalid environment variable name %q", k)
		}
		args = append(args, "-e", k)
		vars = append(vars, k+"="+env[k])
	}
	return args, vars, nil
}

// idleScript keeps the container alive and exits promptly on SIGTERM.
const idleScript = `trap 'exit 0' TERM INT; while :; do sleep 3600 & wait $!; done`

// Create implements sandbox.Provider.
func (p *Provider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Sandbox, error) {
	image := p.opts.Image
	switch {
	case spec.SnapshotID != "":
		image = spec.SnapshotID
	case spec.Template != "":
		image = spec.Template
	}
	cpus, mem := spec.CPUs, spec.MemoryMB
	if cpus <= 0 {
		cpus = p.opts.CPUs
	}
	if mem <= 0 {
		mem = p.opts.MemoryMB
	}
	if len(spec.EgressAllowlist) > 0 {
		p.log.Warn("docker sandbox provider cannot enforce an egress allowlist; outbound traffic is unrestricted",
			"allowlist_hosts", len(spec.EgressAllowlist))
	}
	envFlags, envVars, err := envArgs(spec.Env)
	if err != nil {
		return nil, err
	}

	name := namePrefix + randomHex(6)
	args := []string{"run", "-d", "--name", name, "--init",
		"--label", LabelSandbox + "=1",
		"--workdir", sandbox.WorkspaceDir,
		"--cpus", strconv.Itoa(cpus),
		"--memory", strconv.Itoa(mem) + "m", "--memory-swap", strconv.Itoa(mem) + "m",
		"--pids-limit", strconv.Itoa(DefaultPidsLimit),
		"--security-opt", "no-new-privileges:true",
	}
	for _, k := range slices.Sorted(maps.Keys(p.opts.Labels)) {
		args = append(args, "--label", k+"="+p.opts.Labels[k])
	}
	for _, k := range slices.Sorted(maps.Keys(spec.Labels)) {
		args = append(args, "--label", k+"="+spec.Labels[k])
	}
	for _, h := range p.opts.ExtraHosts {
		args = append(args, "--add-host", h)
	}
	if p.opts.User != "-" {
		args = append(args, "--user", p.opts.User)
	}
	for _, port := range p.opts.Ports {
		args = append(args, "-p", "127.0.0.1::"+strconv.Itoa(port))
	}
	if spec.SnapshotID != "" {
		args = append(args, "--pull", "never") // a snapshot only exists locally
	}
	args = append(args, envFlags...)
	args = append(args, "--entrypoint", "/bin/sh", image, "-c", idleScript)

	if _, err := p.run(ctx, nil, envVars, args...); err != nil {
		// The container may exist even though it failed to start.
		_, _ = p.run(context.WithoutCancel(ctx), nil, nil, "rm", "-f", "-v", name)
		return nil, fmt.Errorf("docker: create sandbox from %s: %w", image, wrap(err))
	}
	return &Sandbox{p: p, id: name}, nil
}

// Get implements sandbox.Provider. Containers without the keith.sandbox
// label are not found.
func (p *Provider) Get(ctx context.Context, id string) (sandbox.Sandbox, error) {
	out, err := p.run(ctx, nil, nil, "inspect", "--type", "container", "--format",
		`{{.Name}} {{index .Config.Labels "`+LabelSandbox+`"}}`, id)
	if err != nil {
		return nil, fmt.Errorf("docker: get sandbox %s: %w", id, wrap(err))
	}
	name, label, _ := strings.Cut(strings.TrimSpace(string(out)), " ")
	if label != "1" {
		return nil, fmt.Errorf("docker: get sandbox %s: %w (not a keith sandbox)", id, sandbox.ErrNotFound)
	}
	return &Sandbox{p: p, id: strings.TrimPrefix(name, "/")}, nil
}

// List returns the IDs of sandboxes carrying all of labels (all
// sandboxes if labels is empty), for reaping orphans.
func (p *Provider) List(ctx context.Context, labels map[string]string) ([]string, error) {
	args := []string{"ps", "-a", "--filter", "label=" + LabelSandbox + "=1", "--format", "{{.Names}}"}
	for _, k := range slices.Sorted(maps.Keys(labels)) {
		args = append(args, "--filter", "label="+k+"="+labels[k])
	}
	out, err := p.run(ctx, nil, nil, args...)
	if err != nil {
		return nil, err
	}
	return strings.Fields(string(out)), nil
}

// DeleteSnapshot removes a snapshot image. Removing a missing one returns
// nil.
func (p *Provider) DeleteSnapshot(ctx context.Context, id string) error {
	_, err := p.run(ctx, nil, nil, "rmi", "-f", id)
	if err = wrap(err); errors.Is(err, sandbox.ErrNotFound) {
		return nil
	}
	return err
}

// Sandbox is one container.
type Sandbox struct {
	p  *Provider
	id string
}

var _ sandbox.Sandbox = (*Sandbox)(nil)

// ID implements sandbox.Sandbox: the container name.
func (s *Sandbox) ID() string { return s.id }

// pidPrefix starts the line the exec wrapper prints on stderr before
// running the command, so a timeout can kill the command's process group.
const pidPrefix = "__KEITH_PID="

// execWrapper records the process ID (docker exec processes lead their own
// session and process group), then replaces itself with `sh -c CMD`.
const execWrapper = `printf '` + pidPrefix + `%s\n' "$$" >&2; exec sh -c "$1"`

// Exec implements sandbox.Sandbox. On Timeout (or ctx cancellation) the
// command's whole process group is killed inside the container.
func (s *Sandbox) Exec(ctx context.Context, req sandbox.ExecRequest, onOutput func(chunk []byte)) (*sandbox.ExecResult, error) {
	dir := req.Dir
	if dir == "" {
		dir = sandbox.WorkspaceDir
	}
	envFlags, envVars, err := envArgs(req.Env)
	if err != nil {
		return nil, err
	}
	args := []string{"exec", "-w", dir}
	if req.User != "" {
		args = append(args, "-u", req.User)
	}
	args = append(args, envFlags...)
	args = append(args, s.id, "sh", "-c", execWrapper, "sh", req.Cmd)

	cmd := exec.Command(s.p.opts.Docker, args...)
	cmd.Env = append(os.Environ(), envVars...)

	// One goroutine delivers chunks to onOutput, in arrival order.
	var chunks chan []byte
	var delivered sync.WaitGroup
	if onOutput != nil {
		chunks = make(chan []byte, 64)
		delivered.Add(1)
		go func() {
			defer delivered.Done()
			for c := range chunks {
				onOutput(c)
			}
		}()
	}
	stdout := sbutil.NewTailBuffer(sandbox.MaxExecOutput)
	stderr := sbutil.NewTailBuffer(sandbox.MaxExecOutput)
	pidCh := make(chan int, 1)
	cmd.Stdout = &streamWriter{tail: stdout, chunks: chunks}
	cmd.Stderr = &pidWriter{w: &streamWriter{tail: stderr, chunks: chunks}, pid: pidCh}

	start := time.Now()
	if err := cmd.Start(); err != nil {
		if chunks != nil {
			close(chunks)
			delivered.Wait()
		}
		return nil, fmt.Errorf("docker exec: %w", err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	var timeout <-chan time.Time
	if req.Timeout > 0 {
		t := time.NewTimer(req.Timeout)
		defer t.Stop()
		timeout = t.C
	}
	var timedOut, canceled bool
	select {
	case err = <-done:
	case <-timeout:
		timedOut = true
	case <-ctx.Done():
		canceled = true
	}
	if timedOut || canceled {
		pid := 0
		select {
		case pid = <-pidCh:
		default:
		}
		if pid > 0 {
			kctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
			_, _ = s.p.run(kctx, nil, nil, "exec", "-u", "0", s.id, "kill", "-KILL", "--", "-"+strconv.Itoa(pid))
			cancel()
		}
		select {
		case err = <-done:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
			err = <-done
		}
	}
	if chunks != nil {
		close(chunks)
		delivered.Wait()
	}
	if canceled {
		return nil, ctx.Err()
	}

	res := &sandbox.ExecResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}
	if timedOut {
		res.ExitCode = -1
		res.TimedOut = true
		return res, nil
	}
	var ee *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &ee):
		res.ExitCode = ee.ExitCode()
		select {
		case <-pidCh:
			// The command ran: its exit status is the result.
		default:
			// The wrapper never ran: docker itself failed.
			return nil, fmt.Errorf("docker exec in %s: %w", s.id,
				wrap(&cliError{args: []string{"exec"}, code: res.ExitCode, stderr: strings.TrimSpace(res.Stderr)}))
		}
	default:
		return nil, fmt.Errorf("docker exec: %w", err)
	}
	return res, nil
}

// streamWriter keeps a tail and forwards chunks to the output channel.
type streamWriter struct {
	tail   *sbutil.TailBuffer
	chunks chan<- []byte
}

func (w *streamWriter) Write(p []byte) (int, error) {
	w.tail.Write(p)
	if w.chunks != nil && len(p) > 0 {
		w.chunks <- bytes.Clone(p)
	}
	return len(p), nil
}

// pidWriter strips the wrapper's PID line from the front of stderr and
// sends the PID on pid.
type pidWriter struct {
	w    io.Writer
	pid  chan<- int
	head []byte
	done bool
}

func (w *pidWriter) Write(p []byte) (int, error) {
	if w.done {
		return w.w.Write(p)
	}
	n := len(p)
	w.head = append(w.head, p...)
	i := bytes.IndexByte(w.head, '\n')
	if i < 0 {
		if len(w.head) > 64 { // not our line after all
			w.done = true
			_, err := w.w.Write(w.head)
			return n, err
		}
		return n, nil
	}
	w.done = true
	line, rest := w.head[:i], w.head[i+1:]
	if v, ok := bytes.CutPrefix(line, []byte(pidPrefix)); ok {
		if pid, err := strconv.Atoi(string(v)); err == nil {
			w.pid <- pid
		}
	} else {
		rest = w.head
	}
	if len(rest) > 0 {
		if _, err := w.w.Write(rest); err != nil {
			return n, err
		}
	}
	w.head = nil
	return n, nil
}

// fileScript exit codes.
const (
	exitNoFile = 44
	exitIsDir  = 45
)

func (s *Sandbox) fileOp(ctx context.Context, op, path string, stdin io.Reader, script string, extra ...string) ([]byte, error) {
	p := sbutil.ResolvePath(path)
	args := []string{"exec"}
	if stdin != nil {
		args = append(args, "-i")
	}
	args = append(args, s.id, "sh", "-c", script, "sh", p)
	args = append(args, extra...)
	out, err := s.p.run(ctx, stdin, nil, args...)
	var ce *cliError
	if errors.As(err, &ce) {
		switch ce.code {
		case exitNoFile:
			return nil, sbutil.NotExist(op, p)
		case exitIsDir:
			return nil, &fs.PathError{Op: op, Path: p, Err: errors.New("is a directory")}
		}
		return nil, &fs.PathError{Op: op, Path: p, Err: wrap(err)}
	}
	if err != nil {
		return nil, &fs.PathError{Op: op, Path: p, Err: err}
	}
	return out, nil
}

// ReadFile implements sandbox.Sandbox.
func (s *Sandbox) ReadFile(ctx context.Context, path string) ([]byte, error) {
	return s.fileOp(ctx, "read", path, nil, `[ -e "$1" ] || exit 44; [ -d "$1" ] && exit 45; exec cat -- "$1"`)
}

// WriteFile implements sandbox.Sandbox. Mode 0 means 0644.
func (s *Sandbox) WriteFile(ctx context.Context, path string, data []byte, mode fs.FileMode) error {
	if mode == 0 {
		mode = 0o644
	}
	_, err := s.fileOp(ctx, "write", path, bytes.NewReader(data),
		`[ -d "$1" ] && exit 45; mkdir -p -- "$(dirname -- "$1")" && cat > "$1" && chmod "$2" "$1"`,
		strconv.FormatUint(uint64(mode.Perm()), 8))
	return err
}

// RemoveFile implements sandbox.Sandbox. It does not remove directories.
func (s *Sandbox) RemoveFile(ctx context.Context, path string) error {
	_, err := s.fileOp(ctx, "remove", path, nil,
		`[ -e "$1" ] || [ -L "$1" ] || exit 44; [ -d "$1" ] && [ ! -L "$1" ] && exit 45; rm -f -- "$1"`)
	return err
}

// PortURL implements sandbox.Sandbox. Only ports in Options.Ports are
// published; others return an error wrapping sandbox.ErrNotSupported.
func (s *Sandbox) PortURL(ctx context.Context, port int) (string, error) {
	if port <= 0 || port > 65535 {
		return "", fmt.Errorf("docker: invalid port %d", port)
	}
	out, err := s.p.run(ctx, nil, nil, "port", s.id, strconv.Itoa(port)+"/tcp")
	if err != nil {
		var ce *cliError
		if errors.As(err, &ce) && strings.Contains(strings.ToLower(ce.stderr), "no public port") {
			return "", fmt.Errorf("docker: port %d is not published (published: %v): %w", port, s.p.opts.Ports, sandbox.ErrNotSupported)
		}
		return "", fmt.Errorf("docker: port URL: %w", wrap(err))
	}
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		host, hostPort, err := net.SplitHostPort(strings.TrimSpace(sc.Text()))
		if err != nil {
			continue
		}
		if ip := net.ParseIP(host); ip == nil || ip.IsUnspecified() {
			host = "127.0.0.1"
		}
		return "http://" + net.JoinHostPort(host, hostPort), nil
	}
	return "", fmt.Errorf("docker: port %d: no host binding in %q", port, out)
}

func (s *Sandbox) state(ctx context.Context) (string, error) {
	out, err := s.p.run(ctx, nil, nil, "inspect", "--type", "container", "--format", "{{.State.Status}}", s.id)
	if err != nil {
		return "", fmt.Errorf("docker: inspect %s: %w", s.id, wrap(err))
	}
	return strings.TrimSpace(string(out)), nil
}

// Pause implements sandbox.Sandbox with docker pause.
func (s *Sandbox) Pause(ctx context.Context) error {
	st, err := s.state(ctx)
	if err != nil || st == "paused" {
		return err
	}
	if _, err := s.p.run(ctx, nil, nil, "pause", s.id); err != nil {
		return fmt.Errorf("docker: pause %s: %w", s.id, wrap(err))
	}
	return nil
}

// Resume implements sandbox.Sandbox with docker unpause (or start, if the
// container was stopped, for example by a daemon restart).
func (s *Sandbox) Resume(ctx context.Context) error {
	st, err := s.state(ctx)
	if err != nil {
		return err
	}
	switch st {
	case "running":
		return nil
	case "paused":
		_, err = s.p.run(ctx, nil, nil, "unpause", s.id)
	default:
		_, err = s.p.run(ctx, nil, nil, "start", s.id)
	}
	if err != nil {
		return fmt.Errorf("docker: resume %s: %w", s.id, wrap(err))
	}
	return nil
}

var tagUnsafe = regexp.MustCompile(`[^a-z0-9_.-]+`)

// Snapshot implements sandbox.Sandbox with docker commit. The image is
// tagged keith-snapshot:<name>-<random> (so image pruning keeps it) and
// labeled keith.snapshot=1; its image ID is returned.
//
// docker commit copies the container's environment into the image, and
// Spec.Env carries the org's secrets. Every variable whose value differs
// from the container's image is therefore blanked in the snapshot
// (ENV NAME=), so secrets never land in a snapshot image; sandboxes
// started from it get their own Spec.Env.
func (s *Sandbox) Snapshot(ctx context.Context, name string) (string, error) {
	scrub, err := s.addedEnv(ctx)
	if err != nil {
		return "", fmt.Errorf("docker: snapshot %s: %w", s.id, err)
	}
	tag := strings.Trim(tagUnsafe.ReplaceAllString(strings.ToLower(name), "-"), "-.")
	if len(tag) > 100 {
		tag = tag[:100]
	}
	if tag == "" {
		tag = "snapshot"
	}
	args := []string{"commit", "--message", name, "--change", "LABEL " + LabelSnapshot + "=1"}
	for _, k := range scrub {
		args = append(args, "--change", "ENV "+k+"=")
	}
	for _, k := range slices.Sorted(maps.Keys(s.p.opts.Labels)) {
		args = append(args, "--change", "LABEL "+strconv.Quote(k)+"="+strconv.Quote(s.p.opts.Labels[k]))
	}
	args = append(args, s.id, snapshotRepo+":"+tag+"-"+randomHex(3))
	out, err := s.p.run(ctx, nil, nil, args...)
	if err != nil {
		return "", fmt.Errorf("docker: snapshot %s: %w", s.id, wrap(err))
	}
	return strings.TrimSpace(string(out)), nil
}

// addedEnv returns the names of the container's environment variables
// that its image does not set to the same value: the Spec.Env entries.
func (s *Sandbox) addedEnv(ctx context.Context) ([]string, error) {
	out, err := s.p.run(ctx, nil, nil, "inspect", "--type", "container", "--format", `{{.Image}}`, s.id)
	if err != nil {
		return nil, wrap(err)
	}
	image := strings.TrimSpace(string(out))
	var cenv, ienv []string
	for _, x := range []struct {
		kind, id string
		dst      *[]string
	}{{"container", s.id, &cenv}, {"image", image, &ienv}} {
		out, err := s.p.run(ctx, nil, nil, "inspect", "--type", x.kind, "--format", `{{json .Config.Env}}`, x.id)
		if err != nil {
			return nil, wrap(err)
		}
		if err := json.Unmarshal(bytes.TrimSpace(out), x.dst); err != nil {
			return nil, fmt.Errorf("inspect %s env: %w", x.kind, err)
		}
	}
	var added []string
	for _, kv := range cenv {
		k, _, _ := strings.Cut(kv, "=")
		if !slices.Contains(ienv, kv) && envKeyRE.MatchString(k) {
			added = append(added, k)
		}
	}
	slices.Sort(added)
	return slices.Compact(added), nil
}

// Delete implements sandbox.Sandbox with docker rm -f. Deleting a missing
// container returns nil.
func (s *Sandbox) Delete(ctx context.Context) error {
	_, err := s.p.run(ctx, nil, nil, "rm", "-f", "-v", s.id)
	if err = wrap(err); err != nil && !errors.Is(err, sandbox.ErrNotFound) {
		return fmt.Errorf("docker: delete %s: %w", s.id, err)
	}
	return nil
}
