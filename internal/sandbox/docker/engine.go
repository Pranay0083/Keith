package docker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// Spec.Env carries the org's secrets, and their names are the org's
// choice. Passed as `docker run -e NAME` they would have to be in the
// docker CLI's own environment, where names like DOCKER_HOST,
// DOCKER_CONTEXT or HTTPS_PROXY redirect the CLI itself; passed as
// NAME=value they would show in the host's process list; and --env-file
// can't carry a multi-line value such as a PEM key. So Create sends the
// container config to the Engine API (POST /containers/create) over
// `docker system dial-stdio`, which connects to whatever daemon the CLI is
// set up for (DOCKER_HOST, the current context, ssh) and carries the
// request on stdin. The CLI runs with the worker's own environment only.

// createConfig is the body of POST /containers/create: the fields Keith
// sets.
type createConfig struct {
	Image        string
	Entrypoint   []string
	Cmd          []string
	Env          []string
	User         string `json:",omitempty"`
	WorkingDir   string
	Labels       map[string]string
	ExposedPorts map[string]struct{} `json:",omitempty"`
	HostConfig   hostConfig
}

type hostConfig struct {
	Init         bool
	NanoCpus     int64
	Memory       int64
	MemorySwap   int64
	PidsLimit    int64
	SecurityOpt  []string
	CapDrop      []string
	CapAdd       []string
	PortBindings map[string][]portBinding `json:",omitempty"`
	ExtraHosts   []string                 `json:",omitempty"`
}

type portBinding struct {
	HostIp   string //nolint:revive // the Engine API's field name
	HostPort string
}

// createConfig builds the container config for Create.
func (p *Provider) createConfig(image string, cpus, memMB int, spec sandbox.Spec) (*createConfig, error) {
	env := make([]string, 0, len(spec.Env))
	for _, k := range slices.Sorted(maps.Keys(spec.Env)) {
		if !envKeyRE.MatchString(k) {
			return nil, fmt.Errorf("docker: invalid environment variable name %q", k)
		}
		env = append(env, k+"="+spec.Env[k])
	}
	labels := map[string]string{}
	maps.Copy(labels, p.opts.Labels)
	maps.Copy(labels, spec.Labels)
	labels[LabelSandbox] = "1"
	cfg := &createConfig{
		Image:      image,
		Entrypoint: []string{"/bin/sh"},
		Cmd:        []string{"-c", idleScript},
		Env:        env,
		WorkingDir: sandbox.WorkspaceDir,
		Labels:     labels,
		HostConfig: hostConfig{
			Init:        true,
			NanoCpus:    int64(cpus) * 1e9,
			Memory:      int64(memMB) << 20,
			MemorySwap:  int64(memMB) << 20,
			PidsLimit:   DefaultPidsLimit,
			SecurityOpt: []string{"no-new-privileges:true"},
			CapDrop:     []string{"ALL"},
			CapAdd:      slices.Clone(rootCaps),
			ExtraHosts:  slices.Clone(p.opts.ExtraHosts),
		},
	}
	if p.opts.User != "-" {
		cfg.User = p.opts.User
	}
	for _, port := range p.opts.Ports {
		if cfg.ExposedPorts == nil {
			cfg.ExposedPorts = map[string]struct{}{}
			cfg.HostConfig.PortBindings = map[string][]portBinding{}
		}
		k := strconv.Itoa(port) + "/tcp"
		cfg.ExposedPorts[k] = struct{}{}
		cfg.HostConfig.PortBindings[k] = []portBinding{{HostIp: "127.0.0.1"}} // a random host port
	}
	return cfg, nil
}

// maxAPIResponse bounds an Engine API response body.
const maxAPIResponse = 1 << 20

// apiError is an Engine API call that returned an error status.
type apiError struct {
	status  int
	message string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("docker engine API: %d: %s", e.status, e.message)
}

// createContainer creates (but does not start) container name. When the
// image is missing and pull is true, it pulls the image (as `docker run`
// would) and tries once more.
func (p *Provider) createContainer(ctx context.Context, name string, cfg *createConfig, pull bool) error {
	err := p.apiCreate(ctx, name, cfg)
	var ae *apiError
	if !pull || !errors.As(err, &ae) || ae.status != http.StatusNotFound {
		return err
	}
	if _, perr := p.run(ctx, nil, "pull", cfg.Image); perr != nil {
		return fmt.Errorf("%w: image %s is not available locally, and pulling it failed: %w", sandbox.ErrNotFound, cfg.Image, perr)
	}
	return p.apiCreate(ctx, name, cfg)
}

func (p *Provider) apiCreate(ctx context.Context, name string, cfg *createConfig) error {
	status, body, err := p.engineAPI(ctx, http.MethodPost, "/containers/create?name="+url.QueryEscape(name), cfg)
	if err != nil {
		return err
	}
	if status == http.StatusCreated {
		return nil
	}
	var msg struct{ Message string }
	if json.Unmarshal(body, &msg) != nil || msg.Message == "" {
		msg.Message = strings.TrimSpace(string(body))
	}
	ae := &apiError{status: status, message: msg.Message}
	if status == http.StatusNotFound {
		return fmt.Errorf("%w (%w)", sandbox.ErrNotFound, ae)
	}
	return ae
}

// engineAPI sends one request to the Docker Engine API through
// `docker system dial-stdio` and returns the response's status and body.
// The body travels on the CLI's stdin: nothing in it reaches the CLI's
// argv or environment.
func (p *Provider) engineAPI(ctx context.Context, method, path string, body any) (int, []byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return 0, nil, fmt.Errorf("docker engine API: %w", err)
	}
	req, err := http.NewRequest(method, "http://docker"+path, bytes.NewReader(payload))
	if err != nil {
		return 0, nil, fmt.Errorf("docker engine API: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Close = true

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	cmd := exec.CommandContext(ctx, p.opts.Docker, "system", "dial-stdio")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return 0, nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, nil, err
	}
	stderr := sbutil.NewTailBuffer(maxCLIStderr)
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		return 0, nil, fmt.Errorf("docker system dial-stdio: %w", err)
	}
	defer func() {
		// The response is read: drop the connection. stdin stays open
		// until now because the daemon may cancel a request whose client
		// has half-closed.
		_ = stdin.Close()
		cancel()
		_ = cmd.Wait()
	}()
	fail := func(err error) (int, []byte, error) {
		if ctx.Err() != nil {
			return 0, nil, ctx.Err()
		}
		return 0, nil, fmt.Errorf("docker system dial-stdio: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if err := req.Write(stdin); err != nil {
		return fail(err)
	}
	resp, err := http.ReadResponse(bufio.NewReader(stdout), req)
	if err != nil {
		return fail(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxAPIResponse))
	if err != nil {
		return fail(err)
	}
	return resp.StatusCode, data, nil
}

// writeEnvFile writes env to a new private file in the docker CLI's
// --env-file format (one NAME=value per line) and returns its path; the
// caller removes it. The format can't carry a newline, a carriage return,
// a NUL or invalid UTF-8 in a value, so those are refused.
func writeEnvFile(env map[string]string) (string, error) {
	var b strings.Builder
	for _, k := range slices.Sorted(maps.Keys(env)) {
		v := env[k]
		if !envKeyRE.MatchString(k) {
			return "", fmt.Errorf("docker: invalid environment variable name %q", k)
		}
		if strings.ContainsAny(v, "\n\r\x00") || !utf8.ValidString(v) {
			return "", fmt.Errorf("docker: exec: the value of %s has a newline, carriage return, NUL or invalid UTF-8, which docker exec can't pass", k)
		}
		b.WriteString(k + "=" + v + "\n")
	}
	f, err := os.CreateTemp("", "keith-exec-env-*")
	if err != nil {
		return "", fmt.Errorf("docker: exec env file: %w", err)
	}
	_, werr := f.WriteString(b.String())
	if err := errors.Join(werr, f.Close()); err != nil {
		_ = os.Remove(f.Name())
		return "", fmt.Errorf("docker: exec env file: %w", err)
	}
	return f.Name(), nil
}
