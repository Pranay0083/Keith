// Package daytona implements sandbox.Provider and sandbox.Sandbox on
// Daytona (https://daytona.io) with the official Go SDK
// (github.com/daytona/clients/sdk-go). No SDK type leaves this package.
//
// # Sandbox source
//
// Create picks what to boot from in this order:
//
//  1. Spec.SnapshotID: a Daytona snapshot name or ID (what Sandbox.Snapshot
//     returns).
//  2. Spec.Template, read with this rule:
//     - "snapshot:<name>" is the Daytona snapshot <name>;
//     - "image:<ref>" is the container image <ref>;
//     - otherwise a value containing '/', ':' or '@' is an image reference
//     (for example "node:22" or "ghcr.io/org/img@sha256:..."), and anything
//     else is a Daytona snapshot name. Use the "snapshot:" prefix for a
//     snapshot whose name looks like an image (Daytona's own default
//     snapshots are named like "daytonaio/sandbox:0.4.3").
//  3. Options.DefaultSnapshot, always a snapshot name.
//  4. Nothing: Daytona's default snapshot for the organization.
//
// An image is built by Daytona ("FROM <ref>") on first use, which can take
// minutes; a snapshot boots in seconds.
//
// # Resources
//
// Spec.CPUs and Spec.MemoryMB apply only to image-based sandboxes: the SDK
// sends resources only with an image, and a snapshot carries its own CPU,
// memory and disk. Daytona sizes memory in whole GB, so MemoryMB is rounded
// up (1500 MB becomes 2 GB).
//
// # Lifecycle
//
// Every sandbox gets the labels Options.Labels, then Spec.Labels, then
// keith.sandbox=1 (later wins). Auto-stop defaults to 15 minutes of
// inactivity, auto-archive to 7 days after stopping, and auto-delete is left
// to Daytona (never) unless Options.AutoDeleteInterval is set. Daytona
// counts these intervals in whole minutes, so durations are rounded up.
// Pause stops the sandbox (Daytona "stop": compute released, disk kept) and
// Resume starts it, including from Daytona's archived and paused states.
//
// # User
//
// The sandbox runs as the snapshot's or image's default user (Daytona's
// "daytona" user in its default snapshots); Create does not override it.
// Exec with ExecRequest.User set to a different user wraps the command in
// `sudo -n -E -H -u <user>`, which works where that user exists and the
// default user has passwordless sudo (true of Daytona's default snapshots);
// elsewhere the command exits non-zero with sudo's message on stderr.
//
// # Network
//
// Spec.EgressAllowlist entries that are IP addresses or CIDR blocks go to
// Daytona's networkAllowList; host names (wildcards such as "*.npmjs.org"
// are passed through) go to its domainAllowList. Schemes, paths and ports
// are stripped ("https://github.com:443/x" becomes "github.com"). How
// Daytona enforces each list, including for traffic that is not HTTP(S),
// is Daytona's policy. An empty list leaves Daytona's default (open egress).
//
// # Exec
//
// Exec runs each command in a fresh Daytona process session as
// `sh -c '<cmd>' < /dev/null` (after `cd` into the working directory and
// with ExecRequest.Env set), streams stdout and stderr over the toolbox's
// log WebSocket, and reads the exit code from the session. If the stream is
// unavailable the full output is fetched when the command ends and passed
// to onOutput in one piece. A timed-out command is stopped by deleting its
// session. Sessions are deleted after every command, so the per-command env
// does not linger in the daemon's session history.
//
// # Preview URLs
//
// PortURL returns a signed preview URL (the access token is embedded in the
// URL) valid for Options.PreviewURLTTL, default one hour. If Daytona cannot
// sign one, it falls back to the plain preview URL, which for a private
// sandbox needs the X-Daytona-Preview-Token header and is logged as a
// warning.
//
// # Snapshot
//
// Snapshot uses Daytona's "create snapshot from sandbox" and returns the
// new snapshot's name once Daytona reports it active. Daytona requires a
// linux-vm class sandbox to be stopped for a disk snapshot, so such a
// sandbox is stopped, snapshotted and started again (running processes are
// lost); a container class sandbox is snapshotted in place.
//
// # Concurrency
//
// The SDK is configured to observe state changes by polling instead of its
// event WebSocket, so no background goroutine mutates a sandbox's SDK state
// and the provider can read it safely. Pause, Resume and Snapshot on one
// sandbox are serialized. They can rewrite the SDK's toolbox endpoint when
// Daytona moves the sandbox, so do not run them while Exec or a file
// operation on the same sandbox is in flight.
package daytona

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	sdkerrors "github.com/daytona/clients/sdk-go/pkg/errors"
	sdkoptions "github.com/daytona/clients/sdk-go/pkg/options"
	sdktypes "github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/Pranay0083/keith/internal/sandbox"
)

// Name is the provider's stable ID.
const Name = "daytona"

// LabelSandbox is set to "1" on every sandbox this provider creates, so
// orphans can be found by label.
const LabelSandbox = "keith.sandbox"

const (
	defaultAutoStop    = 15 * time.Minute
	defaultAutoArchive = 7 * 24 * time.Hour
	defaultPreviewTTL  = time.Hour

	// createTimeout bounds creating a sandbox and waiting for it to start;
	// building an image can take minutes.
	createTimeout   = 10 * time.Minute
	startTimeout    = 10 * time.Minute
	stopTimeout     = 5 * time.Minute
	snapshotTimeout = 30 * time.Minute
	// fileOpTimeout bounds one file transfer when ctx has no deadline.
	fileOpTimeout = 15 * time.Minute
	// cleanupTimeout bounds best-effort deletes after a failure.
	cleanupTimeout = time.Minute
	// responseHeaderTimeout bounds how long the default HTTP client waits
	// for a response once a request is sent. There is deliberately no
	// overall client timeout: it would cut off large file transfers.
	responseHeaderTimeout = 2 * time.Minute
)

// Options configures the provider.
type Options struct {
	// APIKey is a Daytona API key. Required.
	APIKey string
	// APIURL is the Daytona API base URL; empty uses the SDK default
	// (https://app.daytona.io/api, or DAYTONA_API_URL).
	APIURL string
	// Target is the Daytona region; empty uses the organization default.
	Target string
	// DefaultSnapshot is the Daytona snapshot name used when the Spec names
	// neither a snapshot nor a template. Empty uses Daytona's default.
	DefaultSnapshot string

	// AutoStopInterval stops an idle sandbox after this long. 0 means 15
	// minutes; negative disables auto-stop.
	AutoStopInterval time.Duration
	// AutoArchiveInterval archives a stopped sandbox after this long. 0
	// means 7 days; negative means Daytona's maximum.
	AutoArchiveInterval time.Duration
	// AutoDeleteInterval deletes a stopped sandbox after this long. 0
	// leaves Daytona's default (never); negative disables it explicitly.
	AutoDeleteInterval time.Duration

	// Labels are added to every sandbox's labels.
	Labels map[string]string
	// Logger receives provider logs; nil means slog.Default(). Env values
	// and commands are never logged.
	Logger *slog.Logger
	// HTTPClient is used for all Daytona API and toolbox HTTP requests (the
	// SDK copies it). Nil uses a client with no overall timeout and a
	// two-minute response-header timeout.
	HTTPClient *http.Client
	// PreviewURLTTL is how long a URL from PortURL stays valid. 0 means one
	// hour.
	PreviewURLTTL time.Duration
}

// Provider creates and finds Daytona sandboxes. It is safe for concurrent
// use.
type Provider struct {
	client *sdk.Client
	log    *slog.Logger

	labels          map[string]string
	defaultSnapshot string
	autoStop        int  // minutes; 0 disables
	autoArchive     int  // minutes; 0 means Daytona's maximum
	autoDelete      *int // minutes; nil leaves Daytona's default
	previewTTL      time.Duration
}

var _ sandbox.Provider = (*Provider)(nil)

// New returns a Provider. It does not contact Daytona.
func New(opts Options) (*Provider, error) {
	if strings.TrimSpace(opts.APIKey) == "" {
		return nil, errors.New("daytona: APIKey is required")
	}
	log := opts.Logger
	if log == nil {
		log = slog.Default()
	}
	hc := opts.HTTPClient
	if hc == nil {
		hc = defaultHTTPClient()
	}
	polling := true
	client, err := sdk.NewClientWithConfig(&sdktypes.DaytonaConfig{
		APIKey:     opts.APIKey,
		APIUrl:     opts.APIURL,
		Target:     opts.Target,
		HTTPClient: hc,
		// Polling keeps the SDK's per-sandbox state free of writes from the
		// event WebSocket goroutine, which the provider would race with when
		// it reads a sandbox's state. See the package doc.
		UseDeprecatedPolling: &polling, //nolint:staticcheck // intentional, see above
	})
	if err != nil {
		return nil, fmt.Errorf("daytona: new client: %w", err)
	}

	p := &Provider{
		client:          client,
		log:             log,
		labels:          maps.Clone(opts.Labels),
		defaultSnapshot: strings.TrimSpace(opts.DefaultSnapshot),
		previewTTL:      opts.PreviewURLTTL,
	}
	if p.previewTTL <= 0 {
		p.previewTTL = defaultPreviewTTL
	}

	switch {
	case opts.AutoStopInterval == 0:
		p.autoStop = ceilMinutes(defaultAutoStop)
	case opts.AutoStopInterval < 0:
		p.autoStop = 0
	default:
		p.autoStop = ceilMinutes(opts.AutoStopInterval)
	}
	switch {
	case opts.AutoArchiveInterval == 0:
		p.autoArchive = ceilMinutes(defaultAutoArchive)
	case opts.AutoArchiveInterval < 0:
		p.autoArchive = 0
	default:
		p.autoArchive = ceilMinutes(opts.AutoArchiveInterval)
	}
	switch {
	case opts.AutoDeleteInterval < 0:
		p.autoDelete = intPtr(-1)
	case opts.AutoDeleteInterval > 0:
		p.autoDelete = intPtr(ceilMinutes(opts.AutoDeleteInterval))
	}
	return p, nil
}

// Name returns "daytona".
func (p *Provider) Name() string { return Name }

// Create provisions a sandbox, waits for it to start and makes sure
// sandbox.WorkspaceDir exists and is writable by the default user. A
// sandbox that fails after Daytona created it is deleted.
func (p *Provider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Sandbox, error) {
	base := sdktypes.SandboxBaseParams{
		EnvVars:             maps.Clone(spec.Env),
		Labels:              p.sandboxLabels(spec.Labels),
		AutoStopInterval:    intPtr(p.autoStop),
		AutoArchiveInterval: intPtr(p.autoArchive),
	}
	if p.autoDelete != nil {
		base.AutoDeleteInterval = intPtr(*p.autoDelete)
	}
	if cidrs, domains := splitEgress(spec.EgressAllowlist); len(cidrs)+len(domains) > 0 {
		if len(cidrs) > 0 {
			base.NetworkAllowList = strPtr(strings.Join(cidrs, ","))
		}
		if len(domains) > 0 {
			base.DomainAllowList = strPtr(strings.Join(domains, ","))
		}
		p.log.Debug("daytona: egress allow-list", "cidrs", cidrs, "domains", domains)
	}

	params, source := p.createParams(spec, base)
	start := time.Now()
	sb, err := p.client.Create(ctx, params,
		sdkoptions.WithWaitForStart(false),
		sdkoptions.WithTimeout(createTimeout),
	)
	if err != nil {
		return nil, sbErr("create from "+source, err)
	}
	d := newSandbox(p, sb)

	if err := sb.WaitForStart(ctx, createTimeout); err != nil {
		p.deleteAfterFailure(ctx, d)
		return nil, sbErr(fmt.Sprintf("create %s from %s: wait for start", d.id, source), err)
	}
	if err := d.prepareWorkspace(ctx); err != nil {
		p.deleteAfterFailure(ctx, d)
		return nil, err
	}
	p.log.Info("daytona: sandbox created", "sandbox", d.id, "source", source,
		"duration", time.Since(start).Round(time.Millisecond))
	return d, nil
}

// Get returns the sandbox with this ID (or Daytona name) in whatever state
// it is in. A destroyed or destroying sandbox counts as not found.
func (p *Provider) Get(ctx context.Context, id string) (sandbox.Sandbox, error) {
	if id == "" {
		return nil, fmt.Errorf("daytona: get: empty id: %w", sandbox.ErrNotFound)
	}
	sb, err := p.client.Get(ctx, id)
	if err != nil {
		return nil, sbErr("get "+id, err)
	}
	if st := string(sb.State); isGone(st) {
		return nil, fmt.Errorf("daytona: get %s: sandbox is %s: %w", id, st, sandbox.ErrNotFound)
	}
	return newSandbox(p, sb), nil
}

// createParams turns the Spec into SDK create params and a description of
// the source for logs and errors.
func (p *Provider) createParams(spec sandbox.Spec, base sdktypes.SandboxBaseParams) (any, string) {
	if id := strings.TrimSpace(spec.SnapshotID); id != "" {
		p.ignoredResources(spec)
		return sdktypes.SnapshotParams{SandboxBaseParams: base, Snapshot: id}, "snapshot " + id
	}
	isImage, ref := parseTemplate(spec.Template)
	if isImage {
		params := sdktypes.ImageParams{SandboxBaseParams: base, Image: ref}
		if spec.CPUs > 0 || spec.MemoryMB > 0 {
			params.Resources = &sdktypes.Resources{CPU: max(spec.CPUs, 0), Memory: memoryGB(spec.MemoryMB)}
		}
		return params, "image " + ref
	}
	if ref == "" {
		ref = p.defaultSnapshot
	}
	p.ignoredResources(spec)
	if ref == "" {
		return sdktypes.SnapshotParams{SandboxBaseParams: base}, "default snapshot"
	}
	return sdktypes.SnapshotParams{SandboxBaseParams: base, Snapshot: ref}, "snapshot " + ref
}

func (p *Provider) ignoredResources(spec sandbox.Spec) {
	if spec.CPUs > 0 || spec.MemoryMB > 0 {
		p.log.Debug("daytona: CPUs and MemoryMB are ignored for snapshot-based sandboxes",
			"cpus", spec.CPUs, "memory_mb", spec.MemoryMB)
	}
}

func (p *Provider) sandboxLabels(spec map[string]string) map[string]string {
	labels := make(map[string]string, len(p.labels)+len(spec)+1)
	maps.Copy(labels, p.labels)
	maps.Copy(labels, spec)
	labels[LabelSandbox] = "1"
	return labels
}

// deleteAfterFailure deletes a sandbox Create could not finish, so it does
// not leak. Errors are logged; the sandbox is also labelled for reaping.
func (p *Provider) deleteAfterFailure(ctx context.Context, d *daytonaSandbox) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
	defer cancel()
	if err := d.Delete(ctx); err != nil {
		p.log.Warn("daytona: delete after failed create", "sandbox", d.id, "err", err)
	}
}

// parseTemplate applies the Template rule from the package doc. It reports
// whether ref is an image; otherwise ref is a snapshot name (possibly
// empty).
func parseTemplate(t string) (isImage bool, ref string) {
	t = strings.TrimSpace(t)
	switch {
	case strings.HasPrefix(t, "snapshot:"):
		return false, strings.TrimSpace(strings.TrimPrefix(t, "snapshot:"))
	case strings.HasPrefix(t, "image:"):
		return true, strings.TrimSpace(strings.TrimPrefix(t, "image:"))
	case strings.ContainsAny(t, "/:@"):
		return true, t
	default:
		return false, t
	}
}

// splitEgress sorts allow-list entries into CIDR blocks (IPs become /32 or
// /128) and domain names, as described in the package doc.
func splitEgress(hosts []string) (cidrs, domains []string) {
	seen := map[string]bool{}
	add := func(list *[]string, v string) {
		if v != "" && !seen[v] {
			seen[v] = true
			*list = append(*list, v)
		}
	}
	for _, h := range hosts {
		h = strings.TrimSpace(h)
		if h == "" {
			continue
		}
		if strings.Contains(h, "://") {
			if u, err := url.Parse(h); err == nil && u.Host != "" {
				h = u.Host
			}
		}
		if _, n, err := net.ParseCIDR(h); err == nil {
			add(&cidrs, n.String())
			continue
		}
		if i := strings.IndexByte(h, '/'); i >= 0 {
			h = h[:i]
		}
		if host, _, err := net.SplitHostPort(h); err == nil {
			h = host
		}
		h = strings.Trim(h, "[]")
		if ip := net.ParseIP(h); ip != nil {
			bits := 32
			if ip.To4() == nil {
				bits = 128
			}
			add(&cidrs, (&net.IPNet{IP: ip, Mask: net.CIDRMask(bits, bits)}).String())
			continue
		}
		add(&domains, strings.TrimSuffix(strings.ToLower(h), "."))
	}
	return cidrs, domains
}

// memoryGB converts MB to Daytona's whole GB, rounding up.
func memoryGB(mb int) int {
	if mb <= 0 {
		return 0
	}
	return (mb + 1023) / 1024
}

// ceilMinutes converts d to whole minutes, rounding up.
func ceilMinutes(d time.Duration) int {
	if d <= 0 {
		return 0
	}
	return int((d + time.Minute - 1) / time.Minute)
}

func intPtr(v int) *int       { return &v }
func strPtr(v string) *string { return &v }

// Daytona sandbox states (apiclient.SandboxState values), compared as
// strings so the generated API client stays out of this package's imports.
const (
	stateStarted    = "started"
	stateStarting   = "starting"
	stateStopped    = "stopped"
	stateStopping   = "stopping"
	stateArchived   = "archived"
	stateArchiving  = "archiving"
	statePaused     = "paused"
	statePausing    = "pausing"
	stateDestroyed  = "destroyed"
	stateDestroying = "destroying"
)

func isGone(state string) bool { return state == stateDestroyed || state == stateDestroying }

// sbErr wraps an SDK error from a sandbox-level call, mapping Daytona's
// not-found to sandbox.ErrNotFound.
func sbErr(op string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, sdkerrors.ErrNotFound) {
		return fmt.Errorf("daytona: %s: %w: %w", op, sandbox.ErrNotFound, err)
	}
	return fmt.Errorf("daytona: %s: %w", op, err)
}

// isFileNotFound reports whether an SDK error from a file operation means
// the path does not exist.
func isFileNotFound(err error) bool {
	var de *sdkerrors.DaytonaError
	if errors.As(err, &de) {
		if de.Code == "FILE_NOT_FOUND" || de.StatusCode == http.StatusNotFound {
			return true
		}
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "no such file") || strings.Contains(msg, "not found")
}

// withDefaultTimeout bounds ctx by d unless it already has a deadline.
func withDefaultTimeout(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, d)
}

func defaultHTTPClient() *http.Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = responseHeaderTimeout
	return &http.Client{Transport: tr}
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b) // crypto/rand.Read never fails
	return hex.EncodeToString(b)
}
