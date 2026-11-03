package daytona

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"
	"sync"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	sdkerrors "github.com/daytona/clients/sdk-go/pkg/errors"
	sdkoptions "github.com/daytona/clients/sdk-go/pkg/options"
	sdktypes "github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/Pranay0083/keith/internal/sandbox"
	"github.com/Pranay0083/keith/internal/sandbox/internal/sbutil"
)

// daytonaSandbox is one Daytona sandbox.
type daytonaSandbox struct {
	p  *Provider
	sb *sdk.Sandbox

	// id and user are copied from the SDK sandbox at construction so they
	// can be read without touching SDK state.
	id   string
	user string

	// lifecycle serializes calls that refresh or change the SDK sandbox's
	// state (Pause, Resume, Snapshot).
	lifecycle sync.Mutex
}

var _ sandbox.Sandbox = (*daytonaSandbox)(nil)

func newSandbox(p *Provider, sb *sdk.Sandbox) *daytonaSandbox {
	return &daytonaSandbox{p: p, sb: sb, id: sb.ID, user: sb.User}
}

// ID returns the Daytona sandbox ID.
func (s *daytonaSandbox) ID() string { return s.id }

// workspaceScript creates sandbox.WorkspaceDir and makes it writable by the
// default user, using passwordless sudo where the user cannot do it alone.
var workspaceScript = fmt.Sprintf(
	`d=%[1]s; mkdir -p "$d" 2>/dev/null || sudo -n mkdir -p "$d"; `+
		`if [ ! -w "$d" ]; then sudo -n chown "$(id -u):$(id -g)" "$d"; fi; `+
		`test -d "$d" && test -w "$d"`,
	sbutil.Quote(sandbox.WorkspaceDir))

func (s *daytonaSandbox) prepareWorkspace(ctx context.Context) error {
	res, err := s.sb.Process.ExecuteCommand(ctx, workspaceScript, sdkoptions.WithExecuteTimeout(time.Minute))
	if err != nil {
		return sbErr(fmt.Sprintf("create %s: prepare %s", s.id, sandbox.WorkspaceDir), err)
	}
	if res.ExitCode != 0 {
		out := strings.TrimSpace(res.Result)
		if len(out) > 512 {
			out = out[len(out)-512:]
		}
		return fmt.Errorf("daytona: create %s: cannot make %s writable (exit %d): %s",
			s.id, sandbox.WorkspaceDir, res.ExitCode, out)
	}
	return nil
}

// ReadFile is ReadFileLimit with sandbox.MaxReadFile.
func (s *daytonaSandbox) ReadFile(ctx context.Context, p string) ([]byte, error) {
	return s.ReadFileLimit(ctx, p, sandbox.MaxReadFile)
}

// Exit codes of readCheckScript.
const (
	readNoFile     = 44
	readIsDir      = 45
	readNotRegular = 46
	readTooLarge   = 47
)

// readCheckScript exits 0 if abs is a regular file (symlinks followed) of
// at most limit bytes, and with one of the read* codes otherwise. It
// prints nothing, so its response stays small.
func readCheckScript(abs string, limit int64) string {
	script := fmt.Sprintf(`p=%s; [ -e "$p" ] || exit %d; [ -d "$p" ] && exit %d; [ -f "$p" ] || exit %d; `+
		`s=$(stat -L -c %%s -- "$p" 2>/dev/null) && [ "$s" -gt %d ] && exit %d; exit 0`,
		sbutil.Quote(abs), readNoFile, readIsDir, readNotRegular, limit, readTooLarge)
	return "sh -c " + sbutil.Quote(script) + " < /dev/null > /dev/null 2>&1"
}

// ReadFileLimit checks the file with a command, then downloads it with
// the toolbox's streaming download, reading at most limit+1 bytes.
func (s *daytonaSandbox) ReadFileLimit(ctx context.Context, p string, limit int64) ([]byte, error) {
	if limit <= 0 {
		limit = sandbox.MaxReadFile
	}
	limit = min(limit, 1<<50)
	abs := sbutil.ResolvePath(p)
	ctx, cancel := withDefaultTimeout(ctx, fileOpTimeout)
	defer cancel()

	res, err := s.sb.Process.ExecuteCommand(ctx, readCheckScript(abs, limit), sdkoptions.WithExecuteTimeout(time.Minute))
	if err != nil {
		return nil, sbErr(fmt.Sprintf("read %s in %s: check", abs, s.id), err)
	}
	switch res.ExitCode {
	case 0:
	case readNoFile:
		return nil, &fs.PathError{Op: "read", Path: abs, Err: fs.ErrNotExist}
	case readIsDir:
		return nil, &fs.PathError{Op: "read", Path: abs, Err: fmt.Errorf("is a directory: %w", sandbox.ErrNotRegular)}
	case readNotRegular:
		return nil, &fs.PathError{Op: "read", Path: abs, Err: sandbox.ErrNotRegular}
	case readTooLarge:
		return nil, &fs.PathError{Op: "read", Path: abs, Err: sandbox.ErrFileTooLarge}
	default:
		return nil, fmt.Errorf("daytona: read %s in %s: check exited %d", abs, s.id, res.ExitCode)
	}

	rc, err := s.sb.FileSystem.DownloadFileStream(ctx, abs)
	if err != nil {
		return nil, s.fileErr("read", abs, err)
	}
	defer rc.Close()
	data, err := io.ReadAll(io.LimitReader(rc, limit+1))
	if err != nil {
		return nil, fmt.Errorf("daytona: read %s: %w", abs, err)
	}
	if int64(len(data)) > limit {
		// The file grew or was swapped after the check.
		return nil, &fs.PathError{Op: "read", Path: abs, Err: sandbox.ErrFileTooLarge}
	}
	return data, nil
}

// WriteFile creates the parent directory, streams the data up (the SDK
// pipes it into a multipart request without buffering a second copy) and
// sets the mode. A zero mode means 0644.
func (s *daytonaSandbox) WriteFile(ctx context.Context, p string, data []byte, mode fs.FileMode) error {
	abs := sbutil.ResolvePath(p)
	ctx, cancel := withDefaultTimeout(ctx, fileOpTimeout)
	defer cancel()
	if dir := path.Dir(abs); dir != "/" {
		// The upload may create parents itself and older daemons may refuse
		// an existing folder, so a failure here is only fatal if the upload
		// then fails too.
		if err := s.sb.FileSystem.CreateFolder(ctx, dir, sdkoptions.WithMode("0755")); err != nil {
			s.p.log.Debug("daytona: create folder failed; trying the upload anyway",
				"sandbox", s.id, "dir", dir, "err", err)
		}
	}
	if err := s.sb.FileSystem.UploadFileStream(ctx, bytes.NewReader(data), abs); err != nil {
		return s.fileErr("write", abs, err)
	}
	perm := mode.Perm()
	if perm == 0 {
		perm = 0o644
	}
	if err := s.sb.FileSystem.SetFilePermissions(ctx, abs, sdkoptions.WithPermissionMode(fmt.Sprintf("%04o", perm))); err != nil {
		return s.fileErr("chmod", abs, err)
	}
	return nil
}

// RemoveFile deletes a file (or an empty directory).
func (s *daytonaSandbox) RemoveFile(ctx context.Context, p string) error {
	abs := sbutil.ResolvePath(p)
	if err := s.sb.FileSystem.DeleteFile(ctx, abs, false); err != nil {
		return s.fileErr("remove", abs, err)
	}
	return nil
}

func (s *daytonaSandbox) fileErr(op, abs string, err error) error {
	if isFileNotFound(err) {
		return &fs.PathError{Op: op, Path: abs, Err: fs.ErrNotExist}
	}
	return fmt.Errorf("daytona: %s %s in %s: %w", op, abs, s.id, err)
}

// PortURL returns a signed preview URL for port that embeds its access
// token and expires after Options.PreviewURLTTL. If signing fails it
// returns the plain preview URL, which for a private sandbox only works
// with the X-Daytona-Preview-Token header (logged as a warning).
func (s *daytonaSandbox) PortURL(ctx context.Context, port int) (string, error) {
	if port <= 0 || port > 65535 {
		return "", fmt.Errorf("daytona: port url: invalid port %d", port)
	}
	ttl := max(int(s.p.previewTTL/time.Second), 1)
	signed, err := s.sb.GetSignedPreviewLink(ctx, port, ttl)
	if err == nil && signed.URL != "" {
		return signed.URL, nil
	}
	if err == nil {
		err = errors.New("empty signed preview URL")
	}
	link, err2 := s.sb.GetPreviewLink(ctx, port)
	if err2 != nil || link.URL == "" {
		return "", sbErr(fmt.Sprintf("port url %s:%d", s.id, port), err)
	}
	s.p.log.Warn("daytona: signed preview URL unavailable; returning a preview URL that needs the X-Daytona-Preview-Token header",
		"sandbox", s.id, "port", port, "err", err)
	return link.URL, nil
}

// refreshState fetches the sandbox and returns its Daytona state. Callers
// hold s.lifecycle.
func (s *daytonaSandbox) refreshState(ctx context.Context, op string) (string, error) {
	if err := s.sb.RefreshData(ctx); err != nil {
		return "", sbErr(op+" "+s.id, err)
	}
	st := string(s.sb.State)
	if isGone(st) {
		return st, fmt.Errorf("daytona: %s %s: sandbox is %s: %w", op, s.id, st, sandbox.ErrNotFound)
	}
	return st, nil
}

// Pause stops the sandbox (Daytona "stop"). It is a no-op when the sandbox
// is already stopped, archived or paused.
func (s *daytonaSandbox) Pause(ctx context.Context) error {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	st, err := s.refreshState(ctx, "pause")
	if err != nil {
		return err
	}
	switch st {
	case stateStopped, stateArchived, stateArchiving, statePaused:
		return nil
	case stateStopping, statePausing:
		return sbErr("pause "+s.id, s.sb.WaitForStop(ctx, stopTimeout))
	}
	return sbErr("pause "+s.id, s.sb.StopWithTimeout(ctx, stopTimeout, false))
}

// Resume starts the sandbox from stopped, archived or paused. It is a
// no-op when the sandbox is already started and waits when it is starting.
func (s *daytonaSandbox) Resume(ctx context.Context) error {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	st, err := s.refreshState(ctx, "resume")
	if err != nil {
		return err
	}
	switch st {
	case stateStarted:
		return nil
	case stateStarting, "restoring", "resuming", "creating", "pending_build", "building_snapshot", "pulling_snapshot":
		return sbErr("resume "+s.id, s.sb.WaitForStart(ctx, startTimeout))
	case stateStopping:
		if err := s.sb.WaitForStop(ctx, stopTimeout); err != nil {
			return sbErr("resume "+s.id, err)
		}
	}
	return sbErr("resume "+s.id, s.sb.StartWithTimeout(ctx, startTimeout))
}

// Snapshot saves the sandbox's filesystem as a new Daytona snapshot and
// returns its name, usable as Spec.SnapshotID, once Daytona reports it
// active. A started linux-vm sandbox is stopped for the snapshot and
// started again.
func (s *daytonaSandbox) Snapshot(ctx context.Context, name string) (string, error) {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	ctx, cancel := context.WithTimeout(ctx, snapshotTimeout)
	defer cancel()

	st, err := s.refreshState(ctx, "snapshot")
	if err != nil {
		return "", err
	}
	isVM := s.sb.SandboxClass != nil && *s.sb.SandboxClass == sdktypes.SandboxClassLinuxVM
	restart := isVM && st == stateStarted
	if restart {
		if err := s.sb.StopWithTimeout(ctx, stopTimeout, false); err != nil {
			return "", sbErr("snapshot "+s.id+": stop", err)
		}
	}

	snapName := snapshotName(name)
	err = s.sb.CreateSnapshotWithTimeout(ctx, snapName, snapshotTimeout)
	if restart {
		if startErr := s.sb.StartWithTimeout(context.WithoutCancel(ctx), startTimeout); startErr != nil {
			err = errors.Join(err, fmt.Errorf("restart after snapshot: %w", startErr))
		}
	}
	if err != nil {
		return "", sbErr("snapshot "+s.id, err)
	}
	if err := s.waitSnapshotActive(ctx, snapName); err != nil {
		return "", err
	}
	s.p.log.Info("daytona: snapshot created", "sandbox", s.id, "snapshot", snapName)
	return snapName, nil
}

// waitSnapshotActive polls the snapshot until Daytona marks it active.
func (s *daytonaSandbox) waitSnapshotActive(ctx context.Context, name string) error {
	const maxMissing = 30 * time.Second
	start := time.Now()
	interval := 500 * time.Millisecond
	for {
		snap, err := s.p.client.Snapshot.Get(ctx, name)
		switch {
		case err == nil:
			switch snap.State {
			case "active":
				return nil
			case "error", "build_failed":
				reason := ""
				if snap.ErrorReason != nil {
					reason = ": " + *snap.ErrorReason
				}
				return fmt.Errorf("daytona: snapshot %s of %s failed (%s)%s", name, s.id, snap.State, reason)
			}
		case errors.Is(err, sdkerrors.ErrNotFound) && time.Since(start) < maxMissing:
			// Not visible yet.
		default:
			return sbErr(fmt.Sprintf("snapshot %s of %s: get", name, s.id), err)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("daytona: snapshot %s of %s: waiting for it to become active: %w", name, s.id, ctx.Err())
		case <-time.After(interval):
		}
		interval = min(interval*2, 5*time.Second)
	}
}

// snapshotName makes a unique Daytona snapshot name from a label.
func snapshotName(label string) string {
	var b strings.Builder
	dash := false
	for _, r := range strings.ToLower(label) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '.' || r == '_' {
			b.WriteRune(r)
			dash = false
		} else if !dash {
			b.WriteByte('-')
			dash = true
		}
	}
	base := strings.Trim(b.String(), "-._")
	if len(base) > 48 {
		base = strings.Trim(base[:48], "-._")
	}
	if base == "" {
		base = "snapshot"
	}
	return fmt.Sprintf("keith-%s-%s-%s", base, time.Now().UTC().Format("20060102150405"), randomHex(3))
}

// Delete deletes the sandbox. An already deleted (or deleting) sandbox
// returns nil.
func (s *daytonaSandbox) Delete(ctx context.Context) error {
	err := s.sb.Delete(ctx)
	if err == nil || errors.Is(err, sdkerrors.ErrNotFound) {
		return nil
	}
	// Daytona may still return a destroyed sandbox for a while and refuse to
	// delete it again.
	if cur, gerr := s.p.client.Get(ctx, s.id); errors.Is(gerr, sdkerrors.ErrNotFound) ||
		(gerr == nil && isGone(string(cur.State))) {
		return nil
	}
	return sbErr("delete "+s.id, err)
}
