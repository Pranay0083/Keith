package github

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Pushing from the worker
//
// The sandbox's token is read-only (SandboxPermissions), so nothing in a
// sandbox can push. The worker pushes the session branch instead:
//
//  1. It mints a PushPermissions token for the one repo, which never
//     enters the sandbox, and opens the remote's receive-pack service
//     with it (OpenReceivePack), reading the remote's refs.
//  2. The sandbox writes a git bundle of the session branch, relative to
//     the remote's commits it has (git bundle create), and the worker
//     reads it with a size cap.
//  3. ParseBundle checks the bundle's header; the worker accepts it only
//     if it names exactly one ref, refs/heads/agent/<session-id>, at the
//     tip the sandbox reported.
//  4. ReceivePack.Update sends one ref update, for that ref only, with
//     the bundle's pack, and an old value the remote must still hold (a
//     lease). The remote checks the pack (its objects and connectivity)
//     and refuses the update if the ref moved.
//
// The token is sent only in the Authorization header of these requests:
// no git process, argv, URL or environment ever holds it.

// ZeroOID is the all-zero object ID: in a ref update, the ref does not
// exist.
const ZeroOID = "0000000000000000000000000000000000000000"

// Limits on what the remote may send back.
const (
	maxAdvertisement = 32 << 20
	maxPushResponse  = 1 << 20
	maxRemoteMessage = 2000
	maxPktLen        = 65520
)

// gitUserAgent is the User-Agent of the worker's git requests. Some git
// servers only speak the smart protocol to agents starting with "git/".
const gitUserAgent = "git/keith-worker"

// IsObjectID reports whether s is a full SHA-1 object ID in lowercase
// hex, as git prints them.
func IsObjectID(s string) bool {
	if len(s) != len(ZeroOID) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// RefUpdate moves one ref on the remote.
type RefUpdate struct {
	// Ref is the full ref name, refs/heads/<branch>.
	Ref string
	// Old is the value the remote must still hold, or "" (or ZeroOID)
	// when the ref must not exist yet. The remote refuses the update
	// otherwise, atomically.
	Old string
	// New is the commit the ref moves to. Deleting a ref is not
	// supported.
	New string
}

// ReceivePack is an open push to one repo over git's smart HTTP
// protocol: the refs the remote advertised, and Update. Use it from the
// worker only; the token it holds must never reach a sandbox.
type ReceivePack struct {
	client *http.Client
	url    string // the repo URL (…/owner/name.git), after any redirect
	repo   string
	token  string
	refs   map[string]string
	caps   map[string]string
}

// OpenReceivePack reads the receive-pack ref advertisement of repo
// (owner/name) on the remote, authenticating as x-access-token with
// token. client nil means http.DefaultClient's transport; either way the
// request follows redirects only within the remote's scheme and host,
// so the token is never sent anywhere else.
func (r GitRemote) OpenReceivePack(ctx context.Context, client *http.Client, repo, token string) (*ReceivePack, error) {
	if r.Base == nil {
		return nil, errors.New("github: push: no git remote")
	}
	if token == "" {
		return nil, errors.New("github: push: empty token")
	}
	repoURL, err := r.CloneURL(repo)
	if err != nil {
		return nil, err
	}
	p := &ReceivePack{client: sameHostClient(client, r.Base), url: repoURL, repo: repo, token: token}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, repoURL+"/info/refs?service=git-receive-pack", nil)
	if err != nil {
		return nil, err
	}
	p.setHeaders(req)
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("github: push to %s: %w", repo, scrubURLError(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, p.httpError("open", resp)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-git-receive-pack-advertisement" {
		return nil, fmt.Errorf("github: push to %s: the remote is not a smart git HTTP server (content type %q)", repo, ct)
	}
	// Later requests go where a redirect led (as git does), still on the
	// same host.
	if u := resp.Request.URL; u != nil {
		if base, ok := strings.CutSuffix(u.Path, "/info/refs"); ok {
			v := *u
			v.Path, v.RawPath, v.RawQuery, v.Fragment = base, "", "", ""
			p.url = v.String()
		}
	}
	if err := p.readAdvertisement(io.LimitReader(resp.Body, maxAdvertisement)); err != nil {
		return nil, fmt.Errorf("github: push to %s: %w", repo, err)
	}
	return p, nil
}

// Ref returns the object ID ref points at on the remote, or "" when the
// remote has no such ref.
func (p *ReceivePack) Ref(name string) string { return p.refs[name] }

// Update sends u with pack, a git pack file holding the objects the
// remote needs (EmptyPack when it has them all), and checks the remote's
// report. It is the only ref the push touches.
func (p *ReceivePack) Update(ctx context.Context, u RefUpdate, pack []byte) error {
	if err := checkPushRef(u.Ref); err != nil {
		return err
	}
	old := u.Old
	if old == "" {
		old = ZeroOID
	}
	if !IsObjectID(old) || !IsObjectID(u.New) || u.New == ZeroOID {
		return fmt.Errorf("github: push %s: invalid object IDs %q -> %q", u.Ref, old, u.New)
	}
	if len(pack) < 32 || !bytes.HasPrefix(pack, []byte("PACK")) {
		return fmt.Errorf("github: push %s: not a pack file", u.Ref)
	}
	if _, ok := p.caps["report-status"]; !ok {
		return fmt.Errorf("github: push to %s: the remote does not report push results (no report-status)", p.repo)
	}
	if _, ok := p.caps["ofs-delta"]; !ok {
		return fmt.Errorf("github: push to %s: the remote does not accept offset deltas (no ofs-delta)", p.repo)
	}
	caps := []string{"report-status"}
	sideband := false
	if _, ok := p.caps["side-band-64k"]; ok {
		caps, sideband = append(caps, "side-band-64k"), true
	}
	if _, ok := p.caps["quiet"]; ok {
		caps = append(caps, "quiet")
	}
	if _, ok := p.caps["agent"]; ok {
		caps = append(caps, "agent="+strings.TrimPrefix(gitUserAgent, "git/"))
	}

	var body bytes.Buffer
	body.Grow(len(pack) + 256)
	writePkt(&body, fmt.Sprintf("%s %s %s\x00%s\n", old, u.New, u.Ref, strings.Join(caps, " ")))
	body.WriteString("0000")
	body.Write(pack)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url+"/git-receive-pack", bytes.NewReader(body.Bytes()))
	if err != nil {
		return err
	}
	p.setHeaders(req)
	req.Header.Set("Content-Type", "application/x-git-receive-pack-request")
	req.Header.Set("Accept", "application/x-git-receive-pack-result")
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("github: push %s to %s: %w", u.Ref, p.repo, scrubURLError(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return p.httpError("push "+u.Ref, resp)
	}
	report, remote, err := readPushResponse(io.LimitReader(resp.Body, maxPushResponse), sideband)
	if err != nil {
		return fmt.Errorf("github: push %s to %s: %w%s", u.Ref, p.repo, err, remoteSuffix(remote))
	}
	if err := checkReport(report, u.Ref); err != nil {
		return fmt.Errorf("github: push %s to %s: %w%s", u.Ref, p.repo, err, remoteSuffix(remote))
	}
	return nil
}

func (p *ReceivePack) setHeaders(req *http.Request) {
	req.SetBasicAuth("x-access-token", p.token)
	req.Header.Set("User-Agent", gitUserAgent)
}

// httpError describes a failed request with the start of the remote's
// answer (GitHub explains a refused push in the body).
func (p *ReceivePack) httpError(what string, resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	var why string
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		why = ": the push token was refused"
	case http.StatusNotFound:
		why = ": the repo was not found, or the push token cannot see it"
	}
	if msg := cleanRemoteText(string(b)); msg != "" {
		why += " (remote: " + msg + ")"
	}
	return fmt.Errorf("github: %s on %s: HTTP %d%s", what, p.repo, resp.StatusCode, why)
}

func remoteSuffix(remote string) string {
	if remote == "" {
		return ""
	}
	return " (remote: " + remote + ")"
}

// readAdvertisement parses a smart HTTP receive-pack advertisement:
// "# service=git-receive-pack", a flush, then "<oid> <ref>" lines (the
// first with "\0<capabilities>") and a flush.
func (p *ReceivePack) readAdvertisement(r io.Reader) error {
	pr := pktReader{r: bufio.NewReader(r)}
	data, flush, err := pr.next()
	if err != nil || flush || strings.TrimSuffix(string(data), "\n") != "# service=git-receive-pack" {
		return errors.New("the remote's ref advertisement does not start with the receive-pack service line")
	}
	if _, flush, err := pr.next(); err != nil || !flush {
		return errors.New("the remote's ref advertisement is malformed")
	}
	p.refs, p.caps = map[string]string{}, map[string]string{}
	first := true
	for {
		data, flush, err := pr.next()
		if err != nil {
			return fmt.Errorf("the remote's ref advertisement is truncated: %w", err)
		}
		if flush {
			break
		}
		line := strings.TrimSuffix(string(data), "\n")
		if first && line == "version 1" {
			continue
		}
		if first {
			first = false
			var caps string
			line, caps, _ = strings.Cut(line, "\x00")
			for _, c := range strings.Fields(caps) {
				k, v, _ := strings.Cut(c, "=")
				p.caps[k] = v
			}
		}
		if strings.HasPrefix(line, "shallow ") {
			continue
		}
		oid, name, ok := strings.Cut(line, " ")
		if !ok || !IsObjectID(oid) || name == "" {
			return fmt.Errorf("the remote's ref advertisement has a malformed line %q", truncate(line, 100))
		}
		if name == "capabilities^{}" || strings.HasSuffix(name, ".have") {
			continue // an empty repo, or objects from alternates
		}
		p.refs[name] = oid
	}
	if f, ok := p.caps["object-format"]; ok && f != "sha1" {
		return fmt.Errorf("the remote uses object format %q; only sha1 is supported", f)
	}
	return nil
}

// readPushResponse reads receive-pack's answer: the report-status
// pkt-lines, demultiplexed from side band 1 when sideband is set. remote
// collects what the remote printed (side band 2), trimmed.
func readPushResponse(r io.Reader, sideband bool) (report []string, remote string, err error) {
	if !sideband {
		report, err = readReport(r)
		return report, "", err
	}
	pr := pktReader{r: bufio.NewReader(r)}
	var band1 bytes.Buffer
	var msgs strings.Builder
	for {
		data, flush, err := pr.next()
		if errors.Is(err, io.EOF) || flush {
			break
		}
		if err != nil {
			return nil, cleanRemoteText(msgs.String()), fmt.Errorf("reading the remote's answer: %w", err)
		}
		if len(data) == 0 {
			continue
		}
		switch data[0] {
		case 1:
			band1.Write(data[1:])
		case 2:
			if msgs.Len() < 4*maxRemoteMessage {
				msgs.Write(data[1:])
			}
		case 3:
			return nil, cleanRemoteText(msgs.String()), fmt.Errorf("the remote failed: %s", cleanRemoteText(string(data[1:])))
		}
	}
	remote = cleanRemoteText(msgs.String())
	report, err = readReport(&band1)
	return report, remote, err
}

// readReport reads report-status pkt-lines up to the flush.
func readReport(r io.Reader) ([]string, error) {
	pr := pktReader{r: bufio.NewReader(r)}
	var lines []string
	for {
		data, flush, err := pr.next()
		if err != nil {
			if len(lines) == 0 {
				return nil, errors.New("the remote sent no push report")
			}
			return nil, fmt.Errorf("the remote's push report is truncated: %w", err)
		}
		if flush {
			return lines, nil
		}
		lines = append(lines, strings.TrimSuffix(string(data), "\n"))
	}
}

// checkReport checks "unpack ok" and "ok <ref>" in a push report.
func checkReport(lines []string, ref string) error {
	if len(lines) == 0 {
		return errors.New("the remote sent an empty push report")
	}
	if st, ok := strings.CutPrefix(lines[0], "unpack "); !ok || st != "ok" {
		return fmt.Errorf("the remote could not store the pushed objects: %s", cleanRemoteText(strings.TrimPrefix(lines[0], "unpack ")))
	}
	for _, l := range lines[1:] {
		if l == "ok "+ref {
			return nil
		}
		if rest, ok := strings.CutPrefix(l, "ng "+ref+" "); ok {
			return fmt.Errorf("the remote refused the update: %s", cleanRemoteText(rest))
		}
	}
	return fmt.Errorf("the remote's push report does not mention %s", ref)
}

// EmptyPack returns a pack file with no objects, for an update to a
// commit the remote already has.
func EmptyPack() []byte {
	h := []byte{'P', 'A', 'C', 'K', 0, 0, 0, 2, 0, 0, 0, 0}
	sum := sha1.Sum(h)
	return append(h, sum[:]...)
}

// BundleRef is a ref a bundle carries.
type BundleRef struct {
	Name string
	OID  string
}

// Bundle is a parsed git bundle (v2, or v3 with object-format=sha1 and no
// filter).
type Bundle struct {
	// Prerequisites are the commits the receiving side must already have.
	Prerequisites []string
	// Refs are the refs the bundle carries, in order.
	Refs []BundleRef
	// Pack is the pack file after the header.
	Pack []byte
}

const (
	maxBundleHeader = 1 << 20
	maxBundleLines  = 10000
)

// ParseBundle parses the header of a git bundle (git bundle create) and
// returns it with the pack that follows. It checks the format only; the
// caller decides which refs it accepts. Its errors never quote the
// bundle's contents beyond a short, sanitized line.
func ParseBundle(b []byte) (Bundle, error) {
	var out Bundle
	rest := b
	read := 0
	line := func() (string, bool) {
		i := bytes.IndexByte(rest, '\n')
		if i < 0 || read+i+1 > maxBundleHeader {
			return "", false
		}
		l := string(rest[:i])
		rest, read = rest[i+1:], read+i+1
		return l, true
	}
	sig, ok := line()
	var version int
	switch {
	case !ok:
		return out, errors.New("github: not a git bundle")
	case sig == "# v2 git bundle":
		version = 2
	case sig == "# v3 git bundle":
		version = 3
	default:
		return out, errors.New("github: not a git bundle (unsupported signature)")
	}
	for n := 0; ; n++ {
		l, ok := line()
		if !ok || n > maxBundleLines {
			return out, errors.New("github: bundle header is truncated or too long")
		}
		if l == "" {
			break
		}
		switch {
		case l[0] == '@':
			if version < 3 || l != "@object-format=sha1" {
				return out, fmt.Errorf("github: bundle has an unsupported capability %q", truncate(l, 60))
			}
		case l[0] == '-':
			oid, _, _ := strings.Cut(l[1:], " ")
			if !IsObjectID(oid) {
				return out, fmt.Errorf("github: bundle has a malformed prerequisite %q", truncate(l, 60))
			}
			out.Prerequisites = append(out.Prerequisites, oid)
		default:
			oid, name, ok := strings.Cut(l, " ")
			if !ok || !IsObjectID(oid) || name == "" || !utf8.ValidString(name) || strings.ContainsAny(name, " \x00\t\r") {
				return out, fmt.Errorf("github: bundle has a malformed ref line %q", truncate(l, 60))
			}
			out.Refs = append(out.Refs, BundleRef{Name: name, OID: oid})
		}
	}
	if len(rest) < 32 || !bytes.HasPrefix(rest, []byte("PACK")) {
		return out, errors.New("github: bundle has no pack after its header")
	}
	if v := binary.BigEndian.Uint32(rest[4:8]); v != 2 && v != 3 {
		return out, fmt.Errorf("github: bundle pack has unsupported version %d", v)
	}
	out.Pack = rest
	return out, nil
}

// checkPushRef allows only a branch ref with a sane name.
func checkPushRef(ref string) error {
	name, ok := strings.CutPrefix(ref, "refs/heads/")
	if !ok || name == "" || strings.ContainsAny(ref, " \x00\n\r\t~^:?*[\\") || strings.Contains(ref, "..") {
		return fmt.Errorf("github: push: %q is not a branch ref", ref)
	}
	return nil
}

// sameHostClient returns a copy of c (nil: a default client) that
// follows at most 5 redirects, and only to base's scheme and host.
func sameHostClient(c *http.Client, base *url.URL) *http.Client {
	var cc http.Client
	if c != nil {
		cc = *c
	}
	cc.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 {
			return errors.New("too many redirects")
		}
		if req.URL.Scheme != base.Scheme || req.URL.Host != base.Host {
			return fmt.Errorf("refusing a redirect to another host (%s)", req.URL.Host)
		}
		return nil
	}
	return &cc
}

// scrubURLError drops the URL from a *url.Error (it holds no credentials,
// but keeps errors short and stable).
func scrubURLError(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		return ue.Err
	}
	return err
}

// cleanRemoteText makes text from the remote safe to show: printable,
// one line per message, at most maxRemoteMessage bytes.
func cleanRemoteText(s string) string {
	s = strings.ToValidUTF8(s, "?")
	var b strings.Builder
	for _, l := range strings.FieldsFunc(s, func(r rune) bool { return r == '\n' || r == '\r' }) {
		l = strings.TrimSpace(strings.Map(func(r rune) rune {
			if r < 0x20 || r == 0x7f {
				return -1
			}
			return r
		}, l))
		if l == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("; ")
		}
		b.WriteString(l)
	}
	return truncate(b.String(), maxRemoteMessage)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

// pktReader reads git pkt-lines.
type pktReader struct{ r *bufio.Reader }

// next returns the next pkt-line's payload, or flush for "0000".
func (p pktReader) next() (data []byte, flush bool, err error) {
	var hdr [4]byte
	if _, err := io.ReadFull(p.r, hdr[:]); err != nil {
		return nil, false, err
	}
	n, err := strconv.ParseUint(string(hdr[:]), 16, 16)
	if err != nil {
		return nil, false, fmt.Errorf("bad pkt-line length %q", hdr[:])
	}
	switch {
	case n == 0:
		return nil, true, nil
	case n < 4 || n > maxPktLen:
		return nil, false, fmt.Errorf("bad pkt-line length %d", n)
	}
	data = make([]byte, n-4)
	if _, err := io.ReadFull(p.r, data); err != nil {
		return nil, false, io.ErrUnexpectedEOF
	}
	return data, false, nil
}

func writePkt(w *bytes.Buffer, s string) {
	fmt.Fprintf(w, "%04x%s", len(s)+4, s)
}
