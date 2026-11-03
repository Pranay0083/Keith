package daytona

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

// fakeDaytona is an in-process fake of the Daytona API (under /api) and the
// toolbox proxy (under /toolbox/<sandbox id>), covering the endpoints the
// provider uses. Its responses carry the fields the generated SDK clients
// require.

type fakeRun struct {
	stdout, stderr string
	exit           int
	hang           bool // never report an exit code
	endlessLogs    bool // GET .../logs streams output forever
}

type fakeCommand struct {
	id, session, command string
	run                  fakeRun
}

type fakeFile struct {
	data []byte
	mode string
	// kind is "" for a regular file, "special" for one that is not
	// (a device or FIFO), or "endless" for one that passes the read check
	// but downloads forever (a file swapped for /dev/zero after the check).
	kind string
}

type fakeSnapshot struct {
	name  string
	polls int
}

type fakeDaytona struct {
	t   *testing.T
	srv *httptest.Server

	// Set before use.
	sandboxClass string                   // "container" (default) or "linux-vm"
	wsMode       string                   // "stream" (default), "fail" or "hang"
	runner       func(cmd string) fakeRun // nil: exit 0, no output
	createState  string                   // state returned by POST /sandbox; default "starting"

	mu          sync.Mutex
	nextID      int
	sandboxes   map[string]*fakeSandbox
	createBody  []map[string]any
	requests    []string
	executes    []string // /process/execute commands
	commands    map[string]*fakeCommand
	sessions    map[string]chan struct{} // closed when deleted
	deletedSess []string
	files       map[string]fakeFile
	folders     []string
	snapshots   map[string]*fakeSnapshot
	previewQS   []string
	downloaded  int64 // bytes of endless downloads written
	endlessOut  int64 // bytes of endless command output written
	closed      chan struct{}
}

type fakeSandbox struct {
	id     string
	state  string
	labels map[string]string
	env    map[string]string
}

func newFake(t *testing.T) *fakeDaytona {
	t.Helper()
	f := &fakeDaytona{
		t:         t,
		sandboxes: map[string]*fakeSandbox{},
		commands:  map[string]*fakeCommand{},
		sessions:  map[string]chan struct{}{},
		files:     map[string]fakeFile{},
		snapshots: map[string]*fakeSnapshot{},
		closed:    make(chan struct{}),
	}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(func() {
		close(f.closed)
		f.srv.Close()
	})
	return f
}

// provider returns a Provider pointed at the fake.
func (f *fakeDaytona) provider(opts Options) *Provider {
	f.t.Helper()
	opts.APIKey = "test-key"
	opts.APIURL = f.srv.URL + "/api"
	opts.HTTPClient = f.srv.Client()
	if opts.Logger == nil {
		opts.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	p, err := New(opts)
	if err != nil {
		f.t.Fatalf("New: %v", err)
	}
	return p
}

// addSandbox registers an existing sandbox.
func (f *fakeDaytona) addSandbox(id, state string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sandboxes[id] = &fakeSandbox{id: id, state: state, labels: map[string]string{}, env: map[string]string{}}
}

func (f *fakeDaytona) state(id string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if sb := f.sandboxes[id]; sb != nil {
		return sb.state
	}
	return ""
}

func (f *fakeDaytona) requestLog() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.requests...)
}

func (f *fakeDaytona) countRequests(prefix string) int {
	n := 0
	for _, r := range f.requestLog() {
		if strings.HasPrefix(r, prefix) {
			n++
		}
	}
	return n
}

func (f *fakeDaytona) lastCommand() *fakeCommand {
	f.mu.Lock()
	defer f.mu.Unlock()
	var last *fakeCommand
	for _, c := range f.commands {
		if last == nil || c.id > last.id {
			last = c
		}
	}
	return last
}

// read runs fn with the fake's lock held, for tests inspecting its state.
func (f *fakeDaytona) read(fn func()) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fn()
}

func (f *fakeDaytona) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.requests = append(f.requests, r.Method+" "+r.URL.Path)
	f.mu.Unlock()

	switch {
	case strings.HasPrefix(r.URL.Path, "/api/"):
		f.serveAPI(w, r, strings.TrimPrefix(r.URL.Path, "/api"))
	case strings.HasPrefix(r.URL.Path, "/toolbox/"):
		rest := strings.TrimPrefix(r.URL.Path, "/toolbox/")
		id, sub, _ := strings.Cut(rest, "/")
		f.serveToolbox(w, r, id, "/"+sub)
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeDaytona) sandboxPayload(sb *fakeSandbox) map[string]any {
	class := f.sandboxClass
	if class == "" {
		class = "container"
	}
	return map[string]any{
		"id":                  sb.id,
		"organizationId":      "org-1",
		"name":                sb.id,
		"user":                "daytona",
		"env":                 sb.env,
		"labels":              sb.labels,
		"public":              false,
		"networkBlockAll":     false,
		"kvm":                 false,
		"target":              "us",
		"cpu":                 2,
		"gpu":                 0,
		"memory":              4,
		"disk":                10,
		"state":               sb.state,
		"toolboxProxyUrl":     f.srv.URL + "/toolbox",
		"autoStopInterval":    15,
		"autoArchiveInterval": 10080,
		"autoDeleteInterval":  -1,
		"sandboxClass":        class,
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func notFound(w http.ResponseWriter, msg string) {
	writeJSON(w, http.StatusNotFound, map[string]any{"message": msg, "statusCode": 404})
}

func (f *fakeDaytona) serveAPI(w http.ResponseWriter, r *http.Request, p string) {
	parts := strings.Split(strings.Trim(p, "/"), "/")
	f.mu.Lock()
	defer f.mu.Unlock()

	switch {
	case r.Method == http.MethodPost && p == "/sandbox":
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"message": err.Error()})
			return
		}
		f.createBody = append(f.createBody, body)
		f.nextID++
		sb := &fakeSandbox{id: fmt.Sprintf("sb-%d", f.nextID), state: f.createState, labels: map[string]string{}, env: map[string]string{}}
		if sb.state == "" {
			sb.state = "starting"
		}
		if l, ok := body["labels"].(map[string]any); ok {
			for k, v := range l {
				sb.labels[k] = fmt.Sprint(v)
			}
		}
		f.sandboxes[sb.id] = sb
		payload := f.sandboxPayload(sb)
		if sb.state == "starting" {
			sb.state = "started" // the next GET sees it started
		}
		writeJSON(w, http.StatusOK, payload)

	case len(parts) >= 2 && parts[0] == "sandbox":
		sb := f.sandboxes[parts[1]]
		if sb == nil || sb.state == "destroyed" {
			notFound(w, "Sandbox not found")
			return
		}
		switch {
		case len(parts) == 2 && r.Method == http.MethodGet:
			writeJSON(w, http.StatusOK, f.sandboxPayload(sb))
		case len(parts) == 2 && r.Method == http.MethodDelete:
			sb.state = "destroyed"
			writeJSON(w, http.StatusOK, f.sandboxPayload(sb))
		case len(parts) == 3 && parts[2] == "stop":
			sb.state = "stopped"
			writeJSON(w, http.StatusOK, f.sandboxPayload(sb))
		case len(parts) == 3 && parts[2] == "start":
			sb.state = "started"
			writeJSON(w, http.StatusOK, f.sandboxPayload(sb))
		case len(parts) == 3 && parts[2] == "snapshot":
			var body struct {
				Name string `json:"name"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			if f.sandboxClass == "linux-vm" && sb.state != "stopped" {
				writeJSON(w, http.StatusBadRequest, map[string]any{"message": "VM sandbox must be stopped", "statusCode": 400})
				return
			}
			f.snapshots[body.Name] = &fakeSnapshot{name: body.Name}
			writeJSON(w, http.StatusOK, f.sandboxPayload(sb))
		case len(parts) == 5 && parts[2] == "ports" && parts[4] == "signed-preview-url":
			f.previewQS = append(f.previewQS, r.URL.RawQuery)
			port, _ := strconv.Atoi(parts[3])
			writeJSON(w, http.StatusOK, map[string]any{
				"sandboxId": sb.id, "port": port, "token": "tok",
				"url": fmt.Sprintf("https://%d-tok.preview.test", port),
			})
		default:
			http.NotFound(w, r)
		}

	case len(parts) == 2 && parts[0] == "snapshots" && r.Method == http.MethodGet:
		snap := f.snapshots[parts[1]]
		if snap == nil {
			notFound(w, "Snapshot not found")
			return
		}
		snap.polls++
		state := "pending"
		if snap.polls > 1 {
			state = "active"
		}
		now := time.Now().UTC().Format(time.RFC3339)
		writeJSON(w, http.StatusOK, map[string]any{
			"id": "snap-id-" + snap.name, "general": false, "name": snap.name, "state": state,
			"size": nil, "entrypoint": []string{}, "cpu": 2, "gpu": 0, "mem": 4, "disk": 10,
			"errorReason": nil, "createdAt": now, "updatedAt": now, "lastUsedAt": nil,
			"sourceSandboxId": nil,
		})

	default:
		http.NotFound(w, r)
	}
}

var upgrader = websocket.Upgrader{}

func (f *fakeDaytona) serveToolbox(w http.ResponseWriter, r *http.Request, id, p string) {
	f.mu.Lock()
	sb := f.sandboxes[id]
	f.mu.Unlock()
	if sb == nil {
		notFound(w, "sandbox not found")
		return
	}
	parts := strings.Split(strings.Trim(p, "/"), "/")

	switch {
	case p == "/process/execute" && r.Method == http.MethodPost:
		var body struct {
			Command string `json:"command"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.executes = append(f.executes, body.Command)
		exit := f.readCheck(body.Command)
		f.mu.Unlock()
		if body.Command == "ENDLESS" {
			f.endless(w, `{"exitCode":0,"result":"`)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"exitCode": exit, "result": ""})

	case p == "/process/session" && r.Method == http.MethodPost:
		var body struct {
			SessionID string `json:"sessionId"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.sessions[body.SessionID] = make(chan struct{})
		f.mu.Unlock()
		w.WriteHeader(http.StatusCreated)

	case len(parts) == 3 && parts[0] == "process" && parts[1] == "session" && r.Method == http.MethodDelete:
		f.mu.Lock()
		if ch, ok := f.sessions[parts[2]]; ok {
			close(ch)
			delete(f.sessions, parts[2])
		}
		f.deletedSess = append(f.deletedSess, parts[2])
		f.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)

	case len(parts) == 4 && parts[0] == "process" && parts[3] == "exec":
		var body struct {
			Command  string `json:"command"`
			RunAsync bool   `json:"runAsync"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		run := fakeRun{}
		if f.runner != nil {
			run = f.runner(body.Command)
		}
		f.mu.Lock()
		f.nextID++
		cmd := &fakeCommand{id: fmt.Sprintf("cmd-%03d", f.nextID), session: parts[2], command: body.Command, run: run}
		f.commands[cmd.id] = cmd
		f.mu.Unlock()
		if !body.RunAsync {
			f.t.Errorf("exec without runAsync")
		}
		writeJSON(w, http.StatusOK, map[string]any{"cmdId": cmd.id})

	case len(parts) >= 5 && parts[0] == "process" && parts[3] == "command":
		f.mu.Lock()
		cmd := f.commands[parts[4]]
		sessDone := f.sessions[parts[2]]
		f.mu.Unlock()
		if cmd == nil {
			notFound(w, "command not found")
			return
		}
		if len(parts) == 5 {
			out := map[string]any{"id": cmd.id, "command": cmd.command}
			if !cmd.run.hang {
				out["exitCode"] = cmd.run.exit
			}
			writeJSON(w, http.StatusOK, out)
			return
		}
		if parts[5] != "logs" {
			http.NotFound(w, r)
			return
		}
		if !websocket.IsWebSocketUpgrade(r) {
			if cmd.run.endlessLogs {
				f.endless(w, `{"output":"`)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"output": cmd.run.stdout + cmd.run.stderr, "stdout": cmd.run.stdout, "stderr": cmd.run.stderr,
			})
			return
		}
		f.serveLogsWS(w, r, cmd, sessDone)

	case p == "/files/folder" && r.Method == http.MethodPost:
		f.mu.Lock()
		f.folders = append(f.folders, r.URL.Query().Get("path"))
		f.mu.Unlock()
		w.WriteHeader(http.StatusCreated)

	case p == "/files/bulk-upload" && r.Method == http.MethodPost:
		if err := r.ParseMultipartForm(64 << 20); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"message": err.Error()})
			return
		}
		dest := r.MultipartForm.Value["files[0].path"]
		fhs := r.MultipartForm.File["files[0].file"]
		if len(dest) != 1 || len(fhs) != 1 {
			writeJSON(w, http.StatusBadRequest, map[string]any{"message": "bad upload form"})
			return
		}
		fh, _ := fhs[0].Open()
		data, _ := io.ReadAll(fh)
		_ = fh.Close()
		f.mu.Lock()
		f.files[dest[0]] = fakeFile{data: data, mode: "0644"}
		f.mu.Unlock()
		w.WriteHeader(http.StatusOK)

	case p == "/files/permissions" && r.Method == http.MethodPost:
		q := r.URL.Query()
		f.mu.Lock()
		defer f.mu.Unlock()
		file, ok := f.files[q.Get("path")]
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]any{"message": "no such file or directory", "statusCode": 404, "code": "FILE_NOT_FOUND", "source": "DAYTONA_DAEMON"})
			return
		}
		file.mode = q.Get("mode")
		f.files[q.Get("path")] = file
		w.WriteHeader(http.StatusOK)

	case p == "/files/bulk-download" && r.Method == http.MethodPost:
		var body struct {
			Paths []string `json:"paths"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		file, ok := fakeFile{}, false
		if len(body.Paths) == 1 {
			file, ok = f.files[body.Paths[0]]
		}
		f.mu.Unlock()
		mw := multipart.NewWriter(w)
		w.Header().Set("Content-Type", mw.FormDataContentType())
		if ok && file.kind == "endless" {
			part, _ := mw.CreateFormFile("file", body.Paths[0])
			zeros := make([]byte, 32<<10)
			for {
				n, err := part.Write(zeros)
				f.mu.Lock()
				f.downloaded += int64(n)
				f.mu.Unlock()
				if err != nil {
					return
				}
			}
		}
		if ok {
			part, _ := mw.CreateFormFile("file", body.Paths[0])
			_, _ = part.Write(file.data)
		} else {
			part, _ := mw.CreateFormField("error")
			_, _ = part.Write([]byte(`{"message":"open: no such file or directory","statusCode":404,"code":"FILE_NOT_FOUND","source":"DAYTONA_DAEMON"}`))
		}
		_ = mw.Close()

	case p == "/files" && r.Method == http.MethodDelete:
		path := r.URL.Query().Get("path")
		f.mu.Lock()
		defer f.mu.Unlock()
		if _, ok := f.files[path]; !ok {
			writeJSON(w, http.StatusNotFound, map[string]any{"message": "file not found", "statusCode": 404, "code": "FILE_NOT_FOUND", "source": "DAYTONA_DAEMON"})
			return
		}
		delete(f.files, path)
		w.WriteHeader(http.StatusNoContent)

	default:
		http.NotFound(w, r)
	}
}

// endless writes a 200 JSON response that starts with prefix and never
// ends, until the client goes away, counting the bytes in f.endlessOut.
func (f *fakeDaytona) endless(w http.ResponseWriter, prefix string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, prefix)
	chunk := []byte(strings.Repeat("y\\n", 8<<10))
	for {
		n, err := w.Write(chunk)
		f.mu.Lock()
		f.endlessOut += int64(n)
		f.mu.Unlock()
		if err != nil {
			return
		}
	}
}

var (
	readCheckPath  = regexp.MustCompile(`p='([^']*)'`)
	readCheckLimit = regexp.MustCompile(`-gt (\d+) \]`)
)

// readCheck runs readCheckScript against the fake's files and returns its
// exit code; other commands exit 0. Callers hold f.mu.
func (f *fakeDaytona) readCheck(cmd string) int {
	if !strings.Contains(cmd, "stat -L -c") {
		return 0
	}
	m, l := readCheckPath.FindStringSubmatch(strings.ReplaceAll(cmd, `'\''`, "'")), readCheckLimit.FindStringSubmatch(cmd)
	if m == nil || l == nil {
		f.t.Errorf("unexpected read check command %q", cmd)
		return 1
	}
	file, ok := f.files[m[1]]
	limit, _ := strconv.ParseInt(l[1], 10, 64)
	switch {
	case !ok:
		return readNoFile
	case file.kind == "special":
		return readNotRegular
	case int64(len(file.data)) > limit:
		return readTooLarge
	}
	return 0
}

// serveLogsWS streams a command's output the way the daemon does: chunks
// prefixed with 0x01x3 (stdout) or 0x02x3 (stderr), then a normal close.
// In "hang" mode the socket stays open until the session is deleted.
func (f *fakeDaytona) serveLogsWS(w http.ResponseWriter, r *http.Request, cmd *fakeCommand, sessDone chan struct{}) {
	if f.wsMode == "fail" {
		http.Error(w, "websocket unsupported", http.StatusNotFound)
		return
	}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()
	send := func(prefix byte, s string) {
		if s == "" {
			return
		}
		// Split each stream in two messages to exercise reassembly.
		mid := len(s) / 2
		_ = conn.WriteMessage(websocket.BinaryMessage, append([]byte{prefix, prefix, prefix}, s[:mid]...))
		_ = conn.WriteMessage(websocket.BinaryMessage, append([]byte{prefix, prefix, prefix}, s[mid:]...))
	}
	send(1, cmd.run.stdout)
	send(2, cmd.run.stderr)
	if f.wsMode == "hang" || cmd.run.hang {
		select {
		case <-sessDone:
		case <-f.closed:
		}
	}
	_ = conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
}

func (f *fakeDaytona) sessionDeleted(id string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Contains(f.deletedSess, id)
}
