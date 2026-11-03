package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
)

// Edit history limits. Older records are dropped first; a file whose
// previous content is too large keeps a record without it, so undo_edit
// can say why it can't help.
const (
	maxEditDepth     = 20       // records per file
	maxEditPrevBytes = 1 << 20  // previous content kept per record
	maxEditBytes     = 16 << 20 // previous content kept in total
)

// EditHistory is the per-file edit history behind undo_edit, kept in
// Aux.Edits. Each write_file, edit_file and delete_file pushes a record
// holding the file's previous content; undo_edit pops the newest one.
type EditHistory struct {
	// Seq numbers records across files, oldest first.
	Seq int64 `json:"seq,omitempty"`
	// Files maps an absolute path to its records, oldest first.
	Files map[string][]EditRecord `json:"files,omitempty"`
}

// EditRecord is one edit to a file.
type EditRecord struct {
	Seq    int64  `json:"seq"`
	Tool   string `json:"tool"`
	CallID string `json:"call_id,omitempty"`
	// Existed reports whether the file existed before the edit.
	Existed bool `json:"existed"`
	// Prev is the file's content before the edit, when it existed and was
	// small enough to keep.
	Prev []byte `json:"prev,omitempty"`
	// Mode is the file's permission bits before the edit (0 if unknown).
	Mode fs.FileMode `json:"mode,omitempty"`
	// Dropped means Prev was too large to keep, so the edit can't be
	// undone.
	Dropped bool `json:"dropped,omitempty"`
	// After is the SHA-256 (hex) of the content the edit left, or "" if
	// the edit deleted the file. undo_edit refuses when the file no
	// longer matches it (something else changed the file since).
	After string `json:"after"`
}

func contentHash(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// push records an edit to path.
func (h *EditHistory) push(path string, rec EditRecord) {
	if h.Files == nil {
		h.Files = map[string][]EditRecord{}
	}
	if len(rec.Prev) > maxEditPrevBytes {
		rec.Prev, rec.Dropped = nil, true
	}
	h.Seq++
	rec.Seq = h.Seq
	recs := append(h.Files[path], rec)
	if len(recs) > maxEditDepth {
		recs = append([]EditRecord(nil), recs[len(recs)-maxEditDepth:]...)
	}
	h.Files[path] = recs
	h.trim()
}

// trim drops the oldest records until the kept content fits maxEditBytes.
func (h *EditHistory) trim() {
	for {
		total, oldest, oldestSeq := 0, "", int64(-1)
		for p, recs := range h.Files {
			for _, r := range recs {
				total += len(r.Prev)
			}
			if len(recs) > 0 && (oldestSeq < 0 || recs[0].Seq < oldestSeq) {
				oldest, oldestSeq = p, recs[0].Seq
			}
		}
		if total <= maxEditBytes || oldestSeq < 0 {
			return
		}
		h.Files[oldest] = h.Files[oldest][1:]
		if len(h.Files[oldest]) == 0 {
			delete(h.Files, oldest)
		}
	}
}

// last returns the newest record for path.
func (h *EditHistory) last(path string) (EditRecord, bool) {
	recs := h.Files[path]
	if len(recs) == 0 {
		return EditRecord{}, false
	}
	return recs[len(recs)-1], true
}

// pop removes the newest record for path.
func (h *EditHistory) pop(path string) {
	recs := h.Files[path]
	if len(recs) == 0 {
		return
	}
	if len(recs) == 1 {
		delete(h.Files, path)
		return
	}
	h.Files[path] = recs[:len(recs)-1]
}
