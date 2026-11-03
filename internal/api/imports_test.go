package api

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestNoAgentImport enforces docs/architecture.md: internal/api never
// imports internal/agent. (internal/github imports agent for the
// RepoHost types; the rule is about this package's own imports.)
func TestNoAgentImport(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil || len(files) == 0 {
		t.Fatalf("glob: %v %v", files, err)
	}
	for _, f := range files {
		af, err := parser.ParseFile(token.NewFileSet(), f, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range af.Imports {
			p, _ := strconv.Unquote(imp.Path.Value)
			if p == "github.com/Pranay0083/keith/internal/agent" || strings.HasPrefix(p, "github.com/Pranay0083/keith/internal/agent/") {
				t.Errorf("%s imports %s", f, p)
			}
		}
	}
}
