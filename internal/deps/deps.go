//go:build tools

// Package deps pins the control-plane dependencies that later packages
// will import, so `go mod tidy` keeps them in go.mod before any code uses
// them. The tools build tag keeps this file out of every normal build.
//
// Remove an import here once a real package imports it.
package deps

import (
	_ "github.com/daytona/clients/sdk-go/pkg/daytona"
)
