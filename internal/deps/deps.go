//go:build tools

// Package deps pins the control-plane dependencies that later packages
// will import, so `go mod tidy` keeps them in go.mod before any code uses
// them. The tools build tag keeps this file out of every normal build.
//
// Remove an import here once a real package imports it.
package deps

import (
	_ "github.com/anthropics/anthropic-sdk-go"
	_ "github.com/bradleyfalzon/ghinstallation/v2"
	_ "github.com/daytona/clients/sdk-go/pkg/daytona"
	_ "github.com/google/go-github/v93/github"
	_ "github.com/google/uuid"
	_ "github.com/gorilla/websocket"
	_ "github.com/hibiken/asynq"
	_ "github.com/jackc/pgx/v5"
	_ "github.com/jackc/pgx/v5/pgxpool"
	_ "github.com/pressly/goose/v3"
	_ "github.com/redis/go-redis/v9"
	_ "google.golang.org/genai"
)
