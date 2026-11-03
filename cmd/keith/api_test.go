package main

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/Pranay0083/keith/internal/config"
)

func TestRunAPIValidatesConfig(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	load := func() (*config.Config, error) { return config.LoadFrom(func(string) string { return "" }) }
	err := runAPI(context.Background(), logger, load)
	if err == nil || !strings.Contains(err.Error(), "missing") || !strings.Contains(err.Error(), "SESSION_SECRET") {
		t.Fatalf("err = %v", err)
	}
	boom := errors.New("bad env")
	if err := runAPI(context.Background(), logger, func() (*config.Config, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Fatalf("load error: %v", err)
	}
}

func TestAPICommandRegistered(t *testing.T) {
	cmd, _, err := newRootCmd().Find([]string{"api"})
	if err != nil || cmd.Use != "api" {
		t.Fatalf("api command: %v %v", cmd, err)
	}
}
