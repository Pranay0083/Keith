package providers_test

import (
	"testing"

	"github.com/Pranay0083/keith/internal/config"
	"github.com/Pranay0083/keith/internal/sandbox/providers"
)

func TestNew(t *testing.T) {
	for _, c := range []struct {
		cfg     config.Config
		want    string
		wantErr bool
	}{
		{cfg: config.Config{SandboxProvider: config.SandboxProviderDocker}, want: "docker"},
		{cfg: config.Config{SandboxProvider: config.SandboxProviderDaytona, DaytonaAPIKey: "k"}, want: "daytona"},
		{cfg: config.Config{SandboxProvider: config.SandboxProviderDaytona}, wantErr: true},
		{cfg: config.Config{SandboxProvider: "modal"}, wantErr: true},
	} {
		p, err := providers.New(&c.cfg)
		if c.wantErr {
			if err == nil {
				t.Errorf("%+v: want error", c.cfg.SandboxProvider)
			}
			continue
		}
		if err != nil || p.Name() != c.want {
			t.Errorf("%s: got %v, %v", c.cfg.SandboxProvider, p, err)
		}
	}
	if _, err := providers.New(nil); err == nil {
		t.Error("nil config accepted")
	}

	// The config's defaults select docker.
	cfg, err := config.LoadFrom(func(string) string { return "" })
	if err != nil {
		t.Fatal(err)
	}
	if p, err := providers.New(cfg); err != nil || p.Name() != "docker" {
		t.Fatalf("default = %v, %v", p, err)
	}
}
