// Command keith is the entry point for the Keith cloud coding agent.
//
// It hosts the long-running services (api, worker) and, later, the client
// commands that talk to the cloud API. Each service is a subcommand so one
// binary and one container image serve every role.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// version is overridden at build time with
// -ldflags "-X main.version=<version>".
var version = "dev"

func main() {
	if err := newRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:           "keith",
		Short:         "Keith is a multi-tenant cloud coding agent",
		Long:          "Keith runs coding tasks against your GitHub repos in isolated cloud sandboxes and opens pull requests.",
		SilenceUsage:  true,
		SilenceErrors: false,
	}
	root.AddCommand(newAPICmd(), newWorkerCmd(), newVersionCmd())
	root.AddCommand(newSandboxBenchCmd())
	return root
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the keith version",
		Args:  cobra.NoArgs,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Fprintf(cmd.OutOrStdout(), "keith %s\n", version)
		},
	}
}
