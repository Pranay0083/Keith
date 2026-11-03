// Package migrations holds the goose SQL migrations for the Postgres
// schema, embedded so the binary can migrate without the source tree:
//
//	goose.SetBaseFS(migrations.FS)
//	goose.SetDialect("postgres")
//	goose.Up(db, ".")
//
// Add a change as a new file (00002_<name>.sql); never edit an applied
// migration.
package migrations

import "embed"

// FS contains every migration file at its root.
//
//go:embed *.sql
var FS embed.FS
