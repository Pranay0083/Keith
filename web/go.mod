// web/ is the Next.js app, not Go. This file makes web/ its own (empty) Go
// module, so `go build ./...` from the repo root skips it, including npm
// packages that ship .go files under web/node_modules.
module github.com/Pranay0083/keith/web

go 1.26
