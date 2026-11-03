# tygo version used by `make gen`; pinned so regenerating is reproducible.
# Override with `make gen TYGO_VERSION=latest` to try a newer one.
TYGO_VERSION ?= v0.2.21

.PHONY: gen
## gen: regenerate web/src/lib/api/events.gen.ts from internal/events
gen:
	go run github.com/gzuidhof/tygo@$(TYGO_VERSION) generate
