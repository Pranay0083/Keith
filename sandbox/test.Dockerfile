# Small image for Keith's sandbox integration tests (KEITH_IT=1).
#
# It has what the docker provider and tmuxshell need (bash, tmux,
# coreutils, git, ripgrep), less (the pager base.Dockerfile has, to test
# that shells don't wedge on one), busybox for a tiny HTTP server, and the
# same non-root `agent` user and /workspace layout as base.Dockerfile, but
# none of the language toolchains, so it builds in seconds.
#
#   docker build -f sandbox/test.Dockerfile -t keith-sandbox-test sandbox
FROM debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash busybox ca-certificates git less procps ripgrep tmux \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && mkdir -p /workspace \
 && chown agent:agent /workspace

ENV LANG=C.UTF-8
USER agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
