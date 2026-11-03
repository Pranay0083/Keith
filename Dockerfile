# syntax=docker/dockerfile:1
#
# The keith binary. One image serves both control-plane services; the
# subcommand picks the role:
#
#   docker build -t keith:dev --build-arg VERSION="$(git describe --always --dirty)" .
#   docker run --rm keith:dev version
#   docker run ... -p 8080:8080 keith:dev        # keith api (the default command)
#   docker run ... keith:dev worker              # keith worker
#
# Configuration is environment variables only (SETUP.md lists them).
# docker-compose.yml runs both services under its `app` profile with .env;
# note that `docker run --env-file` keeps quotes literally, unlike compose.
# This is not the sandbox image: sandboxes start from
# sandbox/base.Dockerfile, which never holds any of this image's secrets.
#
# The build stage cross-compiles (CGO is off), so
# `docker buildx build --platform linux/amd64 .` is quick on an arm64 host.
# No BuildKit cache mounts, so the file builds unchanged on hosted builders
# (Fly.io, Railway) that restrict them.

ARG GO_VERSION=1.26
ARG DOCKER_CLI_VERSION=29

# The static docker CLI, for the worker's docker sandbox provider (it
# drives the daemon with `docker run/exec/commit/...`). Only the client
# binary is copied; there is no daemon in this image.
FROM docker:${DOCKER_CLI_VERSION}-cli AS docker-cli

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-trixie AS build
# go.mod's go directive may require a newer patch than the image ships;
# auto lets the go command fetch that toolchain instead of failing.
ENV CGO_ENABLED=0 \
    GOTOOLCHAIN=auto
WORKDIR /src

# Modules first, in their own layer, so code changes don't re-download them.
COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .
ARG TARGETOS
ARG TARGETARCH
ARG VERSION=dev
RUN GOOS="$TARGETOS" GOARCH="$TARGETARCH" \
    go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/keith ./cmd/keith

FROM debian:trixie-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="keith" \
      org.opencontainers.image.description="Keith control plane: keith api and keith worker" \
      org.opencontainers.image.source="https://github.com/Pranay0083/keith" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=build /out/keith /usr/local/bin/keith

# Run as a non-root user. The worker with SANDBOX_PROVIDER=docker also
# needs the host's docker socket mounted at /var/run/docker.sock, and this
# user must be able to open it: add the socket's group to the container
# (`docker run --group-add <gid>`; docker-compose.yml does it with
# group_add and KEITH_DOCKER_GID) or run that container as root. Either way
# the socket is root-equivalent on the host, so that setup is for local
# development only; production workers use SANDBOX_PROVIDER=daytona and no
# socket.
RUN groupadd --system --gid 10001 keith \
 && useradd --system --uid 10001 --gid keith --create-home --home-dir /home/keith \
      --shell /usr/sbin/nologin keith
USER keith
WORKDIR /home/keith

# keith api listens on LISTEN_ADDR (default :8080) and serves /healthz.
EXPOSE 8080
ENTRYPOINT ["keith"]
CMD ["api"]
