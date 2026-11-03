# syntax=docker/dockerfile:1
#
# Keith's production sandbox base image. Every session's sandbox starts
# from this image (or from a repo snapshot built on top of it).
#
# Contents: git, ripgrep, tmux, bash, curl, ca-certificates,
# build-essential, Node LTS (+ npm, pnpm, yarn), Python 3 (+ pip, venv)
# and Go. Rust and a JDK are opt-in build args.
#
# It runs as the non-root user `agent` (uid 1000), which owns /workspace.
# Nothing secret belongs in this image: no LLM keys, no GitHub App key,
# no agent code (see docs/architecture.md, "Trust boundary").
#
#   docker build -f sandbox/base.Dockerfile -t keith-sandbox:latest sandbox
#   docker build -f sandbox/base.Dockerfile -t keith-sandbox:full \
#     --build-arg INSTALL_RUST=true --build-arg INSTALL_JDK=true sandbox
FROM debian:trixie-slim

ARG NODE_MAJOR=26
ARG GO_VERSION=1.27.3
ARG INSTALL_RUST=false
ARG INSTALL_JDK=false
ARG JDK_PACKAGE=openjdk-21-jdk-headless

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Base tools and compilers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates curl file git jq less \
      openssh-client pkg-config procps ripgrep tmux unzip xz-utils zip \
      python3 python3-dev python3-pip python3-venv python-is-python3 \
 && if [ "$INSTALL_JDK" = "true" ]; then apt-get install -y --no-install-recommends "$JDK_PACKAGE"; fi \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system init.defaultBranch main

# Node.js LTS from the official tarball, checked against its SHASUMS256.
RUN set -eu; \
    case "$(dpkg --print-architecture)" in \
      amd64) arch=x64 ;; arm64) arch=arm64 ;; *) echo "unsupported arch" >&2; exit 1 ;; \
    esac; \
    base="https://nodejs.org/dist/latest-v${NODE_MAJOR}.x"; \
    curl -fsSLo /tmp/SHASUMS256.txt "$base/SHASUMS256.txt"; \
    file="$(grep -oE "node-v[0-9.]+-linux-${arch}\.tar\.xz" /tmp/SHASUMS256.txt | head -n1)"; \
    curl -fsSLo "/tmp/$file" "$base/$file"; \
    (cd /tmp && grep " $file\$" SHASUMS256.txt | sha256sum -c -); \
    tar -xJf "/tmp/$file" -C /usr/local --strip-components=1 --no-same-owner; \
    rm -f "/tmp/$file" /tmp/SHASUMS256.txt; \
    npm install -g pnpm yarn; \
    npm cache clean --force; \
    node --version; npm --version

# Go from the official tarball, checked against its published sha256.
RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    url="https://dl.google.com/go/go${GO_VERSION}.linux-${arch}.tar.gz"; \
    curl -fsSLo /tmp/go.tgz "$url"; \
    echo "$(curl -fsSL "$url.sha256")  /tmp/go.tgz" | sha256sum -c -; \
    tar -xzf /tmp/go.tgz -C /usr/local; \
    rm -f /tmp/go.tgz; \
    /usr/local/go/bin/go version

RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && mkdir -p /workspace \
 && chown agent:agent /workspace \
 && echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/go/bin:/usr/local/go/bin:$PATH"' > /etc/profile.d/keith-path.sh

USER agent
# PATH here covers non-login shells (docker exec, tmux); profile.d above
# covers login shells, whose /etc/profile resets PATH.
ENV GOPATH=/home/agent/go \
    CARGO_HOME=/home/agent/.cargo \
    RUSTUP_HOME=/home/agent/.rustup
ENV PATH=/home/agent/.local/bin:/home/agent/.cargo/bin:/home/agent/go/bin:/usr/local/go/bin:$PATH

# Rust via rustup (opt-in), installed for the agent user.
RUN if [ "$INSTALL_RUST" = "true" ]; then \
      curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path \
      && rustc --version; \
    fi

WORKDIR /workspace
# Common dev-server ports (the docker provider publishes its own list).
EXPOSE 3000 5173 8000 8080
CMD ["sleep", "infinity"]
