#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/devel/docker-compose.yaml"
ENV_FILE="${REPO_ROOT}/docker/devel/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${REPO_ROOT}/docker/devel/.env.example"
fi

COMPOSE=(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

"${COMPOSE[@]}" up -d neo4j
"${COMPOSE[@]}" up -d ollama
# "${COMPOSE[@]}" up -d huggingface

"${COMPOSE[@]}" run --rm cli
# "${COMPOSE[@]}" run --rm heracles_dev
