#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DB_DIR="${ROOT_DIR}/.kieker"
DB_PATH="${DB_DIR}/fast_agent.sqlite"

mkdir -p "${DB_DIR}"

EXCLUDES=(
  --exclude ".git"
  --exclude ".venv"
  --exclude "dist"
  --exclude "node_modules"
  --exclude "__pycache__"
  --exclude ".pytest_cache"
  --exclude ".mypy_cache"
  --exclude ".ruff_cache"
)

uv tool run kieker index \
  --db "${DB_PATH}" \
  --jobs 0 \
  "${EXCLUDES[@]}" \
  "${ROOT_DIR}/src/fast_agent" \
  "${ROOT_DIR}/tests" \
  "${ROOT_DIR}/scripts"

printf '\nKieker database created at %s\n' "${DB_PATH}"
