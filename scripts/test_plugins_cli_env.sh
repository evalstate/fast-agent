#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PACK_REPO="$WORK_DIR/card-packs-local"
PROJECT_DIR="$WORK_DIR/project"
PROJECT_ENV="$PROJECT_DIR/.fast-agent"
USER_HOME="$WORK_DIR/user-home"
GLOBAL_HOME="$WORK_DIR/global-home"

run_fast_agent() {
  (
    if [ -n "${FAST_AGENT_BIN:-}" ]; then
      "$FAST_AGENT_BIN" --no-update-check "$@"
      return
    fi
    PYTHONPATH="$REPO_ROOT/src" "$REPO_ROOT/.venv/bin/python" -m fast_agent.cli.__main__ --no-update-check "$@"
  )
}

assert_file_exists() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Expected file to exist: $path" >&2
    return 1
  fi
}

assert_contains() {
  local pattern="$1"
  local path="$2"
  if ! grep -q "$pattern" "$path"; then
    echo "Expected '$path' to contain: $pattern" >&2
    echo "---- $path ----" >&2
    cat "$path" >&2
    echo "--------------" >&2
    return 1
  fi
}

git_init() {
  git -C "$1" init >/dev/null
  git -C "$1" config user.email tests@example.com
  git -C "$1" config user.name "Test User"
}

mkdir -p "$PACK_REPO" "$PROJECT_DIR" "$USER_HOME" "$GLOBAL_HOME"
git_init "$PACK_REPO"

mkdir -p "$PACK_REPO/plugins/finder"
cat > "$PACK_REPO/plugins/finder/plugin.yaml" <<'YAML'
schema_version: 1
name: finder
version: 0.1.0
description: Finder test plugin
commands:
  find:
    description: Find useful things
    input_hint: "<query>"
    handler: ./commands.py:find
    key: "c-x f"
YAML
cat > "$PACK_REPO/plugins/finder/commands.py" <<'PY'
async def find(ctx):
    return "finder v1"
PY

mkdir -p "$PACK_REPO/plugins/editor"
cat > "$PACK_REPO/plugins/editor/plugin.yaml" <<'YAML'
schema_version: 1
name: editor
version: 0.1.0
description: Editor test plugin
commands:
  edit-last:
    description: Edit last assistant message
    handler: ./commands.py:edit_last
    key: "c-x e"
YAML
cat > "$PACK_REPO/plugins/editor/commands.py" <<'PY'
async def edit_last(ctx):
    return "editor"
PY

mkdir -p "$PACK_REPO/packs/alpha/agent-cards"
cat > "$PACK_REPO/packs/alpha/agent-cards/alpha.md" <<'MD'
---
name: alpha
model: passthrough
---

hello
MD
cat > "$PACK_REPO/packs/alpha/card-pack.yaml" <<'YAML'
schema_version: 2
name: alpha
kind: card
install:
  agent_cards:
    - agent-cards/alpha.md
  tool_cards: []
  files: []
plugins:
  required:
    - editor
YAML

cat > "$PACK_REPO/marketplace.json" <<JSON
{
  "entries": [
    {
      "name": "alpha",
      "kind": "card",
      "repo_url": "$PACK_REPO",
      "repo_path": "packs/alpha"
    }
  ],
  "command_plugins": [
    {
      "name": "finder",
      "description": "Finder test plugin",
      "repo_url": "$PACK_REPO",
      "repo_path": "plugins/finder"
    },
    {
      "name": "editor",
      "description": "Editor test plugin",
      "repo_url": "$PACK_REPO",
      "repo_path": "plugins/editor"
    }
  ]
}
JSON

git -C "$PACK_REPO" add .
git -C "$PACK_REPO" commit -m "initial plugin marketplace" >/dev/null

cat > "$PROJECT_DIR/fast-agent.yaml" <<YAML
default_model: passthrough
environment_dir: "$PROJECT_ENV"
plugins:
  marketplace_url: "$PACK_REPO/marketplace.json"
YAML

(
  cd "$PROJECT_DIR"

  unset FAST_AGENT_HOME
  HOME="$USER_HOME" run_fast_agent plugins --registry "$PACK_REPO/marketplace.json" add finder --global
  assert_file_exists "$USER_HOME/.fast-agent/plugins/finder/plugin.yaml"
  assert_contains "finder" "$USER_HOME/.fast-agent/fast-agent.yaml"

  run_fast_agent plugins --registry "$PACK_REPO/marketplace.json" add finder
  assert_file_exists "$PROJECT_ENV/plugins/finder/plugin.yaml"
  assert_contains "finder" fast-agent.yaml

  run_fast_agent cards --registry "$PACK_REPO/marketplace.json" add alpha
  assert_file_exists "$PROJECT_ENV/plugins/editor/plugin.yaml"
  assert_contains "editor" fast-agent.yaml

  run_fast_agent plugins list > "$WORK_DIR/plugins.list"
  assert_contains "finder" "$WORK_DIR/plugins.list"
  assert_contains "editor" "$WORK_DIR/plugins.list"
  assert_contains "find" "$WORK_DIR/plugins.list"
  assert_contains "edit-last" "$WORK_DIR/plugins.list"
)

cat > "$GLOBAL_HOME/fast-agent.yaml" <<YAML
plugins:
  marketplace_url: "$PACK_REPO/marketplace.json"
YAML

(
  cd "$PROJECT_DIR"
  FAST_AGENT_HOME="$GLOBAL_HOME" run_fast_agent plugins --registry "$PACK_REPO/marketplace.json" add finder --global
  assert_file_exists "$GLOBAL_HOME/plugins/finder/plugin.yaml"
  assert_contains "finder" "$GLOBAL_HOME/fast-agent.yaml"

  PYTHONPATH="$REPO_ROOT/src" "$REPO_ROOT/.venv/bin/python" - <<'PY'
from fast_agent.config import get_settings
settings = get_settings("fast-agent.yaml")
assert "finder" in settings.plugins.enabled
assert "editor" in settings.plugins.enabled
assert settings.commands is not None
assert "find" in settings.commands
assert "edit-last" in settings.commands
assert settings.commands["find"].key == "c-x f"
assert settings.commands["edit-last"].key == "c-x e"
PY
)

cat > "$PACK_REPO/plugins/finder/commands.py" <<'PY'
async def find(ctx):
    return "finder v2"
PY
git -C "$PACK_REPO" add .
git -C "$PACK_REPO" commit -m "update finder plugin" >/dev/null

(
  cd "$PROJECT_DIR"
  run_fast_agent plugins update > "$WORK_DIR/plugins.update"
  assert_contains "update available" "$WORK_DIR/plugins.update" || {
    cat "$WORK_DIR/plugins.update"
    exit 1
  }
  run_fast_agent plugins update finder
  assert_contains "finder v2" "$PROJECT_ENV/plugins/finder/commands.py"
)

echo "Plugin CLI environment test completed successfully."
