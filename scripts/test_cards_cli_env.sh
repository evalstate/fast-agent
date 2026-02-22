#!/usr/bin/env bash
set -euo pipefail

FAST_AGENT_BIN="${FAST_AGENT_BIN:-fast-agent}"
KEEP_TMP="${KEEP_TMP:-0}"

if ! command -v git >/dev/null 2>&1; then
  echo "❌ git is required for cards CLI smoke test" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fa-cards-env-XXXXXX")"
if [[ "$KEEP_TMP" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

ENV_DIR="$WORK_DIR/custom-env"
PACK_REPO="$WORK_DIR/card-packs-local"
MARKETPLACE="$WORK_DIR/marketplace.json"

mkdir -p "$ENV_DIR" "$PACK_REPO"

git -C "$PACK_REPO" init >/dev/null
git -C "$PACK_REPO" config user.email tests@example.com
git -C "$PACK_REPO" config user.name "Test User"

mkdir -p "$PACK_REPO/packs/alpha/agent-cards"
cat > "$PACK_REPO/packs/alpha/card-pack.yaml" <<'YAML'
schema_version: 1
name: alpha
kind: card
install:
  agent_cards:
    - agent-cards/alpha.md
  tool_cards: []
  files: []
YAML

cat > "$PACK_REPO/packs/alpha/agent-cards/alpha.md" <<'MD'
---
name: alpha
model: passthrough
---

hello from alpha
MD

git -C "$PACK_REPO" add .
git -C "$PACK_REPO" commit -m "add alpha pack" >/dev/null

cat > "$MARKETPLACE" <<JSON
{
  "entries": [
    {
      "name": "alpha",
      "kind": "card",
      "repo_url": "$PACK_REPO",
      "repo_path": "packs/alpha"
    }
  ]
}
JSON

cat > "$ENV_DIR/fastagent.config.yaml" <<YAML
default_model: passthrough
cards:
  marketplace_urls:
    - "$MARKETPLACE"
YAML

echo "[cards-smoke] 1/4 add using --env config (no --registry)"
"$FAST_AGENT_BIN" --env "$ENV_DIR" cards add alpha >/dev/null

test -f "$ENV_DIR/agent-cards/alpha.md"
echo "[cards-smoke]      OK: installed via cards.marketplace_urls in env config"

echo "[cards-smoke] 2/4 remove"
"$FAST_AGENT_BIN" --env "$ENV_DIR" cards remove alpha >/dev/null

test ! -f "$ENV_DIR/agent-cards/alpha.md"
echo "[cards-smoke]      OK: removed"

echo "[cards-smoke] 3/4 bad configured registry should fail"
cat > "$ENV_DIR/fastagent.config.yaml" <<YAML
default_model: passthrough
cards:
  marketplace_urls:
    - "$WORK_DIR/does-not-exist.json"
YAML

if "$FAST_AGENT_BIN" --env "$ENV_DIR" cards add alpha >/dev/null 2>&1; then
  echo "❌ expected failure with bad configured registry" >&2
  exit 1
fi
echo "[cards-smoke]      OK: failed as expected"

echo "[cards-smoke] 4/4 --registry override should work"
"$FAST_AGENT_BIN" --env "$ENV_DIR" cards --registry "$MARKETPLACE" add alpha >/dev/null

test -f "$ENV_DIR/agent-cards/alpha.md"
echo "[cards-smoke]      OK: --registry override works"

echo "✅ cards CLI env smoke passed"
if [[ "$KEEP_TMP" == "1" ]]; then
  echo "[cards-smoke] tmp kept at: $WORK_DIR"
fi
