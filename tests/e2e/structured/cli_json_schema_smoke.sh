#!/usr/bin/env bash
set -euo pipefail

KEEP_TMP="${KEEP_TMP:-0}"
MODELS_RAW="${MODELS:-haiku glm sonnet codexplan kimi25 gemini3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="$SCRIPT_DIR/cli_json_schema"
SCHEMA_PATH="$FIXTURE_DIR/schema.json"
CARD_TEMPLATE="$FIXTURE_DIR/weather_agent.template.md"

if [[ ! -f "$SCHEMA_PATH" ]]; then
  echo "❌ missing schema fixture: $SCHEMA_PATH" >&2
  exit 1
fi

if [[ ! -f "$CARD_TEMPLATE" ]]; then
  echo "❌ missing agent card template: $CARD_TEMPLATE" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fa-structured-cli-XXXXXX")"
if [[ "$KEEP_TMP" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

read -r -a MODELS <<< "$MODELS_RAW"

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "❌ no models configured; set MODELS=\"haiku sonnet ...\"" >&2
  exit 1
fi

echo "[structured-cli] schema fixture: $SCHEMA_PATH"
echo "[structured-cli] models: ${MODELS[*]}"

run_one() {
  local model="$1"
  local model_slug
  model_slug="$(printf '%s' "$model" | tr '/?:=' '_')"
  local card_path="$WORK_DIR/weather-${model_slug}.md"
  local stdout_path="$WORK_DIR/${model_slug}.stdout.json"
  local stderr_path="$WORK_DIR/${model_slug}.stderr.log"

  sed "s|__MODEL__|$model|g" "$CARD_TEMPLATE" > "$card_path"

  echo "[structured-cli] running model: $model"
  if ! uv run fast-agent go \
    --noenv \
    --agent-cards "$card_path" \
    --agent structured_weather \
    --message "What is the weather in London?" \
    --json-schema "$SCHEMA_PATH" \
    >"$stdout_path" 2>"$stderr_path"; then
    echo "❌ fast-agent failed for model: $model" >&2
    cat "$stderr_path" >&2
    return 1
  fi

  uv run python - "$stdout_path" "$SCHEMA_PATH" "$model" <<'PY'
import json
import sys
from pathlib import Path

from jsonschema import validate

output_path = Path(sys.argv[1])
schema_path = Path(sys.argv[2])
model = sys.argv[3]

payload = json.loads(output_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))
validate(payload, schema)

assert payload["city"] == "London", (model, payload)
assert payload["condition"] == "light rain", (model, payload)
assert payload["temperature_c"] == 12, (model, payload)
assert payload["summary"] == "London is cool with light rain.", (model, payload)
PY

  echo "[structured-cli]      OK: $model"
}

for model in "${MODELS[@]}"; do
  run_one "$model"
done

echo "✅ structured CLI json-schema smoke passed"
if [[ "$KEEP_TMP" == "1" ]]; then
  echo "[structured-cli] tmp kept at: $WORK_DIR"
fi
