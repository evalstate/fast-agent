#!/usr/bin/env bash
set -euo pipefail

KEEP_TMP="${KEEP_TMP:-0}"
MODEL="${MODEL:-sonnet}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE_DIR="$SCRIPT_DIR/cli_json_schema_shell"
SCHEMA_PATH="$FIXTURE_DIR/listing_schema.json"

if [[ ! -f "$SCHEMA_PATH" ]]; then
  echo "❌ missing schema fixture: $SCHEMA_PATH" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fa-structured-shell-XXXXXX")"
if [[ "$KEEP_TMP" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

TARGET_DIR="$WORK_DIR/listing-target"
mkdir -p "$TARGET_DIR/nested"
printf 'alpha\n' > "$TARGET_DIR/alpha.txt"
printf '# beta\n' > "$TARGET_DIR/beta.md"

STDOUT_PATH="$WORK_DIR/stdout.json"
STDERR_PATH="$WORK_DIR/stderr.log"

MESSAGE=$(
  cat <<EOF
Use the shell tool to inspect the exact directory at: $TARGET_DIR

Return JSON matching the provided schema with:
- "directory": the exact directory path you inspected
- "entry_count": the number of immediate children in that directory
- "entries": one item per immediate child
- each item must contain:
  - "name": basename only
  - "kind": "file" or "directory"

Only include immediate children, not recursive descendants.
EOF
)

echo "[structured-shell] schema fixture: $SCHEMA_PATH"
echo "[structured-shell] model: $MODEL"
echo "[structured-shell] target dir: $TARGET_DIR"

if ! uv run fast-agent go \
  --noenv \
  -x \
  --model "$MODEL" \
  --message "$MESSAGE" \
  --json-schema "$SCHEMA_PATH" \
  >"$STDOUT_PATH" 2>"$STDERR_PATH"; then
  echo "❌ fast-agent shell structured run failed" >&2
  cat "$STDERR_PATH" >&2
  exit 1
fi

uv run python - "$STDOUT_PATH" "$SCHEMA_PATH" "$TARGET_DIR" <<'PY'
import json
import sys
from pathlib import Path

from jsonschema import validate

output_path = Path(sys.argv[1])
schema_path = Path(sys.argv[2])
target_dir = Path(sys.argv[3])

payload = json.loads(output_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))
validate(payload, schema)

assert payload["directory"] == str(target_dir), payload
assert payload["entry_count"] == 3, payload

entries = payload["entries"]
assert isinstance(entries, list), payload

normalized = {(entry["name"], entry["kind"]) for entry in entries}
assert ("alpha.txt", "file") in normalized, payload
assert ("beta.md", "file") in normalized, payload
assert ("nested", "directory") in normalized, payload
assert len(entries) == 3, payload
PY

cat "$STDOUT_PATH"
echo
echo "✅ structured shell CLI json-schema smoke passed"
if [[ "$KEEP_TMP" == "1" ]]; then
  echo "[structured-shell] tmp kept at: $WORK_DIR"
fi
