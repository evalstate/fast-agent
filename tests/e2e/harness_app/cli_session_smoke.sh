#!/usr/bin/env bash
set -euo pipefail

KEEP_TMP="${KEEP_TMP:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-fa_harness_app_smoke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the harness app CLI session smoke" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for the harness app CLI session smoke" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fa-harness-app-smoke-XXXXXX")"
ENV_DIR="$WORK_DIR/env"
CAPTURE_CREATE="$WORK_DIR/create.capture.txt"
CAPTURE_RESUME="$WORK_DIR/resume.capture.txt"
STDOUT_ONE_SHOT="$WORK_DIR/one-shot.stdout.txt"
STDERR_ONE_SHOT="$WORK_DIR/one-shot.stderr.txt"

cleanup() {
  tmux kill-session -t "${SESSION_PREFIX}_create" 2>/dev/null || true
  tmux kill-session -t "${SESSION_PREFIX}_resume" 2>/dev/null || true
  if [[ "$KEEP_TMP" != "1" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

fail_with_capture() {
  local message="$1"
  local capture_path="$2"
  echo "$message" >&2
  if [[ -f "$capture_path" ]]; then
    echo "--- capture: $capture_path ---" >&2
    cat "$capture_path" >&2
  fi
  exit 1
}

wait_for_capture() {
  local session="$1"
  local pattern="$2"
  local capture_path="$3"
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    tmux capture-pane -p -t "$session" >"$capture_path"
    if grep -Fq "$pattern" "$capture_path"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

send_line() {
  local session="$1"
  local line="$2"
  tmux send-keys -l -t "$session" "$line"
  tmux send-keys -t "$session" Enter
}

echo "[harness-app-smoke] one-shot CLI"
(
  cd "$REPO_ROOT"
  uv run fast-agent go \
    --env "$ENV_DIR" \
    --model passthrough \
    --message "harness app one-shot" \
    --quiet \
    >"$STDOUT_ONE_SHOT" 2>"$STDERR_ONE_SHOT"
)

if [[ "$(tr -d '\r\n' <"$STDOUT_ONE_SHOT")" != "harness app one-shot" ]]; then
  echo "one-shot CLI output did not match passthrough response" >&2
  echo "--- stdout ---" >&2
  cat "$STDOUT_ONE_SHOT" >&2
  echo "--- stderr ---" >&2
  cat "$STDERR_ONE_SHOT" >&2
  exit 1
fi

echo "[harness-app-smoke] interactive session create/list"
tmux kill-session -t "${SESSION_PREFIX}_create" 2>/dev/null || true
tmux new-session -d -s "${SESSION_PREFIX}_create" -x 120 -y 40 \
  "cd '$REPO_ROOT' && uv run fast-agent go --env '$ENV_DIR' --model passthrough"
tmux set-option -t "${SESSION_PREFIX}_create" status off >/dev/null

wait_for_capture "${SESSION_PREFIX}_create" "Use '/' for commands" "$CAPTURE_CREATE" \
  || fail_with_capture "interactive fast-agent did not reach prompt" "$CAPTURE_CREATE"

send_line "${SESSION_PREFIX}_create" "/session new harness-smoke"
wait_for_capture "${SESSION_PREFIX}_create" "Created session: harness-smoke" "$CAPTURE_CREATE" \
  || fail_with_capture "session creation was not shown" "$CAPTURE_CREATE"

send_line "${SESSION_PREFIX}_create" "harness app first turn"
wait_for_capture "${SESSION_PREFIX}_create" "harness app first turn" "$CAPTURE_CREATE" \
  || fail_with_capture "first turn was not shown" "$CAPTURE_CREATE"
wait_for_capture "${SESSION_PREFIX}_create" "Last turn:" "$CAPTURE_CREATE" \
  || fail_with_capture "first turn did not complete" "$CAPTURE_CREATE"

send_line "${SESSION_PREFIX}_create" "/session list"
wait_for_capture "${SESSION_PREFIX}_create" "Sessions:" "$CAPTURE_CREATE" \
  || fail_with_capture "session list was not shown" "$CAPTURE_CREATE"

send_line "${SESSION_PREFIX}_create" "/exit"
sleep 1
tmux kill-session -t "${SESSION_PREFIX}_create" 2>/dev/null || true

HISTORY_PATH=""
while IFS= read -r candidate_history; do
  if grep -Fq "harness app first turn" "$candidate_history"; then
    HISTORY_PATH="$candidate_history"
    break
  fi
done < <(find "$ENV_DIR/sessions" -type f -name 'history_*.json')
if [[ -z "$HISTORY_PATH" ]]; then
  fail_with_capture "could not find persisted history for the created session" "$CAPTURE_CREATE"
fi

SESSION_DIR="$(dirname "$HISTORY_PATH")"
SESSION_ID="$(basename "$SESSION_DIR")"
echo "[harness-app-smoke] created session id: $SESSION_ID"

echo "[harness-app-smoke] root --resume preview"
tmux kill-session -t "${SESSION_PREFIX}_resume" 2>/dev/null || true
tmux new-session -d -s "${SESSION_PREFIX}_resume" -x 120 -y 40 \
  "cd '$REPO_ROOT' && ENVIRONMENT_DIR='$ENV_DIR' uv run fast-agent --resume '$SESSION_ID'"
tmux set-option -t "${SESSION_PREFIX}_resume" status off >/dev/null

wait_for_capture "${SESSION_PREFIX}_resume" "Last assistant message" "$CAPTURE_RESUME" \
  || fail_with_capture "resume startup did not display last assistant heading" "$CAPTURE_RESUME"
grep -Fq "harness app first turn" "$CAPTURE_RESUME" \
  || fail_with_capture "resume startup did not display previous assistant text" "$CAPTURE_RESUME"

send_line "${SESSION_PREFIX}_resume" "harness app resumed turn"
wait_for_capture "${SESSION_PREFIX}_resume" "harness app resumed turn" "$CAPTURE_RESUME" \
  || fail_with_capture "resumed turn was not shown" "$CAPTURE_RESUME"

send_line "${SESSION_PREFIX}_resume" "/exit"
sleep 1
tmux kill-session -t "${SESSION_PREFIX}_resume" 2>/dev/null || true

if ! grep -Fq "harness app resumed turn" "$SESSION_DIR/history_agent.json"; then
  fail_with_capture "resumed turn was not persisted to the original session" "$CAPTURE_RESUME"
fi

echo "harness app CLI session smoke passed"
if [[ "$KEEP_TMP" == "1" ]]; then
  echo "[harness-app-smoke] tmp kept at: $WORK_DIR"
fi
