#!/usr/bin/env bash
set -euo pipefail

KEEP_TMP="${KEEP_TMP:-0}"
MODELS="${MODELS:-gpt-5.5 opus kimi gemini31pro}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fa-attach-media-cli-XXXXXX")"
if [[ "$KEEP_TMP" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

ASSET_PATH="$REPO_ROOT/tests/e2e/multimodal/image.png"
PDF_ASSET_PATH="$REPO_ROOT/tests/e2e/multimodal/sample.pdf"
YOUTUBE_URL="${YOUTUBE_URL:-https://www.youtube.com/watch?v=dQw4w9WgXcQ}"

echo "[attach-media-cli] asset: $ASSET_PATH"
echo "[attach-media-cli] pdf asset: $PDF_ASSET_PATH"
echo "[attach-media-cli] youtube url: $YOUTUBE_URL"
echo "[attach-media-cli] models: $MODELS"

for MODEL in $MODELS; do
  MODEL_SLUG="$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '-')"
  SCENARIOS="image"
  case "$MODEL" in
    gemini*|google.gemini*)
      SCENARIOS="image pdf youtube"
      ;;
    gpt-*|responses.*|openai.*|opus|opus*|claude-*|anthropic.*)
      SCENARIOS="image pdf"
      ;;
  esac

  for SCENARIO in $SCENARIOS; do
    PROMPT_PATH="$WORK_DIR/${MODEL_SLUG}.${SCENARIO}.prompt.md"
    STDOUT_PATH="$WORK_DIR/${MODEL_SLUG}.${SCENARIO}.stdout.txt"
    STDERR_PATH="$WORK_DIR/${MODEL_SLUG}.${SCENARIO}.stderr.txt"
    STREAM_DIR="$WORK_DIR/${MODEL_SLUG}.${SCENARIO}.stream-debug"

    case "$SCENARIO" in
      image)
      cat >"$PROMPT_PATH" <<EOF
Use the attach_media tool to attach this local image:

$ASSET_PATH

After the attachment is available, inspect the image and answer with only the most prominent visible object or scene.
EOF
        ;;
      pdf)
        cat >"$PROMPT_PATH" <<EOF
Use the attach_media tool to attach this local PDF:

$PDF_ASSET_PATH

After the attachment is available, inspect the PDF and answer with one short sentence confirming the document was attached.
EOF
        ;;
      youtube)
        cat >"$PROMPT_PATH" <<EOF
Use the attach_media tool to attach this YouTube video URL:

$YOUTUBE_URL

After the attachment is available, answer with one short sentence confirming the video URL was attached.
EOF
        ;;
      *)
        echo "unknown attach_media CLI smoke scenario: $SCENARIO" >&2
        exit 1
        ;;
    esac

    echo "[attach-media-cli] running model: $MODEL scenario: $SCENARIO"
    echo "[attach-media-cli] prompt-file: $PROMPT_PATH"
    (
      cd "$REPO_ROOT"
      FAST_AGENT_LLM_TRACE=1 "$PYTHON_BIN" -m fast_agent.cli go \
        --no-home \
        -x \
        --model "$MODEL" \
        --prompt-file "$PROMPT_PATH" \
        >"$STDOUT_PATH" 2>"$STDERR_PATH"
    )

    if [[ -d "$REPO_ROOT/stream-debug" ]]; then
      mkdir -p "$STREAM_DIR"
      find "$REPO_ROOT/stream-debug" -maxdepth 1 -type f -print0 \
        | xargs -0 -r -I{} mv "{}" "$STREAM_DIR/"
    fi

    if ! grep -qiE 'tool result|Staged .* as (embedded|linked)' "$STDOUT_PATH"; then
      echo "attach_media CLI smoke failed for model '$MODEL' scenario '$SCENARIO': expected attach_media tool result in stdout" >&2
      echo "--- stdout ---" >&2
      cat "$STDOUT_PATH" >&2
      echo "--- stderr ---" >&2
      cat "$STDERR_PATH" >&2
      exit 1
    fi

    case "$SCENARIO" in
      image)
        if ! grep -qi 'as embedded image/png' "$STDOUT_PATH"; then
          echo "attach_media CLI smoke failed for model '$MODEL': expected local PNG attachment" >&2
          echo "--- stdout ---" >&2
          cat "$STDOUT_PATH" >&2
          echo "--- stderr ---" >&2
          cat "$STDERR_PATH" >&2
          exit 1
        fi
        ;;
      pdf)
        if ! grep -qi 'as embedded application/pdf' "$STDOUT_PATH"; then
          echo "attach_media CLI smoke failed for model '$MODEL': expected local PDF attachment" >&2
          echo "--- stdout ---" >&2
          cat "$STDOUT_PATH" >&2
          echo "--- stderr ---" >&2
          cat "$STDERR_PATH" >&2
          exit 1
        fi
        ;;
      youtube)
        if ! grep -qi 'Staged .* as linked video/mp4' "$STDOUT_PATH"; then
          echo "attach_media CLI smoke failed for model '$MODEL': expected YouTube video attachment" >&2
          echo "--- stdout ---" >&2
          cat "$STDOUT_PATH" >&2
          echo "--- stderr ---" >&2
          cat "$STDERR_PATH" >&2
          exit 1
        fi
        ;;
    esac

    echo "[attach-media-cli] passed: $MODEL $SCENARIO"
  done
done

echo "attach_media CLI smoke passed for: $MODELS"
if [[ "$KEEP_TMP" == "1" ]]; then
  echo "[attach-media-cli] tmp kept at: $WORK_DIR"
fi
