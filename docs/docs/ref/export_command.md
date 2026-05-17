---
title: fast-agent export command
description: Export persisted fast-agent session traces locally or to Hugging Face datasets.
---

# `fast-agent export` command

Use `fast-agent export` to export a persisted session as a Codex-style JSONL
trace. You can write the trace locally, upload it to a Hugging Face dataset, or
do both in one step.

## Usage

```bash
fast-agent export [OPTIONS] [TARGET]
```

## Targets

`TARGET` can be any of:

- `latest`
- a session id
- a session directory
- a `session.json` snapshot path

If omitted, `fast-agent export` uses the latest persisted session.

## Options

| Option | Description |
| --- | --- |
| `--list` | List recent sessions instead of exporting. Cannot be combined with export options. |
| `--agent`, `-a <name>` | Export a specific agent history from the session. |
| `--output`, `-o <path>` | Write the trace to this file path. Parent directories are created as needed. |
| `--hf-dataset <owner/name>` | Upload the exported trace to a Hugging Face dataset repo. |
| `--hf-dataset-path <path>` | Target file or folder path inside the dataset repo. Requires `--hf-dataset`. |
| `--privacy-filter` | Redact exported text content with the local privacy filter. |
| `--privacy-filter-path <path>` | Use a local OpenAI Privacy Filter model directory. Requires `--privacy-filter`. |
| `--download-privacy-filter` | Download the default privacy-filter model if it is not cached. Requires `--privacy-filter`. |
| `--privacy-filter-device auto\|cpu\|cuda` | Choose the ONNX Runtime device. Defaults to `auto`. Requires `--privacy-filter`. |
| `--privacy-filter-variant q4\|q4f16\|q8\|fp16` | Choose the privacy-filter model variant. Defaults to `q8`. Requires `--privacy-filter`. |
| `--privacy-filter-quant ...` | Alias for `--privacy-filter-variant`. |
| `--show-redactions` | Print detected labels and original snippets to stderr for local review. Requires `--privacy-filter`. |

## Examples

```bash
# List recent persisted sessions
fast-agent export --list

# Export the latest session to a local file
fast-agent export latest --output trace.jsonl

# Export a specific agent from a multi-agent session
fast-agent export 2604201303-x5MNlH --agent dev --output dev-trace.jsonl

# Upload the latest session trace to a Hugging Face dataset
fast-agent export latest --hf-dataset your-name/fast-agent-traces

# Upload into a folder in the dataset repo
fast-agent export latest \
  --hf-dataset your-name/fast-agent-traces \
  --hf-dataset-path evals/

# Export a privacy-filtered trace
fast-agent export latest --privacy-filter --output sanitized-trace.jsonl

# First privacy-filter run: allow the model download explicitly
fast-agent export latest --privacy-filter --download-privacy-filter
```

## Behavior

- The current export format is `codex` JSONL.
- If `--output` is omitted, fast-agent writes
  `{session_id}__{agent_name}__codex.jsonl` in the current working directory.
- If `--privacy-filter` is enabled and `--output` is omitted, fast-agent writes
  `{session_id}__{agent_name}__codex-privacy.jsonl`.
- If the session has multiple exportable agent histories, pass `--agent`.
- `--hf-dataset-path` requires `--hf-dataset`.
- If `--hf-dataset-path` ends with `/`, it is treated as a folder and fast-agent
  appends the local filename.
- Uploads require `huggingface_hub`.
- The Hugging Face dataset repo is created automatically when needed.
- `--privacy-filter` requires the optional `privacy` extra and applies before
  local write/upload. See the [privacy filter guide](../guides/privacy_filter.png).
- By default, privacy filtering uses a cached model only. Add
  `--download-privacy-filter` to allow the initial model download.
- `--noenv` runs do not persist sessions, so there is nothing to export later.

## Interactive equivalent

Inside the interactive prompt, use `/session export`:

```text
/session export latest --output trace.jsonl
/session export latest --hf-dataset your-name/fast-agent-traces
/session export latest --privacy-filter
/session export latest --privacy-filter --download-privacy-filter
```
