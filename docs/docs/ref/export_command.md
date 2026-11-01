---
title: fast-agent export command
description: Export persisted fast-agent session traces locally or to Hugging Face
  Hub storage.
social:
  title: Export Sessions
  tagline: Export persisted session traces locally or to Hugging Face Hub storage.
  description: Export persisted session traces locally or to Hugging Face Hub storage.
  alt: fast-agent social card — Export Sessions
---


# `fast-agent export` command

Use `fast-agent export` to export a persisted session as Codex-style JSONL or
an [ATIF v1.8](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
JSON trajectory. You can write the trace locally, upload it to a Hugging Face
URL, or do both in one step.

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
| `--format`, `--export-format codex\|atif` | Select Codex JSONL or ATIF v1.8 JSON. Defaults to `codex`. |
| `--hf-url <hf://...>` | Upload the exported trace to a Hugging Face URL. Supports `hf://buckets/...` and `hf://datasets/...`. |
| `--hf-dataset <owner/name>` | Compatibility option for uploading to a Hugging Face dataset repo. Prefer `--hf-url` for new workflows. |
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

# Export one Harbor-compatible ATIF document, including embedded subagents
fast-agent export latest --format atif --output trajectory.json

# Upload the latest session trace to a Hugging Face URL
fast-agent export latest --hf-url hf://buckets/your-name/fast-agent-traces/

# Upload to a Hugging Face dataset URL
fast-agent export latest --hf-url hf://datasets/your-name/fast-agent-traces/trace.jsonl

# Compatibility: upload into a folder in a dataset repo
fast-agent export latest \
  --hf-dataset your-name/fast-agent-traces \
  --hf-dataset-path evals/

# Export a privacy-filtered trace
fast-agent export latest --privacy-filter --output sanitized-trace.jsonl

# First privacy-filter run: allow the model download explicitly
fast-agent export latest --privacy-filter --download-privacy-filter
```

## Behavior

- The default export format is `codex` JSONL. `atif` writes a single ATIF-v1.8
  JSON document and embeds recorded subagent trajectories with resolvable IDs.
- `--format atif` is pinned to ATIF v1.8. Future schema versions require an
  explicit format implementation; existing ATIF output will not silently
  change versions.
- ATIF exports inline image and audio content as `data:` URIs in the content
  part's `source.path`. Audio MIME aliases such as `audio/mp3` and `audio/x-wav`
  are normalized to the canonical ATIF types; media types ATIF cannot represent
  are omitted.
- If a stream gets interrupted, fast-agent keeps the usage recorded for completed
  attempts. Harbor's updated adapter includes that usage in run totals, with a
  note that the interrupted attempt's usage is unknown.
- If `--output` is omitted, fast-agent writes
  `{session_id}__{agent_name}__codex.jsonl` in the current working directory.
- The corresponding default ATIF filename is
  `{session_id}__{agent_name}__atif.json`.
- If `--privacy-filter` is enabled and `--output` is omitted, fast-agent writes
  `{session_id}__{agent_name}__codex-privacy.jsonl`.
- If the session has multiple exportable agent histories, pass `--agent`.
- If `git_aware: true` was enabled while the session was saved, the
  `session_meta` record includes a `git` block with the repository state
  captured at the start of the session and at the latest save. This includes
  the repo root, commit, capture time, branch, dirty state, GitHub `owner/repo`
  when available, and a sanitized `origin` remote URL.
- `--hf-url` is the preferred Hugging Face upload option. If the URL ends with
  `/`, fast-agent appends the local filename.
- `--hf-dataset-path` requires `--hf-dataset`.
- If `--hf-dataset-path` ends with `/`, it is treated as a folder and fast-agent
  appends the local filename.
- Uploads require `huggingface_hub`.
- Dataset repos are created automatically when needed.
- `--privacy-filter` requires the optional `privacy` extra and applies before
  local write/upload. See the [privacy filter guide](../guides/privacy_filter/).
- Privacy filtering redacts trace text content, not structured provenance fields
  such as `session_meta.git`; review git-aware exports before sharing if paths,
  branch names, or repository names are sensitive.
- By default, privacy filtering uses a cached model only. Add
  `--download-privacy-filter` to allow the initial model download.
- `--no-home` runs do not persist sessions, so there is nothing to export later.

## PTC tool-call metadata in ATIF

For programmatic tool calling (PTC), the parent `execute_python` call still contains
the source program. Its observation's `extra` also carries the intercepted nested
tool calls from result metadata (`fast-agent-ptc-calls`):

```json
{
  "ptc_calls_schema": "fast-agent.ptc-calls/v1",
  "ptc_calls": [
    {
      "id": "parent::ptc-1",
      "tool": "read_text_file",
      "arguments": {"path": "/work/dynamically-selected.txt"},
      "is_error": false,
      "status": "completed"
    }
  ]
}
```

- `id`, `tool`, and `is_error` retain their existing meanings; IDs are opaque.
  Existing nested subagent metadata is retained alongside these fields.
- `arguments` is a detached JSON snapshot of the resolved argument dictionary at
  dispatch, including dynamically constructed values. Records are appended before
  awaiting dispatch, in invocation order, not parallel completion order. JSON
  normalization applies (for example, Python tuples become arrays).
- `status: "completed"` means a tool returned, **not** that it succeeded:
  inspect `is_error`. A returned running-process handle also counts as completed
  dispatch, not completed process execution. `"raised"` means dispatch raised an
  exception (`is_error: true`). `"in_flight"` and `"cancelled"` have
  `is_error: null`; neither asserts success or absence of side effects. Turn
  cancellation can prevent a parent result from being saved at all.
- Arguments that cannot be JSON-serialized are represented by `arguments: null`
  and `arguments_error: "not_json_serializable"`. Capture failure does not block
  dispatch; no stringified substitute is recorded.
- An empty array means that execution recorded no intercepted tool dispatches.
  Older results may omit the ledger, or contain entries without `arguments` and
  `status`; export does not reconstruct them. The schema marker describes this
  metadata envelope, not a guarantee of capture completeness.

This is a tool-invocation ledger, **not a complete execution audit**. It does not
include full nested tool results: PTC deliberately filters those inside Python.
Metadata is not placed in model-visible result text and is not evidence that the
model saw a nested result. Direct Python filesystem pass-through (`open()`,
`pathlib`, `os` file functions) bypasses tool dispatch and tool permission prompts,
so those reads/writes do not appear here. Use intercepted file tools when their
invocations need to be recorded.

Like ordinary tool arguments, snapshots can contain secrets, file contents,
paths, or commands, even when the program does not print them. Protect saved
history and review exports before sharing. ATIF `--privacy-filter` processes
nested argument values just like ordinary tool argument values; it is not a
guarantee that all secrets are removed.

## Interactive equivalent

Inside the interactive prompt, use `/session export`:

```text
/session export latest --output trace.jsonl
/session export latest --format atif --output trajectory.json
/session export latest --hf-url hf://buckets/your-name/fast-agent-traces/
/session export latest --hf-url hf://datasets/your-name/fast-agent-traces/trace.jsonl
/session export latest --privacy-filter
/session export latest --privacy-filter --download-privacy-filter
```
