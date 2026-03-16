# LLM Trace Fixtures

This directory holds the capture, promotion, and replay workflow for provider
streaming traces.

## Layout

- `manifests/trace_matrix.json`
  - capture-time model/scenario matrix used by the harvester
- `manifests/replay_cases.json`
  - replay-time source of truth for committed fixtures
- `scenarios/`
  - prompt files for repeatable capture scenarios
- `raw/`
  - local raw artifacts from live runs
  - gitignored on purpose
- `sanitized/`
  - curated, stable replay fixtures committed to the repo

Replay fixtures live under stable family/model/scenario directories:

```text
tests/fixtures/llm_traces/sanitized/
  responses/
    gpt-5-mini/
      tool_use_weather/
        meta.json
        request.json
        stream.jsonl
        listener_stream.jsonl
        listener_tools.jsonl
        result.json
```

## 1. Harvest a raw trace

Run from the repo root:

```bash
uv run python tests/scripts/harvest_llm_traces.py --list
uv run python tests/scripts/harvest_llm_traces.py --model 'responses.gpt-5-mini?reasoning=low'
```

The harvester:

- enables `FAST_AGENT_LLM_TRACE=1`
- runs one model/scenario at a time
- saves final result JSON
- saves listener-level stream/tool events
- moves provider-native `stream-debug/` artifacts into the run directory
- adds Google-native request/chunk capture via a temporary wrapper

## 2. Promote a raw run into a sanitized replay fixture

```bash
uv run python tests/scripts/normalize_llm_traces.py \
  tests/fixtures/llm_traces/raw/responses.gpt-5-mini?reasoning=low/tool_use_weather/20260314T230520Z
```

By default, the normalizer promotes into a stable destination under
`sanitized/<family>/<model-label>/<scenario>/` and standardizes the committed
filenames:

- `stream.jsonl`
- `request.json`
- `listener_stream.jsonl`
- `listener_tools.jsonl`
- `meta.json`
- `result.json`

Use `--model-label` when you want the committed grouping label to differ from
the raw target name.

Normalization is for fixture stability, not blind archival. Review output before
committing anything under `sanitized/`.

## 3. Add a replay manifest entry

Add one case to `tests/fixtures/llm_traces/manifests/replay_cases.json`.

Each case declares:

- stable `id`
- `family`
- `model_label`
- `scenario`
- `fixture_dir`
- `assertion_profile`

## 4. Run the replay tests

```bash
uv run pytest tests/unit/fast_agent/llm/providers/test_responses_stream_replay.py
uv run pytest tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py
uv run pytest tests/unit/fast_agent/llm/providers/test_google_stream_replay.py
```

## Notes

- Raw captures may contain prompts, answers, URLs, citations, and provider ids.
- Do not commit raw runs blindly.
- `codexspark` is included as a plain-text-only target because it is text-only.
- `kimi25` and `qwen35` are intentionally included even though they use the
  older OpenAI chat-completions path; they belong to a separate replay lane from
  the Responses-family processors.
