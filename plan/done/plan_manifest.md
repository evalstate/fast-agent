# Plan: Manifest-Driven Trace Replay Fixtures

Date: 2026-03-14

## Goal

Build a **manifest-driven replay test system** for provider streaming traces so
that:

- sensitive stream/completion refactors are protected by deterministic tests
- adding a new captured model is mostly **data**, not new test code
- `responses` and `openresponses` can share the same fixture/assertion shape
- raw provider captures stay separate from curated committed fixtures

This work is meant to sit in front of the provider refactor track and provide
the safety net for:

- `responses_streaming.py::_process_stream`
- `openresponses_streaming.py::_process_stream`
- `llm_anthropic.py::_process_stream`
- `llm_google_native.py::_consume_google_stream`

## Non-goals

- Do not build a generic provider runtime abstraction.
- Do not commit raw trace dumps blindly.
- Do not tie tests to timestamped capture directories.
- Do not rely on real-provider e2e tests as the primary regression net.

## Current assumptions

1. **`responses` and `openresponses` are equivalent for now**
   - use the same fixture contract
   - use the same assertion profiles
   - keep separate `family` keys in the manifest so we can split later without
     changing the harness

2. **Provider defaults should be respected**
   - the harvester should not force `maxTokens` unless explicitly asked to
   - only use explicit token overrides when the capture itself is for a
     max-token-specific behavior

3. **Raw captures and replay fixtures are different artifacts**
   - raw captures are local and timestamped
   - replay fixtures are curated, normalized, stable, and committed

## Desired end state

Adding a new model tomorrow should look like:

1. harvest raw trace
2. normalize to a stable fixture directory
3. add one manifest entry
4. run replay tests

No new provider test code should be required for a routine add.

## Directory layout

### Raw captures

Keep raw captures under:

```text
tests/fixtures/llm_traces/raw/
```

Properties:

- timestamped
- gitignored
- may contain provider ids, prompts, outputs, citations, etc.

### Sanitized replay fixtures

Curate normalized fixtures under:

```text
tests/fixtures/llm_traces/sanitized/
  responses/
    <model-label>/
      <scenario>/
  openresponses/
    <model-label>/
      <scenario>/
  anthropic/
    <model-label>/
      <scenario>/
  google/
    <model-label>/
      <scenario>/
  openai-chat/
    <model-label>/
      <scenario>/
```

Examples:

```text
tests/fixtures/llm_traces/sanitized/responses/gpt-5-mini/tool_use_weather/
tests/fixtures/llm_traces/sanitized/responses/gpt-5-mini/web_search_capital/
tests/fixtures/llm_traces/sanitized/anthropic/sonnet/web_search_capital/
tests/fixtures/llm_traces/sanitized/google/gemini-2.0-flash/tool_use_weather/
tests/fixtures/llm_traces/sanitized/openai-chat/kimi25/plain_text/
```

### Manifests

Use:

```text
tests/fixtures/llm_traces/manifests/
```

with:

- `trace_matrix.json`
  - capture-time matrix
- `replay_cases.json`
  - replay-time source of truth

## Replay manifest design

Create a data file like:

```json
{
  "cases": [
    {
      "id": "responses-gpt5mini-tool-use",
      "family": "responses",
      "model_label": "gpt-5-mini",
      "fixture_dir": "responses/gpt-5-mini/tool_use_weather",
      "assertion_profile": "tool_use"
    }
  ]
}
```

### Required fields

- `id`
  - stable test id
- `family`
  - one of:
    - `responses`
    - `openresponses`
    - `anthropic`
    - `google`
    - `openai-chat`
- `model_label`
  - human-readable model grouping label
- `fixture_dir`
  - relative to `tests/fixtures/llm_traces/sanitized/`
- `assertion_profile`
  - reusable assertion contract name

### Optional future fields

- `notes`
- `skip_reason`
- `expected_warning`
- `variant`
  - if we later need to split `responses` vs `openresponses`

## Assertion profiles

Profiles should be reusable and intentionally small.

Initial set:

- `plain_text`
- `tool_use`
- `web_search`

### `plain_text`

Assert:

- final assistant text exists
- no unexpected tool lifecycle
- stream listener output exists where expected

### `tool_use`

Assert:

- final stop/tool-use structure is present
- tool listener sequence matches the captured fixture
- any tool argument deltas reconstruct cleanly

### `web_search`

Assert:

- provider-specific web-search/tool evidence exists
- listener sequence matches fixture
- final response contains expected text/citation/server-tool evidence

## Replay support module

Add a support module:

```text
tests/support/llm_trace_replay.py
```

Responsibilities:

- load replay manifest
- load JSON/JSONL fixture files
- normalize file access for tests
- provide small replay stream/simulator objects per provider family

## Provider-family replay strategy

### Responses / OpenResponses

Use a replay stream that:

- reads sanitized chunk JSONL
- yields event objects with attribute access
- exposes `get_final_response()`

Important:

- `responses` and `openresponses` should share the same loader contract
- they may differ later only in manifest `family`, not test harness structure

### Anthropic

Use a replay stream that:

- reads sanitized jsonl records
- rehydrates beta event models via `.model_validate(...)`
- exposes `get_final_message()`

The final message can be reconstructed from:

- `message_start`
- `content_block_stop` snapshots
- usage/start metadata in the trace

### Google

Use a replay iterator that:

- reads `google_stream_chunks.jsonl`
- rehydrates `types.GenerateContentResponse`
- exposes `aclose()`

## Test file layout

Add replay-focused tests:

```text
tests/unit/fast_agent/llm/providers/test_responses_stream_replay.py
tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py
tests/unit/fast_agent/llm/providers/test_google_stream_replay.py
```

### What these tests should call

- `ResponsesStreamingMixin._process_stream`
- `OpenResponsesStreamingMixin._process_stream`
  - same test shape, different `family` entries once data exists
- `AnthropicLLM._process_stream`
- `GoogleNativeLLM._consume_google_stream`

## Current canonical raw runs to promote first

### Responses

- `responses.gpt-5-mini.low/tool_use_weather/20260314T230520Z`
- `responses.gpt-5-mini.low/web_search_capital/20260314T230523Z`

### Anthropic

- `sonnet/tool_use_weather/20260314T230539Z`
- `sonnet/web_search_capital/20260314T230541Z`

### Google

- `google.gemini-2.0-flash/tool_use_weather/20260314T230558Z`

### Chat-completions lane (future/follow-on)

- `kimi25/plain_text/20260314T230559Z` or newer default-based rerun
- `kimi25/tool_use_weather/20260314T230603Z`
- `qwen35/plain_text/20260314T230604Z`
- `qwen35/tool_use_weather/20260314T230656Z`
- `codexspark/plain_text/20260314T230659Z`

These should not block the first in-scope replay harness.

## Documentation requirement

Document the workflow in `tests/fixtures/llm_traces/README.md` so that adding a
new model tomorrow is obvious.

The README should explicitly describe:

1. harvest raw trace
2. normalize/promote to sanitized fixture
3. add `replay_cases.json` entry
4. run the replay tests

## “Add a model tomorrow” contract

For a new `openresponses` model, the maintainer workflow should be:

1. harvest raw capture
2. normalize into:
   - `tests/fixtures/llm_traces/sanitized/openresponses/<model-label>/<scenario>/`
3. add one `replay_cases.json` entry
4. run tests

That should require no new replay-harness code.

## Sequencing

### Step 1

Promote the current best raw captures into stable sanitized fixture directories.

### Step 2

Create `replay_cases.json`.

### Step 3

Implement `tests/support/llm_trace_replay.py`.

### Step 4

Add Responses replay tests.

### Step 5

Add Anthropic replay tests.

### Step 6

Add Google replay tests.

### Step 7

Only after that, start the provider refactor work.

## Acceptance criteria

This plan is complete when:

- replay fixtures live in stable, model/scenario-based directories
- replay tests are manifest-driven
- `responses` and `openresponses` share the same fixture contract
- adding a new model is mostly data + manifest entry
- the replay harness directly exercises the real provider stream functions
- raw timestamped capture directories are not referenced directly from tests

## Practical note

The current harvester already respects provider defaults unless explicitly
overridden, which is the correct behavior for this workflow.
