# Provider Refactor Handover

Date: 2026-03-15

This file is the handover for the current provider/streaming refactor track.
Read this together with `refactor_plan_provider.md`.

## Short version

The project now has a real replay safety net for the provider stream processors,
and the OpenAI Responses-family stream processors have already had a first
refactor pass.

### Already done

- replay fixture workflow implemented
- replay tests added for:
  - Responses
  - OpenResponses
  - Anthropic
  - Google
- `openresponses` streaming fixed for non-standard SSE event sequences
- shared tool lifecycle tracking added
- shared OpenAI-family tool-state helper added
- shared plain-text stream emission helper added
- first refactor pass completed for:
  - `src/fast_agent/llm/provider/openai/responses_streaming.py`
  - `src/fast_agent/llm/provider/openai/openresponses_streaming.py`

### Best next step

Refactor:

- `src/fast_agent/llm/provider/google/llm_google_native.py::_consume_google_stream`

Then:

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_process_stream`
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_anthropic_completion`

## What the replay system actually proves

The replay fixtures are not just raw provider dumps.

They are harvested from a real:

- `Core(...)`
- `LlmAgent(...)`
- `agent.generate(...)`

call through fast-agent.

For each harvested run we capture:

- provider-native stream events
- fast-agent stream listener output
- fast-agent tool listener output
- final `PromptMessageExtended` result

The replay tests then feed the sanitized provider stream back into the real
provider stream processor and assert against the fast-agent-harvested outputs.

That means replay is asserting:

1. exact fast-agent listener behavior
2. final fast-agent semantic result shape

It is not asserting session/history persistence. That is fine for this refactor
track because the risky logic lives in the provider/completion/streaming layer.

## Important current state

### 1. Replay fixture workflow exists and is in use

Important files:

- `tests/fixtures/llm_traces/README.md`
- `tests/fixtures/llm_traces/manifests/trace_matrix.json`
- `tests/fixtures/llm_traces/manifests/replay_cases.json`
- `tests/scripts/harvest_llm_traces.py`
- `tests/scripts/normalize_llm_traces.py`
- `tests/support/llm_trace_replay.py`

Current committed fixture families:

- `responses/gpt-5-mini/plain_text`
- `responses/gpt-5-mini/tool_use_weather`
- `responses/gpt-5-mini/web_search_capital`
- `openresponses/openai-gpt-oss-120b/plain_text`
- `openresponses/openai-gpt-oss-120b/tool_use_weather`
- `anthropic/sonnet/tool_use_weather`
- `anthropic/sonnet/web_search_capital`
- `google/gemini-2.0-flash/tool_use_weather`

Current replay tests:

- `tests/unit/fast_agent/llm/providers/test_responses_stream_replay.py`
- `tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py`
- `tests/unit/fast_agent/llm/providers/test_google_stream_replay.py`

### 2. OpenResponses is now directly covered

This was an important gap earlier.

OpenResponses was added to:

- trace harvest matrix
- sanitized fixtures
- replay manifest
- replay tests

### 3. OpenResponses streaming has a compatibility fix

Important file:

- `src/fast_agent/llm/provider/openai/openresponses.py`

Important detail:

OpenResponses-compatible backends can emit raw SSE event sequences that the
OpenAI SDK accumulator behind `client.responses.stream(...)` does not tolerate.
The failure mode was an internal SDK `IndexError` before fast-agent saw the
events.

The fix is:

- Responses still uses the SDK stream manager path
- OpenResponses overrides `_response_sse_stream(...)`
- OpenResponses uses `client.responses.create(..., stream=True)` and wraps the
  raw typed SSE events in a thin adapter

Do not casually remove or “simplify away” this distinction unless you have
re-tested against the OpenResponses backend.

### 4. Shared refactor building blocks now exist

Important files:

- `src/fast_agent/llm/tool_tracking.py`
- `src/fast_agent/llm/provider/openai/tool_stream_state.py`
- `src/fast_agent/llm/fastagent_llm.py`

What they do:

#### `ToolCallTracker`

Tracks:

- open tool calls
- completed tool calls
- lookup by `tool_use_id`
- lookup by `index`
- idempotent re-register
- late index attachment

Tests:

- `tests/unit/fast_agent/llm/test_tool_tracking.py`

#### `OpenAIToolStreamState`

Adds OpenAI-family behavior on top of tool lifecycle tracking:

- `item_id` alias handling
- placeholder identity rekeying
- entry metadata for payload building

Tests:

- `tests/unit/fast_agent/llm/providers/test_openai_tool_stream_state.py`

#### `FastAgentLLM._emit_stream_text_delta(...)`

Shared helper for plain assistant text streaming:

- notify stream listeners
- update streaming progress
- emit `"text"` tool-stream event

Use this for plain assistant text only.
Do not force reasoning/thinking through it.

## Files already refactored in this track

### `src/fast_agent/llm/provider/openai/responses_streaming.py`

Changes made:

- replaced parallel dict/set lifecycle bookkeeping with shared tool state
- kept provider-specific event dispatch local
- kept web-search-specific status handling local
- switched plain text emission to `_emit_stream_text_delta(...)`

### `src/fast_agent/llm/provider/openai/openresponses_streaming.py`

Changes made:

- replaced parallel dict bookkeeping with shared tool state
- kept OpenResponses-specific status matching local
- preserved alias resolution behavior
- switched plain text emission to `_emit_stream_text_delta(...)`

## Important behavior invariants

These matter more than line count.

### Must preserve

- tool stream event vocabulary:
  - `"start"`
  - `"delta"`
  - `"status"`
  - `"stop"`
  - `"text"`
- no duplicate start/stop notifications for the same tool
- open tool state at end-of-stream still raises
- fallback tool notifications still work
- OpenResponses status/item-id paths still work
- Responses web-search phantom/status-only paths still work

### Do not do

- do not introduce a generic provider event enum
- do not build a cross-provider streaming superclass
- do not hide provider dispatch logic behind abstraction
- do not move code into helpers unless the caller becomes more obvious

The right shape is still:

- top-level orchestration
- provider-local dispatch
- small shared helpers only where behavior is truly the same

## Important related files

### Core plan / scope

- `refactor_plan_provider.md`

Use that as the canonical list of in-scope functions.

### OpenAI-family runtime files

- `src/fast_agent/llm/provider/openai/responses.py`
- `src/fast_agent/llm/provider/openai/openresponses.py`
- `src/fast_agent/llm/provider/openai/responses_streaming.py`
- `src/fast_agent/llm/provider/openai/openresponses_streaming.py`
- `src/fast_agent/llm/provider/openai/tool_notifications.py`
- `src/fast_agent/llm/provider/openai/tool_stream_state.py`
- `src/fast_agent/llm/provider/openai/streaming_utils.py`

### Remaining in-scope targets

- `src/fast_agent/llm/provider/google/llm_google_native.py`
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`

### Tests worth keeping open while refactoring

- `tests/unit/fast_agent/llm/providers/test_responses_helpers.py`
- `tests/unit/fast_agent/llm/providers/test_responses_stream_replay.py`
- `tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py`
- `tests/unit/fast_agent/llm/providers/test_google_stream_replay.py`
- `tests/unit/fast_agent/llm/test_tool_tracking.py`
- `tests/unit/fast_agent/llm/providers/test_openai_tool_stream_state.py`

## Suggested next steps

## Step 1: refactor Google `_consume_google_stream`

Target:

- `src/fast_agent/llm/provider/google/llm_google_native.py::_consume_google_stream`

Approach:

- adopt `ToolCallTracker`
- keep argument buffering provider-local
- keep final response assembly provider-local
- consider replacing tuple timeline entries with a small typed structure
- use `_emit_stream_text_delta(...)` for plain assistant text if it fits cleanly

Important caution:

- Google final response reconstruction depends on preserving tool-call identity
- do not lose the link between the timeline entry and the final tool payload

Recommended test loop:

```bash
uv run pytest tests/unit/fast_agent/llm/providers/test_google_stream_replay.py -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

## Step 2: refactor Anthropic `_process_stream`

Target:

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_process_stream`

Approach:

- adopt `ToolCallTracker`
- keep thinking/thought handling local
- keep server-tool handling local
- keep provider event dispatch local
- do not over-generalize based on OpenAI-family structure

Important caution:

- Anthropic shares `event.index` across different block kinds
- use the tracker only for actual tool/server-tool lifecycle
- do not let thinking blocks pollute tool state

Recommended test loop:

```bash
uv run pytest tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

## Step 3: refactor Anthropic `_anthropic_completion`

Target:

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_anthropic_completion`

Approach:

Follow the extraction order from `refactor_plan_provider.md`:

1. beta-flag resolution
2. final response assembly
3. cache-plan application
4. stream execution
5. base request assembly

Important caution:

- this is the riskiest remaining target
- do not introduce a request-plan dataclass too early
- extract only real phase boundaries
- preserve:
  - beta ordering
  - streamed-text reconciliation
  - validation fallback behavior
  - structured-output/tool-choice interactions

This should only happen after the lower-risk stream refactors are done.

## Working style that has gone well so far

- make one provider-family move at a time
- keep diffs small
- add/adjust tests before trusting a cleanup
- refactor toward named lifecycle/state concepts, not frameworks
- use replay tests as the regression anchor

## Commands to use repeatedly

Targeted tests:

```bash
uv run pytest tests/unit/fast_agent/llm/test_tool_tracking.py -q
uv run pytest tests/unit/fast_agent/llm/providers/test_openai_tool_stream_state.py -q
uv run pytest tests/unit/fast_agent/llm/providers/test_responses_helpers.py -q
uv run pytest tests/unit/fast_agent/llm/providers/test_responses_stream_replay.py -q
uv run pytest tests/unit/fast_agent/llm/provider/anthropic/test_anthropic_stream_replay.py -q
uv run pytest tests/unit/fast_agent/llm/providers/test_google_stream_replay.py -q
```

Repo checks:

```bash
uv run scripts/lint.py
uv run scripts/typecheck.py
```

## Local OpenResponses note

To reproduce the current OpenResponses fixture harvesting locally, a
gitignored `fastagent.secrets.yaml` at repo root was used with:

```yaml
openresponses:
  api_key: test-key
  base_url: https://coordinated-epa-hydraulic-current.trycloudflare.com/v1
  default_model: openai/gpt-oss-120b
```

That file is not part of the tracked repo state.

## Final recommendation

If resuming from here, do not reopen the Responses/OpenResponses refactor
unless you are fixing a concrete regression.

The best next move is:

1. Google `_consume_google_stream`
2. Anthropic `_process_stream`
3. Anthropic `_anthropic_completion`

That keeps the work aligned with the original plan and uses the new replay net
effectively.
