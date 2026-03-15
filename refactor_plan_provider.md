# Refactor Plan: Provider / Streaming Pipeline Work

Date: 2026-03-14

## Purpose

This track covers large provider-facing completion and streaming functions.

The primary goal is to **reduce cyclomatic complexity without hiding provider
behavior**. The code should become:

- easier to read top-to-bottom
- easier to test in focused pieces
- easier to reason about by inspection
- no less explicit about provider quirks

This is not a “make all providers look the same” exercise. It is a
“make each provider flow obvious, and share only the parts that are genuinely
the same” exercise.

## Primary design rule

Aim for **clear orchestration + small named helpers + typed state**.

Good:

- top-level methods that read like phase orchestration
- small helpers at real phase boundaries
- typed accumulators instead of parallel dict/set/list mutation
- shared utilities where the domain behavior is truly stable

Bad:

- generic provider frameworks introduced ahead of proven need
- cross-provider event enums that hide SDK differences
- “one abstraction to rule them all” for request building or stream dispatch
- large helper objects that only rename complexity

## Correct-by-inspection target

After refactor, the important functions should be understandable without
scrolling through large mutable dict-of-dict state.

Concretely:

1. **Top-level methods orchestrate.**
   They should mostly read as: prepare state → iterate events/steps →
   finalize response.
2. **Mutable stream state lives in one named place.**
   Prefer a tracker or typed accumulator over four parallel collections.
3. **Provider-specific dispatch stays local.**
   The loop that classifies Anthropic/OpenAI/Google events should remain in the
   provider module.
4. **Shared code earns its place.**
   Share code only when at least two concrete call sites already exist and the
   abstraction makes both simpler.

## Scope

### In scope

- `src/fast_agent/llm/provider/google/llm_google_native.py::_consume_google_stream`
- `src/fast_agent/llm/provider/openai/responses_streaming.py::_process_stream`
- `src/fast_agent/llm/provider/openai/openresponses_streaming.py::_process_stream`
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_process_stream`
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_anthropic_completion`

### Explicitly out of scope

- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_bedrock_completion`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_process_stream`
- `src/fast_agent/llm/provider/openai/llm_openai.py::_process_stream`
  - this is the Chat Completions path, not the OpenAI Responses-family path
  - it can be handled in a separate pass if needed

## Why Bedrock stays deferred

Bedrock mixes request assembly, schema fallback, repair logic, and execution in
ways that are materially more fragile than the in-scope providers.

It should be refactored later, after the stream-shape and helper boundaries are
proven on safer surfaces.

## Inventory and current coverage

| Target | Lines | File total | Direct test coverage |
|---|---:|---:|---|
| Google `_consume_google_stream` | 178 | 696 | None/minimal |
| OpenAI Responses `_process_stream` | 320 | 446 | Minimal |
| OpenAI OpenResponses `_process_stream` | 263 | 435 | Minimal |
| Anthropic `_process_stream` | 280 | 1729 | Minimal |
| Anthropic `_anthropic_completion` | 412 | 1729 | Low — mostly sub-concern coverage |
| (deferred) Bedrock `_bedrock_completion` | — | 2437 | — |

## What we should share

The right shared abstractions here are **small utilities and small state
objects**, not a cross-provider pipeline superclass.

### 1. Cross-provider: `ToolCallTracker`

This is the one cross-provider state object that clearly exists today.

Anthropic, Google, and the OpenAI Responses family all track:

- tool registration
- lookup by stream identity
- “has start been emitted?”
- close / incomplete detection

They currently do that with scattered dict and set mutation.

#### Recommended location

Create a small dedicated module:

- `src/fast_agent/llm/tool_tracking.py`

This is cleaner than expanding `stream_types.py`, which currently only holds
`StreamChunk`.

#### Recommended shape

```python
ToolKind = Literal["tool", "server_tool", "web_search"]


@dataclass(slots=True)
class ToolCallState:
    tool_use_id: str
    name: str
    kind: ToolKind = "tool"
    index: int | None = None
    start_notified: bool = False


class ToolCallTracker:
    """Track open and completed tool calls for a single provider stream."""

    def register(
        self,
        *,
        tool_use_id: str,
        name: str,
        index: int | None = None,
        kind: ToolKind = "tool",
    ) -> ToolCallState: ...

    def resolve_open(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
    ) -> ToolCallState | None: ...

    def close(
        self,
        *,
        tool_use_id: str | None = None,
        index: int | None = None,
    ) -> ToolCallState | None: ...

    def incomplete(self) -> list[ToolCallState]: ...
    def completed(self) -> list[ToolCallState]: ...
```

#### Required semantics

These semantics need to be explicit in implementation and tests:

- **Identity is `tool_use_id`.**
  - `index` is a secondary lookup key, not the canonical identity.
- **Registration is idempotent.**
  - if the same tool is registered twice, merge missing fields instead of
    duplicating state
  - if a later registration provides the real index, attach it
- **Closed tools are archived.**
  - `resolve_open()` returns only active tools
  - `completed()` exists for final assembly paths like Google
- **No payload buffering inside the tracker.**
  - argument buffers, preview text, and provider-specific payload assembly stay
    provider-local

That last point is important: the shared concern is lifecycle, not provider
payload accumulation.

### 2. Cross-provider: shared plain-text stream emission helper

There is a small repeated pattern in Google, Anthropic, and the OpenAI
Responses family:

```python
self._notify_stream_listeners(StreamChunk(text=delta, is_reasoning=False))
estimated_tokens = self._update_streaming_progress(delta, model, estimated_tokens)
self._notify_tool_stream_listeners("text", {"chunk": delta})
```

This is worth sharing because it is:

- provider-neutral
- stable
- easy to verify by inspection
- repeated enough to reduce noise in stream loops

#### Recommended location

`fast_agent_llm.py` as a protected helper on the base class.

#### Recommended shape

```python
def _emit_stream_text_delta(
    self,
    *,
    text: str,
    model: str,
    estimated_tokens: int,
) -> int: ...
```

This helper should only cover **plain assistant text**. Reasoning/thinking text
stays provider-local because progress semantics differ.

### 3. Shared within the OpenAI Responses family: tool identity / alias handling

There are two OpenAI Responses-family processors:

- `responses_streaming.py`
- `openresponses_streaming.py`

They both deal with a richer identity model than the other providers:

- stream index
- `tool_use_id`
- provider `item_id`
- status events that may arrive before or after item-added events
- phantom registration for web-search-like events

This should be shared **within the OpenAI family**, but not across all
providers.

#### Recommended shape

Introduce one small OpenAI-family helper built on top of `ToolCallTracker`.

Suggested location:

- `src/fast_agent/llm/provider/openai/tool_stream_state.py`

Suggested responsibilities:

- `item_id -> tool_use_id` alias map
- tool name / tool id extraction from item payloads
- fallback / phantom registration for status-only events
- payload construction for start/status/stop notifications

This can be implemented as either:

- a small helper class, or
- a tight utility module with pure functions plus a tiny alias-map object

Prefer whichever version keeps both call sites shorter and clearer. Do **not**
introduce a deep mixin hierarchy just for this.

## What we should not share

These concerns should remain provider-local.

### Event dispatch

Do not introduce a generic event enum or shared dispatcher.

Why:

- Anthropic uses SDK event classes and `isinstance`
- Responses/OpenResponses use event type strings
- Google iterates `content.parts`

The dispatch logic is the provider API surface. Hiding it makes debugging
harder, not easier.

### Final response assembly

Do not share final response builders across providers.

- Google reconstructs a `GenerateContentResponse`
- Anthropic assembles channels and raw-content side channels
- OpenAI Responses family finalizes via `finalize_stream_response(...)`

The outputs are materially different.

### Request assembly

Do not build a generic provider request planner.

Especially:

- Anthropic request building should be phase helpers, not a shared framework
- Bedrock request building should remain deferred

### Reasoning / thinking handling

Do not force shared reasoning helpers where semantics differ.

- Anthropic “thinking” blocks are not the same as OpenAI reasoning summaries
- Google does not mirror the same structure

Use local helpers if needed, but keep ownership in the provider module.

## Shared phase model

The useful shared shape is conceptual, not structural.

Most provider completion paths fit this sequence:

1. request param resolution
2. message/tool preparation
3. provider feature toggles
4. request argument assembly
5. request/stream execution
6. event processing
7. response/tool-call conversion
8. usage/error finalization

That model should guide naming and extraction, but does **not** justify a
generic implementation.

## Recommended sequencing

Work from lowest-risk shared utility to highest-risk provider function.

### Step 0: introduce shared building blocks first

**Effort: 0.5–1 day**

- Add `ToolCallTracker` with focused unit tests
- Add base-class `_emit_stream_text_delta(...)`
- Add the OpenAI-family tool-state helper if it clearly shortens both OpenAI
  stream processors

#### Required tests

- register / resolve / close happy path
- idempotent re-register
- late index attachment
- close by index vs close by tool id
- incomplete detection
- completed-state visibility after close

### Step 1: Google `_consume_google_stream`

**Effort: 1–1.5 days**

This is still the best first processor target.

#### Refactor shape

- adopt `ToolCallTracker`
- keep argument buffering provider-local
  - e.g. `tool_buffers: dict[str, str]`
- replace tuple timeline entries with small typed entries
  - either dataclasses or named tuples
- extract `_build_google_final_response(...)`

#### Important note

Because final response assembly needs closed tool state, the timeline should
store a stable tool identity (`tool_use_id` or a typed entry), not just a raw
index that disappears on close.

#### Risk

Low. Main subtlety is preserving the current buffer-diff behavior exactly.

### Step 2: OpenAI Responses family

Do these back-to-back while the event model is fresh in mind.

#### Step 2a: `responses_streaming.py::_process_stream`

**Effort: 1–2 days**

- replace `tool_streams`, `tool_streams_by_id`, `closed_tool_ids`,
  `notified_tool_indices` with tracker + OpenAI-family helper state
- remove large inner functions
- preserve phantom registration for status-only web search events

#### Step 2b: `openresponses_streaming.py::_process_stream`

**Effort: 1–1.5 days**

- adopt the same tracker / helper pattern
- preserve `item_id` alias resolution
- preserve stop dedupe when both status and `output_item.done` arrive

#### Risk

Medium-low. The main danger is identity handling, not event branching.

### Step 3: Anthropic `_process_stream`

**Effort: 1.5–2 days**

- adopt `ToolCallTracker`
  - `kind="tool"` for `ToolUseBlock`
  - `kind="server_tool"` for `ServerToolUseBlock`
- keep `thinking_indices` separate
- keep thinking accumulation provider-local
- extract the final-message validation fallback path

#### Risk

Medium. Anthropic shares `event.index` across multiple block kinds, so the
tracker must only be used for actual tool blocks.

### Step 4: Anthropic `_anthropic_completion`

**Effort: 3–5 days**

This remains the largest and riskiest target.

The method mixes too many concerns, but it is still fundamentally a
**linear phase function**, not a state machine.

#### Extraction order

1. `_resolve_anthropic_beta_flags(...)`
   - pure
   - order-sensitive
   - easy first win

2. `_finalize_anthropic_response(...)`
   - large but conceptually isolated
   - owns channel assembly and content reconciliation

3. `_apply_anthropic_cache_plan(...)`
   - handles planner inputs and cache marker application

4. `_execute_anthropic_stream(...)`
   - owns OTel wrapper bypass, awaitable-vs-context-manager handling,
     cancellation, and stream processing call

5. `_build_anthropic_base_args(...)`
   - consolidates request assembly once helper boundaries are stable

#### Typed structure

Do **not** introduce `AnthropicRequestPlan` immediately.

First extract helpers until the stable data boundary is obvious. Then, if it
still improves readability, introduce a small dataclass with the durable fields
that actually cross helper boundaries.

#### Key risk areas

- beta flag ordering
- OTel stream-wrapper bypass
- streamed-text-vs-provider-text reconciliation
- structured output + tool choice interaction
- cache marker placement after provider-argument merge

#### End state

- `_anthropic_completion` should shrink to phase orchestration
- helpers should be individually testable
- no helper should exist just to move code; each should own a real phase

## Stream processor end-state shape

After refactor, stream processors should read roughly like this:

```python
async def _process_stream(self, stream, model, ...):
    tracker = ToolCallTracker()
    reasoning_segments = []
    estimated_tokens = 0

    async for event in stream:
        # Provider-specific dispatch stays here
        ...

        # Shared lifecycle where applicable
        state = tracker.register(...) or tracker.resolve_open(...) or tracker.close(...)

        # Existing listener API stays the same
        self._notify_tool_stream_listeners(...)

    incomplete = tracker.incomplete()
    if incomplete:
        raise RuntimeError(...)

    return self._build_final_response(...)
```

That is the right level of shared shape:

- provider-local dispatch
- shared lifecycle helper
- provider-local finalization

## Acceptance criteria

The refactor is not done when the code is shorter. It is done when behavior is
preserved and the code is easier to verify by inspection.

### Behavioral acceptance

- no change to existing tool-stream event vocabulary:
  - `"start"`, `"delta"`, `"stop"`, `"status"`, `"text"`
- no duplicate start/stop notifications for the same tool
- open tool state at end-of-stream still raises
- Google final response assembly remains byte-for-byte equivalent where
  practical
- Anthropic streamed-text fallback behavior remains intact
- OpenAI fallback notification behavior remains intact

### Readability acceptance

- no large inner helper functions left inside target methods
- no parallel `dict + dict + set + set` state spread across stream loops when a
  named tracker/helper is clearer
- top-level methods read as orchestration, not storage mutation
- shared helpers are small enough to understand without jumping across the repo

### Test acceptance

Add or extend focused unit tests for:

- `ToolCallTracker`
- Google stream tool lifecycle and final assembly
- OpenAI Responses status-only / phantom tool paths
- OpenResponses `item_id` alias paths
- Anthropic server-tool start/stop without delta
- Anthropic final message validation fallback
- Anthropic completion beta ordering and streamed-text reconciliation

### Repo acceptance

After code changes:

- `uv run scripts/lint.py`
- `uv run scripts/typecheck.py`

Run targeted tests for the touched providers before the full suite if needed.

## Deferred: Bedrock

Bedrock still matters, but it should be a follow-up track.

The likely shape remains:

### Layer 1: attempt planning

- `_resolve_bedrock_schema_order(...)`
- `_build_tool_payload_for_schema(...)`
- `_build_system_text_for_schema(...)`
- `_build_bedrock_attempt_args(...)`

### Layer 2: integrity + execution

- `_detect_tool_message_pairing_state(...)`
- `_reconstruct_missing_tool_use_messages(...)`
- `_execute_bedrock_attempt(...)`
- `_finalize_bedrock_response(...)`

### Bedrock design rule

Keep the fallback loop visible at top level. Do not hide it behind a generic
retry framework.

## Decision rules for this track

Use these questions when deciding whether to extract or share code:

1. Is the shared behavior a real domain concept, or just a coincidence of two
   implementations?
2. Does the helper make the caller shorter **and** more obvious?
3. Would a typed accumulator be clearer than parallel mutable collections?
4. Is this cross-provider, or only shared within one provider family?
5. If a new abstraction disappeared tomorrow, would the provider code become
   more or less understandable?

If the answer to (5) is “more understandable,” do not add the abstraction.
