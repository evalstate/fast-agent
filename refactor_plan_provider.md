# Refactor Plan: Provider / Streaming Pipeline Work

Date: 2026-03-14

## Purpose

This track covers large provider-facing completion and streaming functions.

These refactors are important, but they are easy to get wrong by introducing
premature cross-provider abstractions. The goal is to make each provider flow
clearer while keeping provider-specific behavior local and explicit.

## Current scope note

For this refactor pass, **Bedrock is out of scope**.

We may keep Bedrock in the inventory as a known hotspot, but we are **not**
planning to change:

- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_bedrock_completion`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_process_stream`

Reason:

- Bedrock has a high-complexity, provider-specific fallback path
- it is easier to destabilize than the other provider targets
- we want to prove the refactor shape on safer provider surfaces first

## Core principle

Aim for a **shared shape**, not a shared framework.

Good:

- each provider having similar phases
- small typed request/attempt structures
- helper extraction inside the provider module/class

Bad:

- a generic provider pipeline abstraction introduced too early
- forcing Bedrock, Anthropic, OpenAI, and Google into the same helper API when
  their quirks differ materially

## Shared phase model

Most provider completion paths can be understood as:

1. request param resolution
2. message/tool preparation
3. provider-specific feature toggles
4. request argument assembly
5. stream or request execution
6. stream event processing
7. response/tool-call conversion
8. usage/error finalization

That phase model should guide extraction, but it does **not** require a common
implementation.

## Primary targets

### High priority

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py::_anthropic_completion`

### Medium priority follow-ons

- `src/fast_agent/llm/provider/openai/responses_streaming.py::_process_stream`
- `src/fast_agent/llm/provider/google/llm_google_native.py::_consume_google_stream`

### Explicitly deferred

- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_bedrock_completion`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py::_process_stream`

## 1. `_anthropic_completion(...)`

Current complexity: `73`

### Why this matters

This function already has visible internal phases, which makes it a good
candidate for incremental extraction rather than a redesign.

Current concerns mixed together:

- client setup
- request param/message building
- structured output decisions
- tool and web-tool preparation
- thinking configuration
- beta flag assembly
- cache planning / cache-control application
- stream invocation
- response / usage post-processing

### Preferred shape

Candidate helper boundaries:

- `_build_anthropic_client(...)`
- `_build_anthropic_request_messages(...)`
- `_prepare_anthropic_request_tools(...)`
- `_build_anthropic_base_args(...)`
- `_apply_anthropic_structured_output(...)`
- `_resolve_anthropic_beta_flags(...)`
- `_apply_anthropic_cache_plan(...)`
- `_execute_anthropic_stream(...)`
- `_finalize_anthropic_response(...)`

Optional small typed structure:

- `AnthropicRequestPlan`

Potential fields:

- `model`
- `messages`
- `request_tools`
- `structured_mode`
- `base_args`
- `beta_flags`

### Avoid

- moving Anthropic-specific thinking/cache behavior into generic helpers
- hiding beta-flag logic in a way that makes feature activation harder to audit

### End state

- phase-oriented and readable
- ideally below `30`, but clarity matters more than chasing a tiny number

## Deferred: `_bedrock_completion(...)`

Current complexity: `120`

### Why this matters

This is one of the hardest provider surfaces because it mixes request assembly
with fallback strategy and message repair behavior.

Current concerns mixed together:

- credential/client handling
- history/message conversion
- tool schema selection and fallback order
- tool name mapping
- schema-specific tool payload construction
- system prompt mode and model-specific nudges
- tool result / tool use pairing repair
- per-attempt request assembly
- response execution and finalization

### Preferred strategy

Refactor in two layers:

#### Layer 1: attempt planning

Extract helpers for:

- `_resolve_bedrock_schema_order(...)`
- `_build_tool_payload_for_schema(...)`
- `_build_system_text_for_schema(...)`
- `_build_bedrock_attempt_args(...)`

Optional small typed structure:

- `BedrockAttemptPlan`

Potential fields:

- `schema_choice`
- `tool_payload`
- `tool_name_mapping`
- `system_text`
- `system_mode`
- `converse_args`

#### Layer 2: message integrity and execution

Extract helpers for:

- `_detect_tool_message_pairing_state(...)`
- `_reconstruct_missing_tool_use_messages(...)`
- `_execute_bedrock_attempt(...)`
- `_finalize_bedrock_response(...)`

### Important design note

The schema fallback loop is a real phase boundary and should stay visible in the
top-level orchestration. Do not hide it behind a generic “retry framework.”

### Avoid

- unifying Bedrock and Anthropic request builders
- moving model-specific nudges into a global utility without a clear ownership
  rule
- burying tool-pairing repair logic where it becomes hard to test

### End state

- clear attempt loop
- clear per-attempt builder
- clear response finalization path

## 3. Streaming processors

Targets:

- `responses_streaming.py::_process_stream`
- `llm_bedrock.py::_process_stream`
- `llm_google_native.py::_consume_google_stream`

### Why these matter

These are the provider functions most likely to benefit from a more explicit
event model. Their complexity often comes from:

- many event kinds
- partial accumulation
- tool-call assembly
- usage/tracing bookkeeping

### Preferred shape

For these functions, an explicit reducer-style structure can help:

- parse event
- classify event
- update accumulated stream state
- emit side effects only at obvious boundaries

Possible local structures:

- `StreamAccumulator`
- `ToolCallAccumulator`
- `StreamUsageState`

### Avoid

- a repo-wide streaming framework unless multiple providers truly converge on
  the same event contract

## State-machine guidance for provider code

Selective use of state machines can help here, but only for the right
problems.

### Good candidates for an explicit state machine or reducer

- streaming event processors
- Bedrock tool-use/tool-result pairing repair
- request-attempt fallback sequences where invalid transitions are easy to make

### Poor candidates for a full state machine

- mostly linear request assembly
- straightforward argument merging
- feature-flag assembly that is clearer as named helper phases

## Decision rules for this track

Use these questions:

1. Is the main risk invalid event/attempt transitions?
2. Is the function mostly linear assembly, or is it processing a stream of
   events?
3. Would a local typed plan/accumulator make the flow obvious?
4. Would a shared abstraction actually reduce duplication, or just move it?

If `(1)` is yes and `(2)` is “stream/event-heavy,” a reducer or small state
machine may help.

If `(2)` is “mostly linear assembly,” prefer phase helpers instead.

## Validation

Protect provider refactors with:

- existing provider-specific tests
- focused unit tests for pure helper logic
- integration-style tests where available
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run scripts/cpd.py --check`

Do not use mocks or monkeypatches.

## Exit criteria

This track is in good shape when:

- provider completion functions read as explicit phases
- stream processors have clear accumulation rules
- provider-specific quirks remain easy to find and reason about
- any state-machine use is narrow, local, and justified by actual transition
  complexity
