# Handover: cyclomatic-complexity follow-up

Date: 2026-03-14

## Current status

This refactor stream has moved well past the original command-parser hotspot.
The major command-surface and prompt-input work now landed includes:

- `src/fast_agent/ui/prompt/parser.py`
  - split into command-family helpers
  - restored `/history rewind` intent parsing
- `src/fast_agent/ui/interactive/command_dispatch.py`
  - split into command-family dispatch helpers
  - agent refresh logic separated from agent-card handling
- `src/fast_agent/ui/prompt/completion_sources.py`
  - command completions split by family
  - kept partial-input handling local instead of forcing reuse of full intent parsers
- test cleanup
  - redundant parser-only assertions trimmed
  - shared command-surface test support moved to `tests/support/command_surface.py`
- prompt-input naming cleanup
  - canonical modules are now:
    - `src/fast_agent/ui/prompt/input.py`
    - `src/fast_agent/ui/prompt/input_runtime.py`
  - compatibility wrappers kept in:
    - `src/fast_agent/ui/prompt/session.py`
    - `src/fast_agent/ui/prompt/session_runtime.py`
- prompt-input decomposition
  - `src/fast_agent/ui/prompt/input.py::get_enhanced_input`
    - reduced to orchestration
  - new cohesive siblings:
    - `src/fast_agent/ui/prompt/input_toolbar.py`
    - `src/fast_agent/ui/prompt/agent_info.py`

Validation already run in this stream:

- `uv run pytest tests/unit/fast_agent/ui`
- targeted command-surface integration suites
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run scripts/cpd.py --check`

## Small follow-up targets still above threshold

These are now small enough to be optional cleanup passes rather than primary
refactor drivers:

- `src/fast_agent/ui/interactive/command_dispatch.py::dispatch_command_payload`
  - current C901: `11`
  - likely one more tiny split/flatten pass if desired
- `src/fast_agent/ui/prompt/completion_sources.py::_model_command_completions`
  - current C901: `14`
  - can likely be finished by splitting `aliases` / `catalog` handling one step further

Do these only if you want to clean up the remaining near-threshold command UI
functions before moving on.

## Recommended next substantial target

### Pick next

- `src/fast_agent/ui/interactive_prompt.py::prompt_loop`
- current C901: `89`

### Why this should be next

This is now the best adjacent target after the `input.py` split:

- `get_enhanced_input(...)` is no longer the right next move
- command parsing/dispatch/completion seams are already much cleaner
- `prompt_loop` is now the dominant UI orchestrator still mixing too many concerns
- it sits directly above the code that was just decomposed, so the new seams should help

### Why not jump elsewhere first

There are larger or comparable hotspots, but most have a worse blast radius:

- `src/fast_agent/core/fastagent.py::run` (`120`)
  - very high-value eventually, but much broader application lifecycle risk
- `src/fast_agent/ui/tool_display.py::show_tool_result` (`30`)
  - meaningful, but more isolated display logic and less central than `prompt_loop`
- `src/fast_agent/ui/streaming.py::_render_current_buffer` (`25`)
  - useful later, but more rendering-state specific
- `src/fast_agent/ui/stream_segments.py::handle_tool_event` (`21`)
  - smaller and more tactical than `prompt_loop`

## Refactor goal for `prompt_loop`

Current `prompt_loop` still mixes:

- prompt acquisition
- pre-dispatch agent refresh handling
- command dispatch result handling
- hash-agent execution
- shell execution
- normal send flow
- interrupt/retry control flow
- buffer-prefill and available-agent mutation

The goal should be to split by **interaction phase**, not invent a new
framework.

## Recommended refactor shape

### Keep

- `prompt_loop(...)` as the public orchestration entrypoint
- `DispatchResult` contract unchanged
- `get_enhanced_input(...)` and `dispatch_command_payload(...)` as already-cleaned lower layers

### Extract

Prefer phase-oriented helpers, for example:

- `_refresh_agents_if_needed(...)`
- `_apply_dispatch_result(...)`
- `_handle_hash_send(...)`
- `_handle_shell_command(...)`
- `_should_continue_after_command(...)`
- `_send_regular_message(...)`

If one of those grows too much, split by the next real seam, not by arbitrary
line count.

### Avoid

- giant mutable context objects unless parameter passing becomes truly unmanageable
- reflective “step registries”
- abstractions that hide the actual prompt flow

## Existing test seam to rely on

These already give useful coverage around the surrounding behavior:

- `tests/unit/fast_agent/ui/test_interactive_prompt_agent_commands.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_refresh.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_hash_send.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_resource_mentions.py`
- `tests/unit/fast_agent/ui/test_agent_completer.py`
- `tests/unit/fast_agent/ui/test_prompt_input_runtime.py`
- `tests/integration/ui/test_command_dispatch_flows.py`
- `tests/integration/test_command_surface_parity.py`

Prefer reusing those seams before adding more tests.

## Secondary target after `prompt_loop`

If `prompt_loop` lands cleanly, the next sensible targets are:

1. `src/fast_agent/ui/tool_display.py::show_tool_result` (`30`)
2. `src/fast_agent/ui/streaming.py::_render_current_buffer` (`25`)
3. `src/fast_agent/ui/stream_segments.py::handle_tool_event` (`21`)

That keeps the work in the UI/rendering layer before escalating to
`fastagent.py::run`.

## Naming/layout note now in effect

The current convention established in this pass:

- reserve `session` for persisted chat/thread concepts
- use `input*` names for prompt-toolkit input collection/runtime modules

This is now reflected in `AGENTS.md`.

## Compatibility cleanup to keep in mind

There are still compatibility wrappers:

- `src/fast_agent/ui/prompt/session.py`
- `src/fast_agent/ui/prompt/session_runtime.py`

Do **not** prioritize deleting them immediately. Keep them until the next
round of UI refactors settles and imports are clearly stable.

## Summary

The best next move is:

1. refactor `interactive_prompt.py::prompt_loop`
2. keep the split phase-oriented and explicit
3. rely on the existing UI unit suite plus command-surface integration tests
4. optionally clean up the small remaining >10 helpers afterward
