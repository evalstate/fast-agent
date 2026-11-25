# Dependency Update Exceptions

This file tracks places where `fast-agent` relies on private/internal APIs or
version-specific workarounds in third-party libraries.

When upgrading any dependency listed below:
1. Re-check whether the referenced private attributes, imports, or behavioral
   assumptions still hold (and run `uv run scripts/typecheck.py` plus the listed
   targeted tests).
2. Re-check whether upstream has fixed the underlying issue or exposed a public
   API so the workaround can be simplified or removed.
3. Update or remove the entry here when the workaround changes.

---

## `prompt-toolkit`

- **Inline toolbar, dynamic menu height, CPR drain, and resize debounce (`src/fast_agent/ui/prompt/input_runtime.py`, `src/fast_agent/ui/prompt/keybindings.py`)**
  - `_configure_inline_toolbar()` sets `renderer.cpr_support = CPR_Support.NOT_SUPPORTED`,
    zeros `renderer._min_available_height`, wraps `renderer.render` to reset
    `renderer._last_screen = None` (repainting from `(0, 0)` with `erase_down`
    inside DEC Mode 2026 synchronized output on `Vt100_Output`), replaces the
    `class:bottom-toolbar` `ConditionalContainer.filter` to drop
    `renderer_height_is_known`, and overrides
    `session._get_default_buffer_control_height` / `current_window.height` so
    `reserve_space_for_menu` only applies when `complete_state.completions` is
    non-empty.
  - `_install_resilient_resize_handler()` wraps `session.app._on_resize` with an
    80 ms debounce, draining `renderer._waiting_for_cpr_futures` and zeroing
    `renderer._min_available_height`.
  - `reset_renderer_cpr_state()` drains `renderer._waiting_for_cpr_futures` and
    optionally resets `renderer.cpr_support`; `_clear_terminal_screen()` resets
    scroll region/SGR and scrolls the visible viewport into scrollback on
    `Vt100_Output` before clearing.
  - Check on upgrade: verify `Renderer._waiting_for_cpr_futures`,
    `Renderer.cpr_support`, `Renderer._min_available_height`,
    `Renderer._last_screen`, `Renderer.render`, `Application._on_resize`,
    `PromptSession._get_default_buffer_control_height`, and the
    `class:bottom-toolbar` `ConditionalContainer` layout structure.
  - Tests: `tests/unit/fast_agent/ui/test_prompt_input_runtime.py`,
    `tests/unit/fast_agent/ui/test_prompt_keybindings.py`.

- **Typeahead replay (`src/fast_agent/ui/prompt/turn_input.py`)**
  - Uses `prompt_toolkit.input.typeahead.get_typeahead`, `store_typeahead`, and
    `prompt_toolkit.application.current.get_app_session` to capture and replay
    buffered input into the next prompt.
  - Check on upgrade: verify `typeahead` module helpers remain intact.
  - Tests: `tests/unit/fast_agent/ui/test_prompt_turn_outcomes.py`.

---

## `textual-image`

- **Private `_terminal` probe & cell-size helpers (`src/fast_agent/ui/terminal_images/probe.py`, `renderer.py`, `sixel.py`, `halfcell.py`, `src/fast_agent/cli/checks/terminal.py`)**
  - Imports private module `textual_image._terminal` (`probe_terminal`,
    `get_cell_size`, `CellSize`, `TerminalError`) and temporarily scopes a
    compatibility wrapper around its batched terminal probe in `probe.py`.
  - Why: `textual-image 0.14.2` treats primary DA as an end marker during its
    batched probe and does not absorb duplicate Kitty graphics replies (which
    multiplexers such as `herdr-core` can emit), causing false negatives or
    leaked escape replies.
  - Check on upgrade: verify `_terminal` symbols/signatures and whether
    upstream fixed out-of-order DA / duplicate Kitty graphics reply handling.
  - Tests: `tests/unit/fast_agent/ui/` terminal-image tests (`pytest -k terminal_image`).

---

## `openai`

- **Private strict-schema helper (`src/fast_agent/llm/provider/openai/schema_sanitizer.py`)**
  - Imports `openai.lib._pydantic._ensure_strict_json_schema` (private, mutates
    in place; wrapped in `sanitize_response_format_schema` with `deepcopy`).
  - Check on upgrade: verify location/signature or migrate if OpenAI exposes a
    public strict-schema normalizer.
- **Private `_custom_headers` & `_enforce_credentials` (`src/fast_agent/llm/provider/openai/llm_openai.py`, `src/fast_agent/llm/provider/copilot/openai_client.py`)**
  - Mutates `client._custom_headers` after `AsyncOpenAI(...)` construction to
    isolate/strip ambient `OPENAI_CUSTOM_HEADERS` and `Authorization` headers;
    Copilot gateway client also passes `_enforce_credentials=False`.
  - Check on upgrade: verify `AsyncOpenAI` still stores merged custom headers on
    `_custom_headers` and accepts `_enforce_credentials`, or switch to a public
    opt-out if added.
- **Optional reasoning-delta event imports (`src/fast_agent/llm/provider/openai/openresponses_streaming.py`)**
  - Uses a defensive `try/except` around newer SDK reasoning delta event classes.
  - Check on upgrade: simplify once minimum supported `openai` version includes
    those event types.

---

## `anthropic` (and `DefaultAsyncHttpxClient` / `httpx`)

- **Proxy mounts & custom header isolation (`src/fast_agent/llm/provider/copilot/messages.py`)**
  - Calls `http_client._mounts.clear()` on `DefaultAsyncHttpxClient(trust_env=False)`
    and replaces `client._custom_headers = headers` on `AsyncAnthropic(...)`.
  - Why: the pinned Anthropic SDK still installs environment proxy mounts even
    when `trust_env=False`, and merges `ANTHROPIC_CUSTOM_HEADERS` during
    construction.
  - Check on upgrade: verify whether `DefaultAsyncHttpxClient(trust_env=False)`
    no longer installs proxy mounts and whether `AsyncAnthropic._custom_headers`
    behavior changed.

---

## `agent-client-protocol` (`acp`)

- **`field_meta` attribute for wire `_meta` (`src/fast_agent/acp/tool_progress.py`)**
  - Relies on the ACP Python SDK's `field_meta` attribute name when setting or
    reading `_meta` on update payloads.
  - Check on upgrade: verify `field_meta` naming/aliasing on ACP schema models.
