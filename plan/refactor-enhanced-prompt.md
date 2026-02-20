# Enhanced Prompt Refactor Plan (Post-Slices A–D)

## Context

We completed major splits for interactive loop + ACP slash modules, but `src/fast_agent/ui/enhanced_prompt.py` remains large.

Current size snapshot:

- `src/fast_agent/ui/enhanced_prompt.py`: **~2800 lines**
- `src/fast_agent/ui/interactive_prompt.py`: ~529 lines
- `src/fast_agent/acp/slash_commands.py`: ~723 lines

This plan focuses on finishing the enhanced prompt decomposition so prompt/TUI modules are cohesive, testable, and mostly <1000 lines (prefer <500).

---

## Refactor goals

1. Shrink `enhanced_prompt.py` to a thin import/orchestration module.
2. Move tightly related logic into dedicated modules with clear boundaries.
3. Reduce global mutable state and module-level side effects.
4. Reduce test monkeypatching in favor of constructor injection + explicit state objects.
5. Keep command UX/behavior stable.

---

## Non-goals

- No command UX redesign.
- No new protocol behavior in ACP/TUI.
- No broad style-only rewrites.

---

## Current hotspots in `enhanced_prompt.py`

### 1) `AgentCompleter` is still very large (~700+ lines)
Contains:
- command catalog metadata
- path/file completion utilities
- history/session/skills/model/mcp completion behavior
- shell completion behavior

### 2) `get_enhanced_input` is very large (~700+ lines)
Contains:
- state setup
- toolbar rendering logic
- shell runtime introspection
- session construction
- initial hint rendering
- prompt lifecycle instrumentation
- cleanup and error handling

### 3) UI helpers and utility clusters are mixed together
- toolbar helpers and alert flag parsing
- agent info/hierarchy display helpers
- editor integration and keybindings
- special command help rendering

### 4) Global mutable state in module namespace
- `agent_histories`, `available_agents`, `in_multiline_mode`, `_last_copyable_output`, `_copy_notice`, etc.

---

## Target module structure

Expand `src/fast_agent/ui/prompt/`:

```text
src/fast_agent/ui/prompt/
  __init__.py
  parser.py                     # already present
  completion_sources.py         # already present
  completer.py                  # AgentCompleter + minimal wiring
  completion_primitives.py      # file/path/session/helper completion utils
  toolbar.py                    # toolbar rendering + model/TDV/notification segments
  keybindings.py                # create_keybindings + AgentKeyBindings
  editor.py                     # get_text_from_editor
  display.py                    # show_mcp_status + agent hierarchy renderers
  alert_flags.py                # _extract_* + _resolve_alert_flags_from_history
  session.py                    # get_enhanced_input + prompt session lifecycle
  special_commands.py           # handle_special_commands + help text
  command_help.py               # centralized help text/constants
  state.py                      # PromptUiState dataclass + state store
```

Keep `enhanced_prompt.py` as a compatibility surface **temporarily**, then reduce/remove private symbol exports as tests migrate.

---

## Boundary rules

1. `parser.py` remains pure (no I/O).
2. Completion modules do not mutate runtime agent state.
3. `toolbar.py` accepts explicit dependencies/state, not globals.
4. `session.py` owns prompt lifecycle; it should call injected helpers.
5. `special_commands.py` is presentation-only and returns payloads/booleans.
6. No module should rely on hidden global variables from another module.

---

## State model redesign (key to reducing monkeypatching)

Introduce in `state.py`:

```python
@dataclass
class PromptUiState:
    agent_histories: dict[str, InMemoryHistory]
    available_agents: set[str]
    in_multiline_mode: bool
    last_copyable_output: str | None
    copy_notice: str | None
    copy_notice_until: float
    startup_notices: list[str]
    help_message_shown: bool
```

Provide a module-level `DEFAULT_PROMPT_UI_STATE` for backward compatibility.

### Migration rule

- New APIs accept `ui_state: PromptUiState | None = None` and default to `DEFAULT_PROMPT_UI_STATE`.
- Tests can pass an isolated state instance, avoiding `enhanced_prompt.available_agents = ...` monkeypatching.

---

## Detailed phases

## Phase 1 — Split stable utility clusters (low risk)

### Move to `alert_flags.py`
- `_category_to_alert_flag`
- `_extract_alert_flags_from_alert`
- `_extract_alert_flags_from_meta`
- `_resolve_alert_flags_from_history`

### Move to `editor.py`
- `get_text_from_editor`

### Move to `display.py`
- `show_mcp_status`
- `_display_agent_info_helper`
- hierarchy helper functions

### Move to `toolbar.py`
- `_left_truncate_with_ellipsis`
- `_format_parent_current_path`
- `_fit_shell_path_for_toolbar`
- `_fit_shell_identity_for_toolbar`
- `_can_fit_shell_path_and_version`
- `_toolbar_markup_width`
- `_resolve_toolbar_width`
- `_is_smart_agent`
- `_format_toolbar_agent_identity`

### Exit criteria
- Existing toolbar + alert tests pass unchanged.

---

## Phase 2 — Move keybinding and copy behavior

### Move to `keybindings.py`
- `AgentKeyBindings`
- `create_keybindings`

### Adjust
- `create_keybindings(..., ui_state=...)` to avoid direct mutation of module globals.
- Ctrl+Y copy path should read/write state via `ui_state`.

### Exit criteria
- keyboard behavior unchanged.
- no direct global write from keybinding handlers.

---

## Phase 3 — Extract completer fully

### Move class
- `AgentCompleter` -> `completer.py`

### Split internals
- file/path/session utility methods -> `completion_primitives.py`
- keep orchestration and public methods in `completer.py`

### Adjust tests
- switch imports to `from fast_agent.ui.prompt.completer import AgentCompleter`
- keep compatibility re-export in `enhanced_prompt.py` for one transition window only.

### Exit criteria
- `test_agent_completer.py` + parse command tests green.

---

## Phase 4 — Extract special command rendering/help

### Move to `special_commands.py`
- `handle_special_commands`

### Move help text to `command_help.py`
- static command list lines
- keyboard shortcuts lines
- conditional sections (`/history webclear`)

### Improve cohesion
- `special_commands.py` should call `command_help.render_help(...)` instead of inlining giant print block.

### Exit criteria
- `/help` output semantics unchanged.
- no large inline literal blocks in `enhanced_prompt.py`.

---

## Phase 5 — Extract prompt session lifecycle

### Move to `session.py`
- `get_enhanced_input`
- `get_selection_input`
- `get_argument_input`
- `ShellPrefixLexer`

### Decompose `get_enhanced_input`
Break into private helpers:
- `_initialize_prompt_state(...)`
- `_build_toolbar_renderer(...)`
- `_build_prompt_session(...)`
- `_render_startup_hints(...)`
- `_run_prompt_once(...)`
- `_cleanup_prompt_session(...)`

### Injection points
- `AgentCompleter` class injected (default production class)
- keybinding factory injected
- state object injected

### Exit criteria
- interactive prompt tests pass without monkeypatching module globals.

---

## Phase 6 — Reduce compatibility surface + update tests

### Tests to update first
- `tests/unit/fast_agent/ui/test_agent_completer.py`
- `tests/unit/fast_agent/ui/test_enhanced_prompt_toolbar.py`
- `tests/unit/fast_agent/ui/test_alert_flag_extraction.py`
- interactive tests touching `enhanced_prompt.available_agents`

### Move away from monkeypatching globals
- use explicit state fixture and dependency injection in prompt/session tests.
- where possible, patch a constructor parameter instead of module symbol.

### Compatibility policy
- Keep only public API symbols that are intentionally supported:
  - `get_enhanced_input`
  - `parse_special_input`
  - `handle_special_commands` (if still externally used)
- Remove/restrict incidental private exports from `enhanced_prompt.py` after tests migrate.

---

## Acceptance targets

1. `enhanced_prompt.py` under **1000** lines (stretch: <500).
2. `ui/prompt/*` modules each generally under 500 lines.
3. No direct test dependence on mutable module globals for prompt state.
4. Existing behavior/tests preserved.

---

## Test strategy

After each phase:

- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run pytest tests/unit/fast_agent/ui -q`

At milestone boundaries (phase 3, phase 5, phase 6):

- `uv run pytest tests/unit/fast_agent/ui/test_interactive_prompt_agent_commands.py -q`
- `uv run pytest tests/unit/fast_agent/ui/test_interactive_prompt_refresh.py -q`
- `uv run pytest tests/integration/acp/test_acp_slash_commands.py -q`

Before merge:

- `uv run pytest tests/unit > /tmp/unit.log && tail -n 80 /tmp/unit.log`
- `uv run scripts/cpd.py --check`

---

## Risks and mitigations

### Risk: subtle toolbar rendering drift
- Mitigation: keep rendering text identical initially; extract without changing format strings.

### Risk: prompt-toolkit lifecycle regressions
- Mitigation: isolate session cleanup logic and preserve existing exception handling paths.

### Risk: shell runtime display behavior changes
- Mitigation: add focused tests for toolbar shell path/version switching behavior.

### Risk: state migration churn
- Mitigation: dual-mode APIs (`ui_state` optional) and phased test migration.

---

## Immediate next slice (recommended)

1. Implement `state.py` and wire `get_enhanced_input(..., ui_state=...)`.
2. Extract `alert_flags.py` + `toolbar.py` utility functions.
3. Move `AgentCompleter` to `ui/prompt/completer.py` with compatibility re-export.

This gives the fastest line-count reduction and unlocks de-globalized tests.
