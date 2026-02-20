# Prompt / Slash Refactor Plan (High-Cohesion Module Structure)

## Why

The interactive prompt path is working, but the code is concentrated in a few very large files:

- `src/fast_agent/ui/enhanced_prompt.py` (~3600 lines)
- `src/fast_agent/acp/slash_commands.py` (~1900 lines)
- `src/fast_agent/ui/interactive_prompt.py` (~1100 lines)

This increases change risk and slows feature work (like `/model web_search` + ACP parity).

## Goals

1. Keep behavior stable while reducing file size and coupling.
2. Move to high-cohesion modules with clear ownership.
3. Keep most modules under **1000 lines** (prefer <500 where practical).
4. Preserve external imports and user-facing command behavior.

## Non-goals

- No command UX redesign in this refactor.
- No protocol changes for ACP clients beyond current feature set.
- No broad reformat/rewrite of unrelated subsystems.

---

## Target module structure

## 1) TUI prompt parsing/completion (`ui`)

Create a dedicated package:

```text
src/fast_agent/ui/prompt/
  __init__.py
  parser.py
  command_help.py
  completer.py
  completion_sources.py
  toolbar.py
  keybindings.py
  session.py
```

### Responsibilities

- `parser.py`
  - Parse raw input to `CommandPayload` (pure parsing).
  - Own `/model`, `/history`, `/session`, `/mcp`, etc. tokenization.
- `command_help.py`
  - All help text and usage strings (single source of truth).
- `completer.py`
  - `AgentCompleter` orchestration only.
- `completion_sources.py`
  - Reusable completion providers (history files, sessions, model subcommands, MCP servers).
- `toolbar.py`
  - Toolbar rendering and model/status indicators.
- `keybindings.py`
  - Prompt-toolkit keybinding definitions.
- `session.py`
  - PromptSession wiring + `get_enhanced_input` lifecycle.

Keep a thin compatibility facade:

- `src/fast_agent/ui/enhanced_prompt.py` exports existing symbols and delegates to package modules.

---

## 2) TUI execution loop (`interactive_prompt`)

Create a dedicated package:

```text
src/fast_agent/ui/interactive/
  __init__.py
  loop.py
  command_dispatch.py
  command_context.py
  mcp_connect_flow.py
  agent_runtime.py
```

### Responsibilities

- `loop.py`
  - Main prompt loop orchestration only.
- `command_dispatch.py`
  - Pattern matching on `CommandPayload` and dispatch to handlers.
- `command_context.py`
  - Build `CommandContext`, emit `CommandOutcome`.
- `mcp_connect_flow.py`
  - Isolated runtime connect/disconnect progress/cancel/signals.
- `agent_runtime.py`
  - Agent switching, refresh after card/reload, availability updates.

Keep `src/fast_agent/ui/interactive_prompt.py` as thin facade + backward-compatible entrypoint.

---

## 3) ACP slash command system

Create package:

```text
src/fast_agent/acp/slash/
  __init__.py
  command_catalog.py
  dispatch.py
  io.py
  handlers/
    __init__.py
    history.py
    model.py
    session.py
    status.py
    skills.py
    tools.py
    cards.py
    mcp.py
```

### Responsibilities

- `command_catalog.py`
  - Available command definitions and dynamic hint construction (e.g. `/model`).
- `dispatch.py`
  - Parse + route command name to handler.
- `handlers/*.py`
  - One domain per module, calling existing shared command handlers.
- `io.py`
  - ACP output formatting and progress update helpers.

Keep `src/fast_agent/acp/slash_commands.py` as orchestration shell + compatibility class wrapper.

---

## 4) Shared capability helpers (already started)

Continue centralizing model capability checks in one place:

- `src/fast_agent/commands/handlers/model.py`
  - `model_supports_text_verbosity`
  - `model_supports_web_search`
  - `model_supports_web_fetch`

TUI completer and ACP catalog should consume these helpers (avoid duplicate provider/model logic).

---

## Migration sequence (safe slices)

### Phase 1: Extract pure modules first

1. Move parser logic to `ui/prompt/parser.py`.
2. Move help text to `ui/prompt/command_help.py`.
3. Move completion sources to `ui/prompt/completion_sources.py`.
4. Keep `enhanced_prompt.py` re-exporting old function/class names.

**Exit criteria**: parse/completion tests unchanged.

### Phase 2: Extract TUI dispatch

1. Move `match`-based command execution to `ui/interactive/command_dispatch.py`.
2. Move context/outcome emission helpers to `command_context.py`.

**Exit criteria**: interactive command tests unchanged.

### Phase 3: Isolate MCP connect flow

1. Move MCP runtime connect/cancel/progress/signals to `mcp_connect_flow.py`.

**Exit criteria**: MCP interactive tests unchanged.

### Phase 4: ACP slash split

1. Add `acp/slash/handlers/*` modules by domain.
2. Move model/history/session/status commands first.
3. Keep `SlashCommandHandler` public class and API stable.

**Exit criteria**: ACP slash integration tests unchanged.

### Phase 5: Cleanup + tighten boundaries

1. Remove dead/refactor shim modules once migrated (`src/fast_agent/ui/refactor/*` if unused).
2. Ensure no module >1000 lines in target area.

---

## Cohesion / boundary rules

1. **Parser modules do not perform I/O** (only return payloads/errors).
2. **Completer modules do not mutate runtime**.
3. **Dispatch modules do not render markdown directly**; they return/forward outcomes.
4. **Handler modules call shared command handlers** (avoid duplicate business logic).
5. **Help/hint text in one place** per interface (TUI vs ACP), with helper-driven dynamic parts.

---

## Test strategy

Run after each phase:

- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run pytest tests/unit/fast_agent/ui -q`
- `uv run pytest tests/integration/acp/test_acp_slash_commands.py -q`

And before merge:

- `uv run pytest tests/unit > /tmp/unit.log && tail -n 80 /tmp/unit.log`
- `uv run scripts/cpd.py --check`

(Existing baseline duplications outside this scope may still fail CPD; track separately.)

---

## Risks and mitigations

- **Risk:** subtle parser behavior drift.
  - **Mitigation:** keep parser pure, preserve tokenization logic verbatim first, then cleanup.
- **Risk:** completion regressions.
  - **Mitigation:** expand completion unit tests before/with extraction.
- **Risk:** ACP metadata drift in available commands.
  - **Mitigation:** assert dynamic hints in integration tests.
- **Risk:** circular imports after splitting.
  - **Mitigation:** strict dependency direction: parser/completion -> shared helpers; dispatch -> handlers.

---

## Suggested immediate next slice

1. Extract `parse_special_input` into `ui/prompt/parser.py`.
2. Extract `/model` and `/history` completion branches into `ui/prompt/completion_sources.py`.
3. Keep `enhanced_prompt.py` facade-only for these paths.

This gives high cohesion quickly, with low behavior risk and immediate line-count reduction.
