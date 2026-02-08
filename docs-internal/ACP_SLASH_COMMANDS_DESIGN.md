# ACP Slash Command Design & Rendering Plan

## Purpose

Document the design pattern for ACP slash commands, how command output is rendered, and the refactoring direction for keeping logic, rendering, and ACP wiring aligned. The `/tools` and `/skills` commands are the model examples after recent refactors, with `/history`, `/session`, `/status`, `/card`, `/agent`, and `/reload` now following the same pattern.

## Current Slash Command Surface

`SlashCommandHandler` (in `src/fast_agent/acp/slash_commands.py`) implements ACP-facing commands, split into:

- Session commands: `/status`, `/history`, `/session`, `/clear`, `/save`, `/load`.
- Agent/registry commands: `/tools`, `/skills` (add/remove/registry/list), `/card`, `/agent`, `/reload`.
- Auth/diagnostics: `/status auth`, `/status authreset`.

All routing is in `execute_command`, which delegates to `_handle_*` methods.

## Design Pattern (Target)

1. **Command Logic in Handlers**
   - Use `fast_agent.commands.handlers.*` for core behavior when possible (skills/history/sessions/card/agent/reload).
   - Work with a shared `CommandContext`, return `CommandOutcome`.

2. **Rendering in Dedicated Renderers**
   - Convert domain summaries into Markdown in `fast_agent.commands.renderers.*`.
   - Standard outcomes use `render_command_outcome_markdown` to format `CommandOutcome` in ACP.
   - `/tools` uses `render_tools_markdown` and `build_tool_summaries`.
   - `/skills` uses `render_skills_by_directory`, `render_skill_list`, `render_marketplace_skills`, and `render_skills_registry_overview`.
   - `/history` uses `render_history_overview_markdown` (summary from `history_summaries`).
   - `/session` list uses `render_session_list_markdown` (summary from `session_summaries`).
   - `/status` uses `render_status_markdown` (summary from `status_summaries`).

3. **ACP Adapter Layer**
   - ACP builds a `CommandContext` via `_build_command_context`.
   - ACP converts `CommandOutcome` to Markdown via `_format_outcome_as_markdown` → `render_command_outcome_markdown`.
   - ACP uses specialized renderers for list output (tools/skills/history/session/status).

## Rendering Flow (Examples)

### `/tools`
- ACP handler calls agent `list_tools()`.
- `build_tool_summaries` extracts name/title/description/args/suffix.
- `render_tools_markdown` formats ordered list, blockquoted description, args lines.

### `/skills`
- ACP uses `skills_handlers` for add/remove actions (install/remove).
- `/skills` list uses `render_skills_by_directory` to produce ordered list entries with:
  - name and index
  - blockquoted description
  - blockquoted source path on its own line
- `/skills add` (no args) uses `render_marketplace_skills` for marketplace listings.
- `/skills registry` (no args) uses `render_skills_registry_overview` for registry listings.
- `/skills remove` (no args) uses `render_skills_remove_list` for local skills listings.
- Override section uses `render_skill_list` for consistent formatting.

### `/history`
- ACP uses `history_handlers.handle_show_history` to build the overview.
- `history_summaries.build_history_overview` extracts counts + recent snippets.
- `render_history_overview_markdown` formats the summary.
- `/history save` and `/history load` share handlers and use `render_command_outcome_markdown`.

### `/session`
- ACP uses `sessions_handlers` for list/new/resume/title/fork/clear.
- List output uses `session_summaries.build_session_list_summary` + `render_session_list_markdown`.
- Other session actions format their `CommandOutcome` with `render_command_outcome_markdown`.

### `/status`
- ACP builds a `StatusSummary` with `build_status_summary`.
- `render_status_markdown` formats version, client info, models, statistics, warnings.
- `/status system` uses `build_system_prompt_summary` + `render_system_prompt_markdown`.
- `/status auth` / `/status authreset` use `render_permissions_markdown`.

### `/card` / `/agent` / `/reload`
- ACP uses `commands.handlers.agent_cards` for card loading, tool attach/detach, dumps, and reload.
- `render_command_outcome_markdown` formats the resulting `CommandOutcome`.

## Tests & Coverage Notes

Existing ACP coverage:
- `tests/integration/acp/test_acp_skills_manager.py` exercises skills add/remove/registry and validates system prompt changes.
- `tests/integration/acp/test_acp_slash_commands.py` checks ACP command wiring and common commands like `/status`.

Coverage added during refactor:
- `/skills` list/add/remove tests assert key elements (name, description, source path, repository).
- `/history save` + `/history load` tests assert key results without depending on exact markdown.
- `/session resume` test asserts the agent switch message.
- `/card` `/agent` `/reload` ACP tests assert semantic output (loaded cards, attached tools, reload status).

Guideline: for ACP list outputs, assert presence of key semantic elements (name, description, source, counts) rather than exact formatting.

## Design Limitations / Follow-ups

1. **Remaining ACP-only handlers**
   - `/tools` is still ACP-only logic (no shared handler yet).
   - `/status auth` and `/status authreset` are ACP-only (rendered via shared Markdown renderer).

2. **Markdown vs Rich formatting divergence**
   - ACP uses Markdown, TUI uses `rich.Text`.
   - Aligning on shared summary objects (like `ToolSummary`, `HistoryOverview`, `StatusSummary`) reduces divergence.

3. **Test matrix gaps**
   - `/tools` lacks a direct ACP integration test asserting key elements.
   - `/skills registry` list output is still only covered indirectly via numbered-selection tests.

## Next Refactor Plan

1. **Introduce shared handler for `/tools`**
   - Move tool listing logic to `commands/handlers` to keep ACP/TUI in sync.

2. **Add semantic ACP tests for `/tools`**
   - Assert tool name, description, args/labels without pinning markdown layout.

3. **Keep outcome rendering centralized**
   - Prefer `render_command_outcome_markdown` for standard outcomes.
   - Use dedicated renderers only for list/summary-heavy commands.

## Input & Interaction Model

- **TUI**: uses prompt_toolkit completions and interactive prompts (numbered selection) via `CommandIO`.
- **ACP**: is non-interactive; commands must include explicit arguments (e.g. `/skills add my-skill`).
  - ACP list outputs should include the number/name hints so users can copy a direct command.
