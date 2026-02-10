# Handover: hot MCP runtime work + follow-up hardening

Date: 2026-02-09

## Context

This session implemented and iterated on the **hot MCP** runtime attach/detach feature set described in:
- `plan/hot-mcp.md`
- `plan/hot-mcp-slices.md`

A review document was written to:
- `plan/hot-mcp-impl-review.md`

---

## High-level outcomes

Implemented end-to-end runtime MCP server management across:
- core runtime (`MCPAggregator`, `MCPConnectionManager`, `McpAgent`)
- app wiring (`AgentApp`, `FastAgent`)
- TUI (`/mcp`, `/connect`)
- ACP slash commands (`/mcp list|connect|disconnect`)

Also implemented auth unification for runtime URL connect:
- `/mcp connect ... --auth <token>` now supported
- URL auth/HF token behavior now reuses CLI URL parser path (`parse_server_urls` + HF auth header logic)

---

## Commits made

### Main feature commit
- **`570e4fa2`**
- Message: `Implement runtime MCP connect/disconnect across core, TUI, and ACP`
- Includes:
  - runtime attach/detach models + APIs
  - startup MCP load path routed through attach path
  - TUI `/mcp` + `/connect`
  - ACP `/mcp`
  - command handlers + tests
  - implementation review doc

### Post-commit updates (NOT yet committed in this transcript)
After `570e4fa2`, additional fixes were made but not committed yet:
1. Scoped npm shorthand improvements for `/mcp connect` (`@scope/pkg` -> `npx` + args)
2. Better already-attached messaging (suggest `--reconnect`)
3. Removed earlier delay/retry behavior from aggregator attach path (by user request)
4. Added runtime `/mcp connect --auth` support + shared CLI URL/HF auth handling
5. Reduced noisy network-failure output:
   - suppress MCP SDK `Error in post_writer` tracebacks for streamable-http
   - dedupe repeated `MCP server ... offline` prints during outage

You should inspect working tree and commit these follow-up changes.

---

## Key files touched

### Core MCP runtime
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/mcp_connection_manager.py`
- `src/fast_agent/mcp/mcp_agent_client_session.py`
- `src/fast_agent/mcp/types.py`
- `src/fast_agent/mcp/interfaces.py`
- `src/fast_agent/agents/mcp_agent.py`

### App/server wiring
- `src/fast_agent/core/agent_app.py`
- `src/fast_agent/core/fastagent.py`
- `src/fast_agent/acp/server/agent_acp_server.py`
- `src/fast_agent/acp/slash_commands.py`

### TUI / command flow
- `src/fast_agent/ui/command_payloads.py`
- `src/fast_agent/ui/enhanced_prompt.py`
- `src/fast_agent/ui/interactive_prompt.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py`

### Tests added/updated
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py`
- `tests/unit/fast_agent/core/test_agent_app_mcp_callbacks.py`
- `tests/unit/fast_agent/ui/test_parse_mcp_commands.py`
- `tests/unit/fast_agent/ui/test_agent_completer.py`
- `tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py`
- `tests/unit/fast_agent/acp/test_slash_commands_mcp.py`

### Docs
- `plan/hot-mcp-impl-review.md`
- (this file) `handover.md`

---

## Notable behavior discussions and decisions

1. **Startup uses attach path**
   - User explicitly requested startup MCP server loading to use same attach/detach routines.
   - Implemented by routing `load_servers()` through `attach_server(...)`.

2. **Delay/retry on attach removed**
   - There was a brief retry/delay heuristic added for empty tool lists.
   - User rejected this as cargo-cult; it was removed.
   - Current behavior: no delay; rely on proper initialize + tool fetch.

3. **Scoped npm shorthand support**
   - `/mcp connect @modelcontextprotocol/server-everything` now treated as npx form.

4. **Auth parity request**
   - User requested runtime TUI/ACP connect to share CLI auth/HF_TOKEN logic.
   - Implemented via `parse_server_urls` reuse for URL mode.

5. **Network sleep/wake failure handling**
   - User reported huge traceback spam (`Error in post_writer`) + repeated offline lines after laptop sleep/network change.
   - Added logging filter for streamable-http post_writer noise.
   - Added deduping for repeated offline notices.

6. **Stuck Ctrl+C in interactive session**
   - User could not break with Ctrl+C; `SIGQUIT` (Ctrl+\) worked.
   - No code change yet for this specific SIGINT robustness; follow-up recommended.

---

## Validation performed

Repeatedly ran:
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- targeted unit tests for changed areas

Known unrelated/environment-sensitive failures in full unit suite remain in this workspace:
- `tests/unit/core/test_prompt_templates.py::test_load_skills_for_context_handles_missing_directory`
- `tests/unit/fast_agent/session/test_session_manager.py::test_apply_session_window_appends_pinned_overflow`

These were pre-existing and align with repo notes about environment/session ordering effects.

---

## Current user-reported state

- Runtime `/mcp connect` now works in scenarios that previously failed.
- User observed one session where Ctrl+C no longer interrupted; SIGQUIT terminated it.
- User wants robust behavior around long-lived remote MCP connections over sleep/wake.

---

## Suggested next steps for new operator

1. **Commit pending follow-up changes**
   - Check `git status`
   - Commit post-`570e4fa2` fixes as a new commit

2. **Run focused regression checks**
   - hot MCP TUI commands (`/mcp`, `/connect`, `--auth`)
   - ACP `/mcp ...`
   - sleep/wake or temporary DNS failure behavior for streamable-http MCP server

3. **Optional hardening**
   - Improve SIGINT handling in interactive loop:
     - first Ctrl+C -> graceful cancel
     - second Ctrl+C within timeout -> forced exit
   - Consider user-facing reconnect status line for network outages

4. **Operator e2e**
   - Ask operator to run e2e scenarios (as required by project workflow)

---

## Quick commands used during session

- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run pytest tests/unit/...`
- log checks:
  - `tail -n 80 fastagent.jsonl`
  - `rg -n "..." fastagent.jsonl`

