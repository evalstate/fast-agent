# Runtime MCP Attach/Detach — Implementation Slices (`hot-mcp-slices`)

This breaks `plan/hot-mcp.md` into small PR-sized slices with explicit ordering.

## Scope assumptions (carried from `hot-mcp`)
- Runtime MCP attach/detach is **in-memory only** for this phase.
- Detach removes attachment from agent runtime state, **not** config definitions.
- ACP shared scope behavior is shared-state mutation across sessions.
- `/mcp list` shows attached servers; `/mcp connect` can expose configured-but-detached options.

---

## Recommended PR sequence

1. **PR1 — Core runtime API + connection options**
2. **PR2 — Agent/App/FastAgent callback wiring**
3. **PR3 — TUI commands (`/mcp ...`, `/connect`)**
4. **PR4 — ACP `/mcp` slash command + session refresh/cache invalidation**
5. **PR5 — Test hardening + UX polish + docs/help pass**

---

## PR1 — Core runtime API + connection options

### Goal
Add attach/detach primitives in MCP core without touching UI/ACP flows.

### Files
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/mcp_connection_manager.py`
- `src/fast_agent/mcp/types.py` (if shared type exposure is needed)

### Work
1. Add typed models:
   - `MCPAttachOptions`
   - `MCPAttachResult`
   - `MCPDetachResult`
2. Add connection-manager option threading:
   - `trigger_oauth` override into `_prepare_headers_and_auth(...)`
   - per-attach startup timeout in `get_server(...)` / `launch_server(...)`
3. Add aggregator runtime operations:
   - `attach_server(...)`
   - `detach_server(...)`
   - `list_attached_servers()`
   - `list_configured_detached_servers()`
4. Ensure attach/detach correctly updates runtime maps/caches:
   - `server_names`
   - `_namespaced_tool_map`
   - `_server_to_tool_map`
   - `_prompt_cache`
   - `_skybridge_configs`

### Acceptance
- Existing startup behavior (`load_servers`) unchanged.
- Attach returns tools/prompts delta data.
- Detach disconnects persistent connection and removes in-memory indexes.
- No config file writes.

### Unit tests (target)
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py`
- `tests/unit/fast_agent/mcp/test_mcp_connection_manager_attach_options.py`

---

## PR2 — Agent/App/FastAgent callback wiring

### Goal
Expose runtime attach/detach through agent and app boundaries so frontends do not touch private internals.

### Files
- `src/fast_agent/agents/mcp_agent.py`
- `src/fast_agent/mcp/types.py`
- `src/fast_agent/core/agent_app.py`
- `src/fast_agent/core/fastagent.py`

### Work
1. Add `McpAgent` public methods:
   - `attach_mcp_server(...)`
   - `detach_mcp_server(...)`
   - `list_attached_mcp_servers()`
2. Extend `McpAgentProtocol` with those methods.
3. Add `AgentApp` callback setters + wrappers:
   - attach/detach/list attached/list configured MCP servers
4. Wire callbacks in `FastAgent` wrapper setup (near existing card/tool callback setup).
5. Non-MCP agent target returns friendly/typed failure.

### Acceptance
- Runtime MCP attach/detach is callable from `AgentApp` API.
- Behavior works for active instances (interactive/server modes).
- No regressions to existing agent-card tooling callbacks.

### Unit tests (target)
- `tests/unit/fast_agent/core/test_agent_app_mcp_callbacks.py`

---

## PR3 — TUI commands (`/mcp ...`, `/connect`)

### Goal
Add runtime MCP command UX to interactive prompt.

### Files
- `src/fast_agent/ui/command_payloads.py`
- `src/fast_agent/ui/enhanced_prompt.py`
- `src/fast_agent/ui/interactive_prompt.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py` (new)
- `src/fast_agent/commands/handlers/__init__.py` (export as needed)

### Work
1. Add payloads:
   - `McpListCommand`
   - `McpConnectCommand`
   - `McpDisconnectCommand`
2. Parse commands in `parse_special_input`:
   - `/mcp`
   - `/mcp list`
   - `/mcp connect ...`
   - `/mcp disconnect ...`
   - `/connect ...` alias with autodetect (`url` / `npx` / `uvx` / stdio fallback)
3. Add completer/help text updates:
   - `/mcp` subcommands
   - disconnect suggestions from attached servers
4. Add command handler module returning `CommandOutcome`.
5. Route new payloads in `InteractivePrompt` match/case.

### Acceptance
- `/mcp` without args remains backward compatible (status-style behavior).
- `/connect` is alias-only for connect flow.
- Clear connect/disconnect output includes tool/prompt changes.

### Unit tests (target)
- `tests/unit/fast_agent/ui/test_parse_mcp_commands.py`
- Extend `tests/unit/fast_agent/ui/test_agent_completer.py`
- Extend `tests/unit/fast_agent/ui/test_interactive_prompt_agent_commands.py`
- `tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py`

---

## PR4 — ACP `/mcp` command + shared-session refresh

### Goal
Expose the same runtime MCP operations in ACP slash commands and keep session/system state fresh.

### Files
- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/acp/server/agent_acp_server.py`
- `src/fast_agent/acp/acp_context.py`

### Work
1. Add `mcp` to ACP session command registry.
2. Implement `_handle_mcp(arguments)` with:
   - `list`
   - `connect ...`
   - `disconnect <name>`
3. Add ACP server callback plumbing (parallel to existing card/tool callback pattern).
4. Ensure shared-scope behavior mutates shared instance state.
5. After attach/detach:
   - rebuild instruction (`rebuild_agent_instruction(...)`)
   - invalidate/update ACP instruction cache (`ACPContext.invalidate_instruction_cache(...)`)
   - send available commands update where needed

### Acceptance
- ACP `/mcp` commands function end-to-end.
- `/status system` and prompt-time instruction content are fresh after connect/disconnect.
- Shared-scope side effects are documented in command output/help.

### Tests (target)
- Add ACP slash unit coverage if missing:
  - `tests/unit/fast_agent/acp/test_slash_commands_mcp.py`
- Integration (if available in workspace):
  - ACP slash connect/list/disconnect roundtrip

---

## PR5 — Hardening + UX polish + docs/help cleanup

### Goal
Finish quality pass and reduce edge-case regressions.

### Files
- Touch-up across prior files + docs/help text

### Work
1. Improve error messaging:
   - startup timeout message includes timeout seconds
   - auth-needed guidance when `trigger_oauth=False`
2. Reconnect policy checks:
   - preserve current no-infinite-loop safeguards
   - support per-attach reconnect override in-memory
3. Final output polish:
   - list configured detached options after connect
   - consistent markdown/TUI phrasing
4. Update docs/help where command lists are duplicated.

### Acceptance
- Consistent UX between TUI and ACP.
- Edge-case errors are actionable.
- No stale instruction/cache issues observed.

---

## Cross-slice dependency notes
- **PR2 depends on PR1** (core methods must exist).
- **PR3 depends on PR2** (TUI should call app callbacks, not internals).
- **PR4 depends on PR2** (ACP also uses app/server callback plumbing).
- **PR5 depends on PR3/PR4** (final polish after both frontends are live).

---

## Validation checklist per PR
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run pytest tests/unit`
- Run relevant integration tests for changed ACP/TUI paths.
- Ask operator to run e2e when ready.

---

## Notes for implementation discipline
- Keep diffs focused per slice; avoid unrelated cleanup.
- Respect project rule: no mocks/monkeypatch in tests for this feature area.
- Prefer protocol-first additions (`McpAgentProtocol`, callback interfaces) before UI wiring.
