# hot-mcp Implementation Review

Date: 2026-02-09

## Summary

The hot MCP plan (`hot-mcp.md` + `hot-mcp-slices.md`) has been implemented across core runtime, app wiring, TUI, and ACP.

Key outcome: **startup MCP loading now routes through the same attach path used for runtime attach**, and runtime MCP server management is now available via both:
- TUI: `/mcp list|connect|disconnect` and `/connect` alias
- ACP: `/mcp list|connect|disconnect`

---

## What was implemented

## PR1 — Core runtime API + connection options

Implemented in:
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/mcp_connection_manager.py`
- `src/fast_agent/mcp/types.py`
- `src/fast_agent/mcp/interfaces.py`
- `src/fast_agent/agents/mcp_agent.py`

### Added models and runtime APIs
- `MCPAttachOptions`
- `MCPAttachResult`
- `MCPDetachResult`
- `MCPAggregator.attach_server(...)`
- `MCPAggregator.detach_server(...)`
- `MCPAggregator.list_attached_servers()`
- `MCPAggregator.list_configured_detached_servers()`

### Startup path unification
- `MCPAggregator.load_servers()` now uses `attach_server(...)` for startup-loaded servers.
- `load_on_start=False` remains honored unless `force_connect=True`.

### Connection manager option threading
- `_prepare_headers_and_auth(..., trigger_oauth=...)`
- `get_server(...)`, `launch_server(...)`, `reconnect_server(...)` now accept:
  - `startup_timeout_seconds`
  - `trigger_oauth`
- Startup timeout produces explicit `ServerInitializationError` and requests shutdown.

### Agent surface and protocol updates
- `McpAgent.attach_mcp_server(...)`
- `McpAgent.detach_mcp_server(...)`
- `McpAgent.list_attached_mcp_servers()`
- `McpAgentProtocol` extended accordingly.

---

## PR2 — Agent/App/FastAgent callback wiring

Implemented in:
- `src/fast_agent/core/agent_app.py`
- `src/fast_agent/core/fastagent.py`

### AgentApp additions
- runtime MCP callback setters and wrappers for:
  - attach/detach
  - list attached
  - list configured detached
- capability checks:
  - `can_attach_mcp_servers()`
  - `can_detach_mcp_servers()`

### FastAgent wiring
- New runtime MCP callbacks are wired when wrapper is built.
- Non-MCP target agents return friendly runtime errors.
- Attach/detach path rebuilds instruction post-change.

---

## PR3 — TUI `/mcp ...` and `/connect`

Implemented in:
- `src/fast_agent/ui/command_payloads.py`
- `src/fast_agent/ui/enhanced_prompt.py`
- `src/fast_agent/ui/interactive_prompt.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py`

### Added payloads
- `McpListCommand`
- `McpConnectCommand`
- `McpDisconnectCommand`

### Parser behavior
- `/mcp` remains backward-compatible status view.
- `/mcp list`
- `/mcp connect ...` with support for:
  - `--name`
  - `--timeout`
  - `--oauth|--no-oauth`
  - `--reconnect|--no-reconnect`
- `/mcp disconnect <server>`
- `/connect ...` alias with target-mode autodetect (`url|npx|uvx|stdio`).

### Completer/help updates
- `/mcp` subcommand completions added.
- `/mcp disconnect` suggests attached servers.
- Help text updated with new MCP runtime commands.

---

## PR4 — ACP `/mcp`

Implemented in:
- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/acp/server/agent_acp_server.py`
- `src/fast_agent/core/fastagent.py` (ACP server construction wiring)

### ACP slash command support
- Added `mcp` to session command registry.
- Added handler for:
  - `list`
  - `connect`
  - `disconnect`

### ACP server callback plumbing
- `AgentACPServer` now accepts and wires runtime MCP callbacks:
  - attach/detach/list attached/list configured-detached

### Session refresh/cache behavior
- After connect/disconnect:
  - ACP instruction cache is invalidated for the current agent
  - available commands update is sent

---

## PR5 — tests, hardening, polish

### Added/updated tests
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py`
- `tests/unit/fast_agent/core/test_agent_app_mcp_callbacks.py`
- `tests/unit/fast_agent/ui/test_parse_mcp_commands.py`
- `tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py`
- `tests/unit/fast_agent/acp/test_slash_commands_mcp.py`
- `tests/unit/fast_agent/ui/test_agent_completer.py` (extended)

### Validation status
- `uv run scripts/lint.py --fix` ✅
- `uv run scripts/typecheck.py` ✅
- New/modified targeted unit tests ✅

Full unit suite still has two known unrelated environment-sensitive failures in this workspace:
- `tests/unit/core/test_prompt_templates.py::test_load_skills_for_context_handles_missing_directory`
- `tests/unit/fast_agent/session/test_session_manager.py::test_apply_session_window_appends_pinned_overflow`

---

## Notable implementation details / deviations

1. **Attached vs configured server tracking in aggregator**
   - Added `_attached_server_names` alongside configured list behavior to preserve existing expectations around `server_names` while enabling runtime attach/detach semantics.

2. **Runtime handler abstraction**
   - Introduced `commands/handlers/mcp_runtime.py` to share connect/list/disconnect behavior across TUI and ACP command frontends.

3. **Reconnect semantics in TUI parser**
   - `--reconnect` is interpreted as `force_reconnect=True` for attach operation.
   - `--no-reconnect` maps to `reconnect_on_disconnect=False` override.

---

## Remaining follow-up recommendations

1. **E2E validation**
   - Run operator-level e2e for:
     - TUI `/mcp connect` + `/connect` + disconnect
     - ACP `/mcp connect/list/disconnect` in shared scope
     - instruction freshness in `/status system`

2. **Integration stability pass**
   - Re-run ACP integration suite in a clean environment/session directory to avoid session-list contamination.

3. **Optional polish**
   - Align exact output wording between TUI and ACP for MCP runtime messages.
   - Consider adding explicit ACP command help for `/mcp` usage variants.

