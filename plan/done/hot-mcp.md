# Runtime MCP Attach/Detach Plan (`hot-mcp`)

## Goal
Enable MCP servers to be attached and detached at runtime for active `McpAgent` instances, with command support in both TUI and ACP slash flows.

This plan is implementation-ready and mapped to current code in:
- `src/fast_agent/agents/mcp_agent.py`
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/mcp_connection_manager.py`
- `src/fast_agent/ui/enhanced_prompt.py`
- `src/fast_agent/ui/interactive_prompt.py`
- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/acp/server/agent_acp_server.py`

---

## Decisions Locked In

1. Persistence behavior for this phase: runtime attach/detach is **live in-memory only** (no config file mutation), and behaves as runtime state for the current process/session.
2. Detach behavior: **only detach from agent** (do not delete configured server definitions).
3. ACP shared instance scope behavior: `/mcp connect` and `/mcp disconnect` affect **all sessions on that shared instance**.
4. `/mcp list` shows **attached servers only**. `/mcp connect` should also show available configured-but-detached options from config.
5. Connect result messaging must show what tools/prompts became available.
6. `/connect` should support auto-detection of likely input shape (URL vs NPX/UVX command).  
   We will implement `/connect` as a TUI alias of `/mcp connect`.

---

## User-visible UX

## Commands

### TUI
- `/mcp` -> status/list (backward compatible with existing `/mcpstatus` behavior)
- `/mcp list`
- `/mcp connect <target> [--name <server>] [--timeout <seconds>] [--no-oauth|--oauth] [--reconnect|--no-reconnect]`
- `/mcp disconnect <server_name>`
- `/connect ...` (alias to `/mcp connect ...`, with auto-detect)

### ACP
- `/mcp list`
- `/mcp connect ...`
- `/mcp disconnect <server_name>`

(ACP does not need `/connect` alias unless explicitly requested later.)

## Connect target forms

Accepted by `/mcp connect` and `/connect`:
- `npx <pkg> [args...]`
- `uvx <pkg> [args...]`
- `<http(s)://...>` URL

Auto-detect rules for `/connect`:
1. Starts with `http://` or `https://` => URL mode.
2. Starts with `npx ` => stdio command with `command="npx"`.
3. Starts with `uvx ` => stdio command with `command="uvx"`.
4. Otherwise: treat as stdio command string (split with `shlex.split`) and infer server name.

---

## Current behavior summary (what we are extending)

- `McpAgent` currently exposes aggregator/status but no runtime MCP attach/detach API.
- `MCPAggregator.load_servers()` handles startup loading and tool/prompt index population.
- `MCPConnectionManager` handles persistent launch/reconnect/disconnect.
- TUI `/mcp` currently only renders status (`ShowMcpStatusCommand`).
- ACP slash commands currently have no `mcp` session command.

---

## API and type additions

## New result/option models

Add in `src/fast_agent/mcp/mcp_aggregator.py` (or nearby dedicated typed module if preferred):

- `@dataclass(frozen=True, slots=True) class MCPAttachOptions:`
  - `startup_timeout_seconds: float = 10.0`
  - `trigger_oauth: bool = True`
  - `force_reconnect: bool = False` (optional)
  - `reconnect_on_disconnect: bool | None = None` (optional per-attach override)

- `@dataclass(frozen=True, slots=True) class MCPAttachResult:`
  - `server_name: str`
  - `transport: str`
  - `attached: bool`  
  - `already_attached: bool`
  - `tools_added: list[str]`
  - `prompts_added: list[str]`
  - `warnings: list[str]`

- `@dataclass(frozen=True, slots=True) class MCPDetachResult:`
  - `server_name: str`
  - `detached: bool`
  - `tools_removed: list[str]`
  - `prompts_removed: list[str]`

## McpAgent methods

Add to `McpAgent`:
- `async def attach_mcp_server(self, *, server_name: str, server_config: MCPServerSettings | None = None, options: MCPAttachOptions | None = None) -> MCPAttachResult`
- `async def detach_mcp_server(self, server_name: str) -> MCPDetachResult`
- `def list_attached_mcp_servers(self) -> list[str]`

Also extend `McpAgentProtocol` in `src/fast_agent/mcp/types.py` with these capabilities.

---

## Aggregator changes

## Attach behavior

Add `MCPAggregator.attach_server(...)`:
1. Validate context + server registry.
2. If `server_config` provided, upsert into `context.server_registry.registry[server_name]`.
3. If server already attached:
   - if `force_reconnect`, reconnect and refresh tool/prompt indexes;
   - else return `already_attached=True`.
4. Connect server with options (timeout, oauth policy).
5. Fetch tools/prompts for that server only.
6. Merge into:
   - `self.server_names`
   - `_namespaced_tool_map`
   - `_server_to_tool_map`
   - `_prompt_cache`
   - `_skybridge_configs`
7. Emit progress events and return detailed `MCPAttachResult`.

## Detach behavior

Add `MCPAggregator.detach_server(server_name)`:
1. Validate server is attached.
2. Disconnect via connection manager (persistent mode) if running.
3. Remove server entries from in-memory maps/caches.
4. Remove server from `server_names`.
5. Keep server config in registry (per decision #2).
6. Return `MCPDetachResult` with removed tool/prompt names.

## Listing behavior

Add helper:
- `def list_attached_servers(self) -> list[str]`
- `def list_configured_detached_servers(self) -> list[str]` (for connect help)

`configured_detached = registry.keys() - attached`

---

## Connection manager changes

## Per-attach startup timeout

`MCPConnectionManager.get_server(...)` and/or `launch_server(...)` should accept startup timeout.

Implementation:
- pass optional timeout through to `ServerConnection.wait_for_initialized()` call path via `asyncio.wait_for(...)` (or `anyio.fail_after`), default 10s for runtime attach APIs.
- on timeout, request shutdown and raise a clear `ServerInitializationError` message.

## Per-attach OAuth policy

Refactor `_prepare_headers_and_auth(server_config)` to:
- `_prepare_headers_and_auth(server_config, *, trigger_oauth: bool = True)`.
- if `trigger_oauth=False`, never build OAuth provider.

Thread this through connection manager launch path.

---

## Command system changes

## TUI command payloads

In `src/fast_agent/ui/command_payloads.py` add:
- `McpListCommand`
- `McpConnectCommand`
- `McpDisconnectCommand`

Include typed fields:
- raw target text
- parsed connection mode (`"url" | "npx" | "uvx" | "stdio"`)
- `server_name` optional
- timeout/oauth/reconnect flags

## TUI parser/completion/help

In `src/fast_agent/ui/enhanced_prompt.py`:
- Extend `parse_special_input` for `/mcp ...` and `/connect ...`.
- Keep `/mcp` no-args backward-compatible.
- Add completions for:
  - `/mcp` subcommands `list`, `connect`, `disconnect`
  - disconnect server-name suggestions from attached servers
- Update help text to document new syntax.

In `src/fast_agent/ui/interactive_prompt.py`:
- Handle new command payloads.
- Route to new command handlers (new module `commands/handlers/mcp_runtime.py`).

## ACP slash commands

In `src/fast_agent/acp/slash_commands.py`:
- Add `mcp` to `_session_commands`.
- Add `_handle_mcp(arguments: str)`.
- Subcommands:
  - `list`
  - `connect ...`
  - `disconnect <name>`

In `execute_command`, wire `command_name == "mcp"`.

---

## Runtime operation wiring in app/server layers

To avoid command handlers reaching into private attributes, add explicit callbacks, analogous to existing card/tool callbacks.

## AgentApp callbacks (`src/fast_agent/core/agent_app.py`)
Add callback plumbing:
- `set_attach_mcp_server_callback`
- `set_detach_mcp_server_callback`
- `set_list_attached_mcp_servers_callback`
- `set_list_configured_mcp_servers_callback` (for connect suggestions)

Expose methods:
- `attach_mcp_server(agent_name, server_name, server_config, options)`
- `detach_mcp_server(agent_name, server_name)`
- list methods

## FastAgent wiring (`src/fast_agent/core/fastagent.py`)
During wrapper setup (near existing `set_attach_agent_tools_callback`):
- provide runtime MCP callbacks that resolve the target agent instance and call new `McpAgent` methods.
- if agent is not MCP-capable, return friendly error.

## ACP server wiring (`src/fast_agent/acp/server/agent_acp_server.py`)
Add corresponding callback support passed into `SlashCommandHandler`, similar to agent-card callback pattern.

Given decision #3 (shared scope affects all sessions):
- callback should mutate shared instance state directly.
- after connect/disconnect, trigger:
  - `send_available_commands_update()` (if needed)
  - instruction cache refresh/update for affected agent(s).

---

## Instruction and cache refresh

After successful attach/detach:
1. Rebuild agent instruction (for `{{serverInstructions}}`) via `rebuild_agent_instruction(...)`.
2. If ACP context is present, update session instruction cache:
   - call `slash_handler.update_session_instruction(...)` and/or ACPContext cache path.

This prevents stale `/status system` output and stale prompt-time system prompts.

---

## Connect/disconnect output requirements

## Required user feedback

After connect, show:
- server connected message
- transport and server name
- `tools_added` list
- `prompts_added` list
- configured-but-detached options (for discoverability)

After disconnect, show:
- server detached
- tools/prompts removed counts/names

For status/list:
- `/mcp list` remains attached-only.
- `/mcp connect` with no args should show usage + available configured detached servers.

---

## Error handling/recovery policy

1. **Startup timeout** (new): fail fast with clear message including timeout seconds.
2. **OAuth deferred** (`trigger_oauth=False`): return explicit auth-needed error guidance if connect fails due to missing auth.
3. **Reconnect behavior**:
   - keep existing session-terminated and connection-error reconnect logic.
   - runtime attach can optionally override reconnect-on-disconnect policy in-memory for that server config instance.
4. **No infinite reconnect loops**: preserve current single retry safeguards.

---

## Detailed implementation steps

## Phase 1: Core runtime attach/detach
1. Add typed option/result dataclasses.
2. Implement `MCPAggregator.attach_server`, `detach_server`, listing helpers.
3. Add connection manager support for startup timeout + oauth policy.
4. Add `McpAgent` wrapper methods and protocol updates.

## Phase 2: App/ACP callback wiring
1. Add new callbacks and methods in `AgentApp`.
2. Wire callbacks in `FastAgent` wrapper construction.
3. Add ACP server callback plumbing into `SlashCommandHandler`.

## Phase 3: Commands (TUI + ACP)
1. Add new command payloads.
2. Add parser + completions + help text updates in enhanced prompt.
3. Add new command handlers module (`commands/handlers/mcp_runtime.py`).
4. Wire interactive prompt match cases.
5. Add ACP `/mcp` command execution handler and markdown rendering.

## Phase 4: cache refresh and UX polish
1. Rebuild instruction on attach/detach.
2. Update ACP session instruction cache.
3. Ensure progress events are emitted for connect/disconnect lifecycle.
4. Finalize output text and warnings.

---

## Testing plan

## Unit tests
- `tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py`
  - attach new server
  - attach already attached
  - detach existing/non-existing
  - map/cache updates
- `tests/unit/fast_agent/mcp/test_mcp_connection_manager_attach_options.py`
  - startup timeout behavior
  - oauth trigger flag behavior
- `tests/unit/fast_agent/ui/test_parse_mcp_commands.py`
  - `/mcp list|connect|disconnect`
  - `/connect` autodetect url/npx/uvx/stdio
- `tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py`
- `tests/unit/fast_agent/core/test_agent_app_mcp_callbacks.py`

## Integration tests
- `tests/integration/acp/test_acp_slash_commands.py`
  - add `/mcp` command tests (list/connect/disconnect)
- TUI-focused integration (if present) for end-to-end connect/disconnect output.

## Test constraints
- No mocks/monkeypatch (project rule): use real lightweight test servers/fixtures and fake in-process structs where needed.

---

## Validation commands after implementation

Run:
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- `uv run pytest tests/unit`

Run integration tests relevant to MCP runtime attach/detach paths.
Ask operator to run e2e as needed.

---

## Risks and mitigations

- **Shared-scope side effects in ACP:** clearly document that runtime MCP changes are shared.
- **Instruction cache staleness:** enforce rebuild + cache update in one helper used by all connect/disconnect entry points.
- **Connection hangs:** enforce low timeout defaults on runtime attach.
- **Command parsing ambiguity:** keep `/mcp connect` explicit; `/connect` alias limited to connect flow only.

---

## Follow-up (optional, post-MVP)

1. Persist runtime attach choices to optional session metadata.
2. Add command to promote runtime-attached server config into file config.
3. Add richer `/mcp connect` interactive picker for configured detached servers.
4. Add ACP client capability-aware OAuth prompt formatting.
