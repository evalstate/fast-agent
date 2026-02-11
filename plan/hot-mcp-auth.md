# Runtime MCP OAuth UX + Timeout Separation Plan (`hot-mcp-auth`)

Date: 2026-02-10

## Problem statement

Runtime MCP connect currently applies a single startup timeout to the full connect path. For remote OAuth flows, user authorization time is counted against startup time, causing false failures like:

- `Startup timed out after 10.0s`

Additionally, in ACP mode, OAuth links are emitted to server console output, not reliably surfaced to the connected ACP client.

## Goals

1. **Separate machine startup timeout from human OAuth wait time**.
2. **Preserve runtime connect timeout behavior for non-OAuth startup hangs**.
3. **Surface OAuth links properly in ACP**:
   - send as live progress updates
   - include link in slash-command response content
4. Keep TUI behavior friendly and explicit (link visible, timer paused while waiting for callback).

## Non-goals

- No config file mutation or persistence behavior changes.
- No changes to agent-card format.
- No redesign of MCP SDK OAuth internals beyond integration hooks.

---

## Design overview

Introduce a small OAuth event channel that flows through runtime attach APIs:

`/mcp connect (TUI/ACP)`
→ `commands.handlers.mcp_runtime`
→ `MCPAttachOptions`
→ `MCPAggregator.attach_server`
→ `MCPConnectionManager.get_server/launch_server`
→ `build_oauth_provider(..., event_handler=...)`
→ OAuth lifecycle events emitted from `oauth_client.py`

Connection manager uses these events to track **OAuth waiting windows** and excludes those windows from startup timeout accounting.

---

## Detailed implementation plan

## 1) Add OAuth lifecycle event model and callback plumbing

### Files
- `src/fast_agent/mcp/oauth_client.py`
- `src/fast_agent/mcp/mcp_connection_manager.py`
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/mcp/types.py` (if protocol typing needs exposure)

### Changes
1. Add typed event model in `oauth_client.py` (or shared MCP typing module), e.g.:
   - `OAuthEventType = Literal["authorization_url", "wait_start", "wait_end", "oauth_error"]`
   - dataclass payload with fields like `server_name`, `url`, `message`, timestamps.
2. Extend `build_oauth_provider(...)` signature with optional async event callback.
3. Emit events at key points:
   - just before/after showing authorization URL
   - just before waiting for callback
   - immediately after callback resolves/fails/times out
4. Keep existing console output as default fallback when no callback is supplied.

### Notes
- Event callback should be best-effort: exceptions in callback must not break OAuth flow.
- Keep URL emission single-source to avoid duplicate/cross-channel divergence.

---

## 2) Separate timeout budgets in connection manager

### Files
- `src/fast_agent/mcp/mcp_connection_manager.py`

### Changes
1. Extend `ServerConnection` with OAuth wait bookkeeping:
   - `_oauth_wait_active: bool`
   - `_oauth_wait_started_at: float | None`
   - `_oauth_wait_accumulated_seconds: float`
   - optional helper methods `mark_oauth_wait_start/end()`
2. Register OAuth event handler from connection manager into `build_oauth_provider(...)`.
3. Replace one-shot `asyncio.wait_for(server_conn.wait_for_initialized(), startup_timeout)` with budget loop:
   - poll initialized event at short interval (e.g., 100ms)
   - compute elapsed machine startup time as wall time minus accumulated OAuth wait
   - timeout only on machine startup budget exhaustion
4. Keep OAuth callback timeout separate (currently 300s, optionally configurable later).
5. Improve timeout error messaging:
   - startup timeout: mention non-OAuth startup timeout
   - OAuth callback timeout: explicit OAuth authorization timeout guidance

### Acceptance criteria
- User can spend >10s authorizing without triggering startup timeout.
- Non-OAuth startup still times out at configured `--timeout`.

---

## 3) Ensure async responsiveness during callback wait

### Files
- `src/fast_agent/mcp/oauth_client.py`

### Changes
1. In async callback handler, move blocking callback wait (`server.wait(...)`) to `asyncio.to_thread(...)`.
2. Emit `wait_start` before entering thread wait, `wait_end` in `finally`.
3. Preserve paste-URL fallback behavior and emit matching events when fallback path is used.

### Rationale
Avoid blocking event loop while waiting for human callback; enables progress/event delivery and cleaner ACP behavior.

---

## 4) Extend runtime attach options to carry OAuth event handling

### Files
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py`
- `src/fast_agent/core/agent_app.py` (if callback shaping requires changes)

### Changes
1. Extend `MCPAttachOptions` with optional OAuth event callback field.
2. Thread through:
   - `handle_mcp_connect(...)` → `MCPAttachOptions`
   - `MCPAggregator.attach_server(...)` → manager `get_server/reconnect_server`
3. Maintain default behavior for existing call sites (`None` callback).

---

## 5) TUI UX behavior

### Files
- `src/fast_agent/ui/interactive_prompt.py` (optional status polish)
- `src/fast_agent/commands/handlers/mcp_runtime.py`

### Changes
1. Keep clickable link output in local TUI via existing rich console behavior.
2. Add progress line(s) around OAuth wait when event callback present:
   - “Open this link to authorize: <url>”
   - “Waiting for OAuth callback (startup timer paused)…”
3. Ensure final success/failure messages remain unchanged except for improved timeout diagnostics.

---

## 6) ACP UX behavior (critical)

### Files
- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py`

### Changes
1. In ACP `/mcp connect`, pass an OAuth-aware progress/event hook into command handler.
2. On `authorization_url` event:
   - send immediate session progress update via `_send_progress_update(...)`
   - record link for command outcome so slash response body also contains it
3. On `wait_start` / `wait_end` events:
   - emit progress status text so user sees flow state in client
4. Render final slash response with an explicit message section containing the OAuth link (if one was used), e.g.:
   - `OAuth authorization link: <url>`

### Important behavior target
ACP users should not need server logs/terminal access to complete OAuth.

---

## 7) Error handling and user-facing messages

### Changes
1. Differentiate error classes/messages:
   - startup timeout (`--timeout`) exceeded while **not** in OAuth wait
   - OAuth callback timeout (authorization not completed in time)
2. Suggested actionable guidance:
   - startup timeout: “Try increasing --timeout or verify server/network startup.”
   - OAuth timeout: “Authorization was not completed in time; retry /mcp connect.”

---

## Suggested phases / slices

## Phase 1 — Core timeout decoupling + OAuth events

**Scope**
- OAuth event model + provider callback plumbing
- connection manager timeout budget separation
- no ACP-specific rendering yet

**Primary files**
- `mcp/oauth_client.py`
- `mcp/mcp_connection_manager.py`
- `mcp/mcp_aggregator.py`

**Tests**
- new/updated unit tests in `tests/unit/fast_agent/mcp/`
  - startup timeout excludes OAuth wait
  - startup timeout still triggers for non-OAuth hangs
  - reconnect path honors same logic

---

## Phase 2 — ACP link delivery + progress UX

**Scope**
- wire OAuth events into `/mcp connect` ACP path
- send link in progress + include in final slash response

**Primary files**
- `commands/handlers/mcp_runtime.py`
- `acp/slash_commands.py`

**Tests**
- `tests/unit/fast_agent/acp/test_slash_commands_mcp.py`
  - verifies progress updates emitted
  - verifies response includes OAuth link when auth flow is triggered

---

## Phase 3 — TUI polish + message harmonization

**Scope**
- optional TUI status line polish
- normalize wording across TUI/ACP errors

**Primary files**
- `commands/handlers/mcp_runtime.py`
- `ui/interactive_prompt.py` (if needed)

**Tests**
- `tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py`
  - verifies OAuth wait messaging and timeout text clarity

---

## Phase 4 — hardening / regression pass

**Scope**
- ensure no regressions for non-OAuth connect
- verify startup load path still unaffected
- tidy docs/help text if needed

**Validation**
- `uv run scripts/lint.py --fix`
- `uv run scripts/typecheck.py`
- targeted unit tests (pipe output to file and inspect tail)
- optional `uv run scripts/cpd.py --check`

---

## Suggested test matrix

### Unit tests
1. **Connection manager**
   - timeout paused during OAuth wait window
   - timeout counts again after wait_end
   - immediate timeout for pure startup stall
2. **OAuth client events**
   - URL event emitted once
   - wait_start/wait_end paired under success/failure paths
3. **Runtime handler**
   - event→message mapping for TUI and ACP callbacks
4. **ACP slash commands**
   - progress updates include authorization URL
   - final markdown response includes authorization URL message

### Manual verification (operator)
1. TUI: `/mcp connect https://... --timeout 10` with deliberate 20s authorize delay succeeds.
2. ACP: same flow, link visible in client updates and slash response.
3. Negative path: do not authorize; confirm OAuth timeout error (not startup timeout).

---

## Risks and mitigations

1. **Risk:** event callback failures break auth flow.
   - **Mitigation:** wrap callback calls in defensive try/except and log debug only.
2. **Risk:** duplicate link messages across channels.
   - **Mitigation:** centralize URL event emission and dedupe by last URL in command handler.
3. **Risk:** subtle timing bugs in budget loop.
   - **Mitigation:** deterministic unit tests with controlled event signaling.

---

## Definition of done

- OAuth authorization time no longer consumes MCP startup timeout budget.
- ACP users receive OAuth links in-client (progress + final slash command response).
- Timeout/error messages distinguish startup failures from uncompleted authorization.
- Lint/typecheck/tests pass for touched areas.
