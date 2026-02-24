# MCP Sessions Integration Material Report

_Location:_ `examples/experimental/mcp_sessions/`  
_Date:_ 2026-02-23

This report summarizes exactly what integration and experimentation material exists
today for experimental MCP sessions in this repository.

## Executive summary

Yes — we have both:

1. **SDK-style MCP server code** to experiment with experimental session behavior.
2. **fast-agent integration code** that exercises the same behavior through
   `MCPAggregator` and `MCPAgentClientSession`.

You can test:

- server-first capability discovery,
- optional client-first capability advertisement (version-only hint),
- session-required gatekeeping,
- session-scoped state,
- selective (per-tool) session enforcement.

---

## 1) What exists in `examples/experimental/mcp_sessions/`

### A. Shared experimental session infrastructure

- **`_session_base.py`**
  - In-memory `SessionStore`
  - `session/create`, `session/list`, `session/delete` handlers
  - `ExperimentalServerSession` (custom request union that accepts `session/*`)
  - Transport runners for stdio and streamable-http
  - Cookie helpers for `_meta["mcp/session"]`

This file is the main **SDK-style experimentation scaffold** for server-side
session protocol wiring.

### B. Demo MCP servers

- **`session_server.py`**
  - Original probe server
  - Tool: `session_probe(action=status|revoke|new)`
  - Demonstrates cookie lifecycle and session rotation

- **`session_required_server.py`**
  - Global gatekeeper model
  - Rejects tool calls without valid session (`-32002`)
  - Tools: `echo`, `whoami`

- **`notebook_server.py`**
  - Session-scoped note storage
  - Tools: `notebook_append`, `notebook_read`, `notebook_clear`, `notebook_status`

- **`hashcheck_server.py`**
  - Session-scoped hash KV store
  - Tools: `hashcheck_store`, `hashcheck_verify`, `hashcheck_list`, `hashcheck_delete`

- **`selective_session_server.py`**
  - Mixed policy model (public + session-only tools)
  - Public tools: `public_echo`, `session_start`, `session_reset`
  - Session-required tools: `session_counter_inc`, `session_counter_get`
  - Demonstrates selective/exclusive session application per tool

### C. fast-agent-facing demo clients

- **`demo_fast_agent_sessions.py`**
  - End-to-end fast-agent demo against `session_server.py`
  - Uses `MCPAggregator` directly (no LLM API key needed)
  - Can run via stdio or HTTP
  - Supports `--advertise-session-capability` for client-first hint experiments

- **`demo_all_sessions.py`**
  - Runs multiple demos (including selective policy demo) in separate subprocesses
  - Exercises each server with scripted calls and status snapshots

- **`README.md`**
  - Usage docs and commands for all above material

### D. fast-agent environment + per-scenario server cards

- **`demo/fastagent.config.yaml`**
  - Configures one MCP server entry per scenario:
    - `mcp_sessions_probe`
    - `mcp_sessions_required`
    - `mcp_sessions_notebook`
    - `mcp_sessions_hashcheck`
    - `mcp_sessions_selective`
  - Uses stdio launches via `uv run python <server>.py`

- **`demo/agent-cards/`**
  - `probe` (default)
  - `session_required`
  - `notebook`
  - `check_kv`
  - `session_selective`
  - Each card is wired to exactly one scenario server for focused demos
  - Cards use `model: $system.demo`
  - `demo/fastagent.config.yaml` defines `model_aliases.system.demo: kimi`

---

## 2) fast-agent runtime integration material (outside example folder)

### A. Core client behavior

- **`src/fast_agent/mcp/mcp_agent_client_session.py`**
  - Detects server `experimental.session`
  - Optionally advertises client support (`experimental_session_advertise` + version)
  - Calls `session/create` when server supports and advertises `create`
  - Echoes and updates `_meta["mcp/session"]` across tool/resource/prompt requests
  - Handles revocation (`mcp/session = null`)

### B. Status/reporting support

- **`src/fast_agent/mcp/mcp_aggregator.py`**
  - Collects session support, features, cookie, title for server status

- **`src/fast_agent/ui/mcp_display.py`**
  - `/mcp` display includes `exp sess` and `cookie`

### C. Config surface

- **`src/fast_agent/config.py`** (`MCPServerSettings`)
  - `experimental_session_advertise: bool`
  - `experimental_session_advertise_version: int`

### D. Documentation

- **`examples/experimental/mcp_sessions/README.md`**
  - Includes server run commands, demo scripts, and fast-agent environment usage

---

## 3) Test coverage material

- **Unit**
  - `tests/unit/fast_agent/mcp/test_mcp_agent_client_session_sessions.py`
  - Covers capability capture, metadata merge/update, session create, advertise behavior

- **Integration**
  - `tests/integration/mcp_sessions/test_experimental_sessions.py`
  - Verifies auto-create, cookie echo, revocation, re-establishment via real example server

---

## 4) “Do we have SDK-style code to experiment with?”

**Yes.**

You can experiment at two levels:

1. **Pure MCP SDK server-side level**
   - Build/modify servers in this directory using `FastMCP` + low-level session bridge in `_session_base.py`.

2. **fast-agent client/runtime level**
   - Use `demo_fast_agent_sessions.py` / `demo_all_sessions.py` and observe behavior via status snapshots and `/mcp` rendering.

This gives a practical “spec + runtime” integration lab in-repo.

---

## 5) Quick run commands

```bash
# Single server demo (fast-agent client against session_server)
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py

# Same, with client-side session capability advertisement hint
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py \
  --advertise-session-capability

# All demos
uv run python examples/experimental/mcp_sessions/demo_all_sessions.py

# Selective policy only
uv run python examples/experimental/mcp_sessions/demo_all_sessions.py selective

# Run any server directly
uv run python examples/experimental/mcp_sessions/selective_session_server.py

# Run default fast-agent environment card (probe)
uv run fast-agent go \
  --env examples/experimental/mcp_sessions/demo \
  --message 'Call session_probe with action=status and note=first'

# Run selective-policy card
uv run fast-agent go \
  --env examples/experimental/mcp_sessions/demo \
  --agent session_selective \
  --message 'Start a session labeled demo and increment the session counter'
```

---

## 6) Current caveats

- This is explicitly **experimental** session material.
- `session/*` support still relies on local request-union extensions in the demo
  server bridge (`ExperimentalServerSession` path), which is intentional for experimentation.
- `scripts/cpd.py --check` currently reports pre-existing repository duplications not
  specific to this session material.

---

## 7) Supplement (2026-02-24): Experimental session protocol contract

This supplement is intentionally scoped to **protocol/runtime semantics**.
It excludes display-only/UI formatting updates.

### 7.1 Cookie envelope and metadata key

- Session state is carried in `_meta["mcp/session"]`.
- Cookie payload shape:
  - `id: str` (required for a usable session)
  - `expiry: str | null` (optional lease end)
  - `data: object` (optional server-defined metadata; commonly title/createdAt)

### 7.2 Capability negotiation contract

- Server advertises experimental session support via
  `capabilities.experimental.session`.
- fast-agent currently treats **version 2** as the supported protocol version.
- If server advertises an unsupported version, the client ignores session capability
  for that connection.
- Client-side capability hinting is optional and controlled by:
  - `experimental_session_advertise`
  - `experimental_session_advertise_version`

### 7.3 Lifecycle + metadata propagation contract

- If server supports sessions and includes feature `create`, fast-agent may
  auto-establish a session (`session/create`) when no cookie is present.
- Outbound MCP requests merge the active cookie into `_meta["mcp/session"]`
  when not already set by the caller.
- Inbound responses update local cookie state when `_meta["mcp/session"]` is present:
  - object => replace active cookie with server-returned value
  - `null` => revoke/clear active cookie

### 7.4 Lease / expiry contract

- `expiry` is a **server lease hint**, not a hard client guarantee.
- Server is authoritative for validity and may reject a session before expiry
  (restart, eviction, revocation, policy change).
- Server **may extend lease** by returning a refreshed cookie (same id with later
  expiry, or rotated id) in `_meta["mcp/session"]`.
- Client should always treat server-returned cookie as canonical and replace local state.
- Recommended semantics:
  - `expiry` missing or `null` => non-expiring lease (conceptually `∞`)
  - `expiry` present => best-effort resumability window, server-enforced

### 7.5 Resume semantics

- Resume is semantic/session-cookie based (not transport `Mcp-Session-Id` based).
- fast-agent can restore a previously used cookie from its local jar before
  initialize/connect completes, reducing unnecessary `session/create` calls.
- When `/mcp session use <id>` is used:
  - client prefers canonical payload from `session/list` when available
  - otherwise falls back to `{"id": <id>}` and lets server accept/reject on use

### 7.6 Persistence contract (client jar)

- Cookie jar path: `.fast-agent/mcp-cookie.json`
- On-disk format version: `2`
- Records are keyed by server identity (initialize name) when available,
  otherwise server alias.
- Jar stores cookie history + active selection metadata (`last_used_id`, `updatedAt`, hash).

---

## 8) Protocol-level delta matrix (runtime capability changes)

The following summarizes protocol/runtime capability work that is now in-repo,
excluding display tweaks.

1. **Session capability parsing + version gating**
   - `src/fast_agent/mcp/mcp_agent_client_session.py`
   - Captures `experimental.session` capability, features, and version handling.

2. **Optional client capability advertisement**
   - `src/fast_agent/mcp/mcp_agent_client_session.py`
   - Injects client experimental session hint during initialize when configured.

3. **Automatic session establishment when supported**
   - `src/fast_agent/mcp/mcp_agent_client_session.py`
   - Calls `session/create` opportunistically when supported and feature `create` is advertised.

4. **Cookie propagation + canonical server update semantics**
   - `src/fast_agent/mcp/mcp_agent_client_session.py`
   - Merges outbound cookie metadata and applies inbound updates/revocation (`null`).

5. **Persistent session jar + identity-aware resume state**
   - `src/fast_agent/mcp/experimental_session_client.py`
   - Adds JSON-backed store (v2), per-identity records, active cookie selection,
     and disconnected-server cookie visibility.

6. **Connection bootstrap from jar cookies**
   - `src/fast_agent/mcp/mcp_aggregator.py`
   - Hydrates new MCP sessions with best-effort prior cookie before first calls.

7. **Programmatic session control surface**
   - `src/fast_agent/mcp/experimental_session_client.py`
   - `list_jar`, `list_server_cookies`, `create_session`, `resume_session`,
     `clear_cookie`, `clear_all_cookies`, `list_sessions`, `resolve_server_name`.

8. **Runtime command surface for operators**
   - `src/fast_agent/commands/handlers/mcp_runtime.py`
   - `src/fast_agent/acp/slash/handlers/mcp.py`
   - Exposes protocol operations via `/mcp session [list|jar|new|use|clear]`.

9. **Session termination handling**
   - `src/fast_agent/mcp/mcp_agent_client_session.py`
   - Maps terminated-session server errors into explicit runtime exceptions.

10. **Protocol lab servers for policy experimentation**
    - `examples/experimental/mcp_sessions/_session_base.py` and scenario servers
    - Covers global gatekeeping, selective enforcement, session-scoped state,
      revocation, and cookie rotation flows.

### 8.1 Important note about current demo expiry behavior

- Current demo `SessionStore` implementations stamp `expiry` and `createdAt`, but do
  not yet enforce TTL eviction/validation in `get/ensure_from_cookie`.
- This means demo expiry is currently illustrative metadata unless explicitly
  enforced by a server variant.
