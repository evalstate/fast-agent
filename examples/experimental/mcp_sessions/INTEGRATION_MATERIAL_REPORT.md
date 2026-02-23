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

- **`docs/docs/mcp/index.md`**
  - Includes config guidance for optional client-side session capability advertisement

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
```

---

## 6) Current caveats

- This is explicitly **experimental** session material.
- `session/*` support still relies on local request-union extensions in the demo
  server bridge (`ExperimentalServerSession` path), which is intentional for experimentation.
- `scripts/cpd.py --check` currently reports pre-existing repository duplications not
  specific to this session material.

