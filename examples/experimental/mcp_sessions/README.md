# Experimental MCP Sessions demos

This directory contains demo servers and a client script demonstrating
**experimental MCP session v2** — application-level session state that is
decoupled from the underlying transport connection.

## Architecture

All servers share a common base (`_session_base.py`) that provides:

- `SessionStore` — in-memory session record management
- `session/create`, `session/list`, `session/delete` request handlers
- `ExperimentalServerSession` — ServerSession variant accepting session/* methods
- Transport runners for stdio and Streamable HTTP
- Cookie helpers for `_meta["mcp/session"]` echo

## Servers

### 1. `session_server.py` — Session probe (original demo)

The original demo server with a single `session_probe` tool that supports
`status`/`revoke`/`new` actions. Good for testing the basic cookie lifecycle.

### 2. `session_required_server.py` — Gatekeeper pattern

A minimal server that **rejects** any tool call with a JSON-RPC error
(`-32002`) if no session has been established. Demonstrates enforcing
session-first access control.

**Tools:** `echo(text)`, `whoami()`

### 3. `notebook_server.py` — Per-session notebook

Maintains a per-session list of notes. Each session has isolated storage.
Demonstrates stateful, multi-turn, session-scoped data.

**Tools:** `notebook_append(text)`, `notebook_read()`, `notebook_clear()`, `notebook_status()`

### 4. `hashcheck_server.py` — Per-session hash KV store

Hashes strings and stores digests in a per-session key-value store.
A reconnecting client with the same session cookie can verify strings
against previously stored hashes — the strongest demo of why
application-level sessions matter beyond transport-level `Mcp-Session-Id`.

**Tools:** `hashcheck_store(key, text)`, `hashcheck_verify(key, text)`, `hashcheck_list()`, `hashcheck_delete(key)`

## Running

### Any server via stdio

```bash
uv run python examples/experimental/mcp_sessions/<server>.py
```

### Any server via Streamable HTTP

```bash
uv run python examples/experimental/mcp_sessions/<server>.py \
  --transport streamable-http --host 127.0.0.1 --port 8765
```

### Demo client (against session_server.py)

```bash
# stdio (spawns server as subprocess)
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py

# or against a running HTTP server
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py \
  --transport http --url http://127.0.0.1:8765/mcp
```

## Session flow

```
Client                              Server
  |                                    |
  |---- initialize ------------------>|
  |<--- capabilities (experimental    |
  |      .session = {v2, features})   |
  |                                    |
  |---- session/create -------------->|
  |<--- _meta["mcp/session"] cookie   |
  |                                    |
  |---- tools/call ------------------>|
  |     + _meta["mcp/session"]        |
  |<--- result + updated cookie       |
  |                                    |
  |  ... disconnect / reconnect ...    |
  |                                    |
  |---- tools/call ------------------>|
  |     + _meta["mcp/session"] (same) |
  |<--- result (state preserved)      |
```

## What these demos prove

1. **Session enforcement**: Servers can require sessions before allowing tool access
2. **Session-scoped state**: Each session gets isolated data (notebook, KV store)
3. **Session persistence**: State survives across tool calls within a session
4. **Reconnection**: The hashcheck server demonstrates that a client reconnecting
   with the same session cookie retains access to previously stored data
5. **Cookie lifecycle**: Create -> echo -> update -> revoke -> re-establish
