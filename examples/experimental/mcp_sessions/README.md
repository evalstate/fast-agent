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

### 4. `hashcheck_server.py` — Cookie-carried hash KV store

Hashes strings and stores digests in the session cookie payload itself
(`_meta["mcp/session"].data`). A reconnecting client with the same cookie can
verify strings against previously stored hashes without relying on server-side
in-memory session records.

**Tools:** `hashcheck_store(key, text)`, `hashcheck_verify(key, text)`, `hashcheck_list()`, `hashcheck_delete(key)`

### 5. `selective_session_server.py` — Mixed public + session-only tools

Demonstrates **selective** session enforcement. Public tools work without a
session, while session-scoped tools return a session-required error when the
cookie is missing/invalid.

**Tools:** `public_echo(text)`, `session_start(label?)`, `session_reset()`, `session_counter_inc()`, `session_counter_get()`

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

# optionally advertise session capability from the client during initialize
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py \
  --advertise-session-capability

# run all demo variants (including selective policy example)
uv run python examples/experimental/mcp_sessions/demo_all_sessions.py

# run only the selective-policy demo
uv run python examples/experimental/mcp_sessions/demo_all_sessions.py selective
```

## fast-agent environment (server cards per scenario)

This directory now includes a ready-to-run fast-agent environment:

- Environment root: `examples/experimental/mcp_sessions/demo/`
- Agent cards: `examples/experimental/mcp_sessions/demo/agent-cards/`
- MCP server wiring: `examples/experimental/mcp_sessions/demo/fastagent.config.yaml`

Cards included:

- `probe` (default)
- `session_required`
- `notebook`
- `check_kv`
- `session_selective`

From the repository root:

```bash
# run the default card (probe)
uv run fast-agent go \
  --env examples/experimental/mcp_sessions/demo \
  --message 'Call session_probe with action=status and note=first'

# target a specific scenario card
uv run fast-agent go \
  --env examples/experimental/mcp_sessions/demo \
  --agent session_selective \
  --message 'Start a session labeled demo and then increment the session counter'

uv run fast-agent go \
  --env examples/experimental/mcp_sessions/demo \
  --agent session_selective \
  --message 'Get the current session counter value'
```

Notes:

- These cards use `model: $system.demo`.
- `demo/fastagent.config.yaml` defines `model_aliases.system.demo: kimi`.
- Run interactive mode (omit `--message`) and use `/mcp` to inspect `exp sess` and cookie state.

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
2. **Session-scoped state**: Each session gets isolated data (notebook, cookie-carried KV store)
3. **Session persistence**: State survives across tool calls within a session
4. **Reconnection**: The hashcheck server demonstrates that a client reconnecting
   with the same session cookie payload retains access to previously stored data
5. **Cookie lifecycle**: Create -> echo -> update -> revoke -> re-establish
6. **Selective enforcement**: `selective_session_server.py` shows that servers can
   require sessions only for specific tools while allowing others without sessions
