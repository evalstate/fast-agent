# Experimental MCP Sessions demo

This example demonstrates the **experimental session v2** flow end-to-end:

- server advertises `experimental.session` capability (`version=2`)
- fast-agent sends `session/create` automatically on initialize
- session cookie is echoed in `_meta["mcp/session"]` on tool responses
- server can revoke the cookie (`_meta["mcp/session"] = null`)
- `/mcp`-style status output shows session capability and cookie details

## Files

- `session_server.py` – MCP stdio server implementing:
  - `session/create`
  - `session/list`
  - `session/delete`
  - demo tool `session_probe`
- `demo_fast_agent_sessions.py` – fast-agent runtime demo script (no LLM key required)

## Run the demo

From repo root:

```bash
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py
```

Or against a separately running Streamable HTTP server:

```bash
uv run python examples/experimental/mcp_sessions/session_server.py --transport streamable-http --host 127.0.0.1 --port 8765
```

Then in another terminal:

```bash
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py --transport http --url http://127.0.0.1:8765/mcp
```

You should see status snapshots that include fields similar to:

- `exp sess: enabled (create, delete, list)`
- `cookie: {"id": "sess-...", "data": {"title": ...}}`
- after revoke, `cookie: none`
- next call establishes/echoes a new cookie

## Run server standalone (for manual testing)

```bash
uv run python examples/experimental/mcp_sessions/session_server.py
```

(The server uses MCP stdio transport.)

HTTP mode:

```bash
uv run python examples/experimental/mcp_sessions/session_server.py --transport streamable-http --host 127.0.0.1 --port 8765
```
