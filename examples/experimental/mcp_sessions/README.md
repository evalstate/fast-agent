# Experimental MCP Data-Layer Sessions (draft)

This directory contains draft-aligned session demos for MCP Data-Layer Sessions.

## Core reference pair

- `session_server.py` — minimal reference server
- `demo_fast_agent_sessions.py` — minimal reference client demo via `MCPAggregator`

The reference pair demonstrates:

- `capabilities.sessions = {}`
- `sessions/create`
- `sessions/delete`
- `_meta["io.modelcontextprotocol/session"]` request/response flow
- `-32043 Session not found`

## Additional demo servers

Shared helpers live in `_session_base.py`.

- `session_required_server.py` — all tools require session
  - tools: `echo`, `whoami`
- `notebook_server.py` — per-session note storage
  - tools: `notebook_append`, `notebook_read`, `notebook_clear`, `notebook_status`
- `hashcheck_server.py` — per-session hash set (no key argument)
  - tools: `hashcheck_store(text)`, `hashcheck_verify(text)`, `hashcheck_list()`, `hashcheck_delete(text)`
- `selective_session_server.py` — mixed public + session-only tools
  - tools: `public_echo`, `session_start`, `session_reset`, `session_counter_inc`, `session_counter_get`
- `client_notes_server.py` — client-managed notes encoded in session `state`
  - tools: `client_notes_add`, `client_notes_list`, `client_notes_clear`, `client_notes_status`

State behavior in this demo set is intentionally mixed:

- `session_server.py` and `client_notes_server.py` emit/update `state`
- `session_required`, `notebook`, `hashcheck`, and `selective` omit `state`

## Run

From repo root:

```bash
# Run the reference server
uv run python examples/experimental/mcp_sessions/session_server.py

# Run the reference client (spawns server subprocess)
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py

# Optional: advertise client test capability (experimental/sessions)
uv run python examples/experimental/mcp_sessions/demo_fast_agent_sessions.py \
  --advertise-session-capability

# Run all scenario demos
uv run python examples/experimental/mcp_sessions/demo_all_sessions.py
```

## Simplifications (intentional)

For clarity and portability:

1. Demos are **stdio-focused**.
2. Session records are **in-memory**.
3. The server-side `state` token logic is intentionally simple.
4. Some tools (for example `session_reset`) are scenario conveniences layered on top of
   the protocol-level `sessions/delete` primitive.
