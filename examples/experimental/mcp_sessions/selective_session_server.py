"""Selective session MCP server (draft sessions).

- `public_echo` works with or without a session.
- `session_*` tools require a valid session.
"""

from __future__ import annotations

import argparse

import mcp.types as types
from _session_base import (
    SESSION_META_KEY,
    SessionStore,
    add_transport_args,
    optional_session,
    register_session_handlers,
    require_session,
    run_server,
    session_meta,
)
from mcp.server.fastmcp import Context, FastMCP


class SessionCounterStore:
    """Per-session counter storage."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def increment(self, session_id: str) -> int:
        count = self._counts.get(session_id, 0) + 1
        self._counts[session_id] = count
        return count

    def get(self, session_id: str) -> int:
        return self._counts.get(session_id, 0)

    def reset(self, session_id: str) -> None:
        self._counts.pop(session_id, None)



def _resolve_session_title(label: str | None) -> str:
    if label and label.strip():
        return label.strip()
    return "selective-session"



def build_server() -> FastMCP:
    mcp = FastMCP("selective-session", log_level="WARNING")
    sessions = SessionStore(include_state=False)
    counters = SessionCounterStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="public_echo")
    async def public_echo(ctx: Context, text: str) -> types.CallToolResult:
        """Always works; session is optional."""
        record = optional_session(ctx, sessions)
        meta: dict[str, object] | None = None
        if record is not None:
            sessions.touch(record)
            meta = session_meta(record, sessions)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"public tool ok: {text}",
                )
            ],
            _meta=meta,
        )

    @mcp.tool(name="session_start")
    async def session_start(label: str | None = None) -> types.CallToolResult:
        """Create a fresh session via tool call."""
        record = sessions.create(title=_resolve_session_title(label), reason="tool/session_start")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"new session started: {record.session_id}",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="session_reset")
    async def session_reset(ctx: Context) -> types.CallToolResult:
        """Delete active session and reset per-session counter."""
        record = require_session(ctx, sessions)
        counters.reset(record.session_id)
        _ = sessions.delete(record.session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"session reset: deleted {record.session_id}",
                )
            ],
            _meta={
                SESSION_META_KEY: {
                    "sessionId": record.session_id,
                }
            },
        )

    @mcp.tool(name="session_counter_inc")
    async def session_counter_inc(ctx: Context) -> types.CallToolResult:
        """Requires an active session; increments per-session counter."""
        record = require_session(ctx, sessions)
        sessions.touch(record)
        value = counters.increment(record.session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"session counter for {record.session_id}: {value}",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="session_counter_get")
    async def session_counter_get(ctx: Context) -> types.CallToolResult:
        """Requires an active session; reads per-session counter."""
        record = require_session(ctx, sessions)
        sessions.touch(record)
        value = counters.get(record.session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"session counter for {record.session_id}: {value}",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    return mcp



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selective session MCP demo server (public + session-only tools)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
