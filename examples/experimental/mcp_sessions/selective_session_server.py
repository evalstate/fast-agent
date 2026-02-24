"""Selective session MCP server.

Demonstrates mixed policy enforcement:

- Public tools can be called without a session.
- Session-scoped tools require a valid experimental session cookie and return
  a JSON-RPC session-required error when the cookie is missing/invalid.

This showcases that MCP servers can apply sessions selectively per tool,
rather than globally for all requests.
"""

from __future__ import annotations

import argparse
from typing import Any

import mcp.types as types
from _session_base import (
    SESSION_REQUIRED_ERROR_CODE,
    SessionStore,
    add_transport_args,
    cookie_meta,
    register_session_handlers,
    run_server,
    session_cookie_from_meta,
    session_id_from_cookie,
)
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.exceptions import McpError


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


def _require_session(ctx: Context, sessions: SessionStore) -> tuple[dict[str, Any], str]:
    cookie = session_cookie_from_meta(ctx.request_context.meta)
    session_id = session_id_from_cookie(cookie)
    if not session_id or sessions.get(session_id) is None:
        raise McpError(
            types.ErrorData(
                code=SESSION_REQUIRED_ERROR_CODE,
                message=(
                    "Session not found for this tool. Establish a session with session/create "
                    "first."
                ),
            )
        )
    normalized_cookie = dict(cookie) if isinstance(cookie, dict) else {"id": session_id}
    return normalized_cookie, session_id


def _session_meta_if_valid(ctx: Context, sessions: SessionStore) -> dict[str, Any] | None:
    cookie = session_cookie_from_meta(ctx.request_context.meta)
    session_id = session_id_from_cookie(cookie)
    if not session_id:
        return None
    if sessions.get(session_id) is None:
        return None
    return cookie_meta(cookie)


def _resolve_session_title(label: str | None) -> str:
    if label and label.strip():
        return label.strip()
    return "selective-session"


def build_server() -> FastMCP:
    mcp = FastMCP("selective-session", log_level="WARNING")
    sessions = SessionStore()
    counters = SessionCounterStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="public_echo")
    async def public_echo(ctx: Context, text: str) -> types.CallToolResult:
        """Always works; session is optional."""
        maybe_meta = _session_meta_if_valid(ctx, sessions)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"public tool ok: {text}",
                )
            ],
            _meta=maybe_meta,
        )

    @mcp.tool(name="session_start")
    async def session_start(label: str | None = None) -> types.CallToolResult:
        """Create a fresh session cookie explicitly via a tool call."""
        record = sessions.create(title=_resolve_session_title(label), reason="tool/session_start")
        cookie = sessions.to_cookie(record)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"new session started: {record.session_id}",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="session_reset")
    async def session_reset(ctx: Context) -> types.CallToolResult:
        """Delete active session (if present) and clear cookie metadata."""
        cookie = session_cookie_from_meta(ctx.request_context.meta)
        session_id = session_id_from_cookie(cookie)
        deleted = sessions.delete(session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"session reset: deleted {session_id}"
                        if deleted and session_id
                        else "session reset: no active session"
                    ),
                )
            ],
            _meta=cookie_meta(None),
        )

    @mcp.tool(name="session_counter_inc")
    async def session_counter_inc(ctx: Context) -> types.CallToolResult:
        """Requires an active session; increments per-session counter."""
        cookie, session_id = _require_session(ctx, sessions)
        value = counters.increment(session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"session counter for {session_id}: {value}",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="session_counter_get")
    async def session_counter_get(ctx: Context) -> types.CallToolResult:
        """Requires an active session; reads per-session counter."""
        cookie, session_id = _require_session(ctx, sessions)
        value = counters.get(session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"session counter for {session_id}: {value}",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selective session MCP server (public + session-only tools)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
