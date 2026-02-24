"""Session-required MCP server.

A minimal server that **rejects** any tool call with a JSON-RPC error
if the client has not first established an experimental session via
``session/create``.

This demonstrates the "gatekeeper" pattern: the server advertises
``experimental.session`` and enforces that every request carries a valid
session cookie in ``_meta["mcp/session"]``.

Tools:
  - ``echo(text)`` — returns the text back; only works with a valid session.
  - ``whoami()``   — returns session metadata; only works with a valid session.
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


def _require_session(
    ctx: Context, store: SessionStore
) -> tuple[dict[str, Any], str]:
    """Extract and validate the session cookie.

    Returns ``(cookie, session_id)`` or raises ``McpError`` if no valid
    session is present.
    """
    cookie = session_cookie_from_meta(ctx.request_context.meta)
    session_id = session_id_from_cookie(cookie)

    if not session_id or store.get(session_id) is None:
        raise McpError(
            types.ErrorData(
                code=SESSION_REQUIRED_ERROR_CODE,
                message=(
                    "Session required. Send session/create before calling tools."
                ),
            )
        )
    return cookie, session_id  # type: ignore[return-value]


def build_server() -> FastMCP:
    mcp = FastMCP("session-required", log_level="WARNING")
    store = SessionStore()
    register_session_handlers(mcp._mcp_server, store)

    @mcp.tool(name="echo")
    async def echo(ctx: Context, text: str) -> types.CallToolResult:
        """Echo text back — requires an active session."""
        cookie, session_id = _require_session(ctx, store)
        record = store.get(session_id)
        if record:
            record.tool_calls += 1
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="whoami")
    async def whoami(ctx: Context) -> types.CallToolResult:
        """Return session metadata — requires an active session."""
        cookie, session_id = _require_session(ctx, store)
        record = store.get(session_id)
        info = {
            "session_id": session_id,
            "tool_calls": record.tool_calls if record else 0,
            "data": cookie.get("data", {}),
        }
        if record:
            record.tool_calls += 1
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Session: {session_id}, calls: {info['tool_calls']}",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-required MCP demo server"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
