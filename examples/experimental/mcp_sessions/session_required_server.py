"""Session-required MCP server (draft data-layer sessions).

Every tool call requires a valid session established via `sessions/create`.
"""

from __future__ import annotations

import argparse

import mcp.types as types
from _session_base import (
    SessionStore,
    add_transport_args,
    register_session_handlers,
    require_session,
    run_server,
    session_meta,
)
from mcp.server.fastmcp import Context, FastMCP


def build_server() -> FastMCP:
    mcp = FastMCP("session-required", log_level="WARNING")
    sessions = SessionStore(include_state=False)
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="echo")
    async def echo(ctx: Context, text: str) -> types.CallToolResult:
        """Echo text back — requires an active session."""
        record = require_session(ctx, sessions)
        sessions.touch(record)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="whoami")
    async def whoami(ctx: Context) -> types.CallToolResult:
        """Return basic session information."""
        record = require_session(ctx, sessions)
        sessions.touch(record)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Session {record.session_id}, calls={record.tool_calls}",
                )
            ],
            _meta=session_meta(record, sessions),
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
