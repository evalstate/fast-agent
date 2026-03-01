"""Notebook MCP server — per-session note storage (draft sessions)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

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


class NotebookStore:
    """Per-session notebook storage."""

    def __init__(self) -> None:
        self._notebooks: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, text: str) -> int:
        if session_id not in self._notebooks:
            self._notebooks[session_id] = []
        self._notebooks[session_id].append(
            {
                "text": text,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        return len(self._notebooks[session_id])

    def read(self, session_id: str) -> list[dict[str, str]]:
        return list(self._notebooks.get(session_id, []))

    def clear(self, session_id: str) -> int:
        notes = self._notebooks.pop(session_id, [])
        return len(notes)

    def count(self, session_id: str) -> int:
        return len(self._notebooks.get(session_id, []))



def build_server() -> FastMCP:
    mcp = FastMCP("notebook", log_level="WARNING")
    sessions = SessionStore(include_state=False)
    notebooks = NotebookStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="notebook_append")
    async def notebook_append(ctx: Context, text: str) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        count = notebooks.append(record.session_id, text)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Note added (#{count}): {text}",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="notebook_read")
    async def notebook_read(ctx: Context) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        notes = notebooks.read(record.session_id)
        if not notes:
            text = "(notebook is empty)"
        else:
            lines = [f"  {idx}. [{note['timestamp']}] {note['text']}" for idx, note in enumerate(notes, 1)]
            text = f"Notebook ({len(notes)} notes):\n" + "\n".join(lines)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="notebook_clear")
    async def notebook_clear(ctx: Context) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        removed = notebooks.clear(record.session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Notebook cleared ({removed} notes removed).",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="notebook_status")
    async def notebook_status(ctx: Context) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        count = notebooks.count(record.session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"Session {record.session_id}: "
                        f"{count} notes, {record.tool_calls} calls"
                    ),
                )
            ],
            _meta=session_meta(record, sessions),
        )

    return mcp



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Notebook MCP demo server (per-session notes)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
