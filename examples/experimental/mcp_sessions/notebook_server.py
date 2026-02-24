"""Notebook MCP server — per-session note storage.

A server that maintains a per-session notebook. Each session has its own
isolated list of notes. Demonstrates stateful, multi-turn, session-scoped
data via the experimental session cookie mechanism.

Tools:
  - ``notebook_append(text)`` — append a note to the session's notebook.
  - ``notebook_read()``       — read all notes in the session's notebook.
  - ``notebook_clear()``      — clear the session's notebook.
  - ``notebook_status()``     — show note count and session info.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
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


class NotebookStore:
    """Per-session notebook storage."""

    def __init__(self) -> None:
        self._notebooks: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, text: str) -> int:
        """Append a note; returns the new note count."""
        if session_id not in self._notebooks:
            self._notebooks[session_id] = []
        entry = {
            "text": text,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._notebooks[session_id].append(entry)
        return len(self._notebooks[session_id])

    def read(self, session_id: str) -> list[dict[str, str]]:
        return list(self._notebooks.get(session_id, []))

    def clear(self, session_id: str) -> int:
        """Clear notebook; returns the number of notes removed."""
        notes = self._notebooks.pop(session_id, [])
        return len(notes)

    def count(self, session_id: str) -> int:
        return len(self._notebooks.get(session_id, []))


def _require_session(
    ctx: Context, store: SessionStore
) -> tuple[dict[str, Any], str]:
    """Extract and validate the session cookie, or raise McpError."""
    cookie = session_cookie_from_meta(ctx.request_context.meta)
    session_id = session_id_from_cookie(cookie)
    if not session_id or store.get(session_id) is None:
        raise McpError(
            types.ErrorData(
                code=SESSION_REQUIRED_ERROR_CODE,
                message="Session required. Send session/create before using the notebook.",
            )
        )
    return cookie, session_id  # type: ignore[return-value]


def build_server() -> FastMCP:
    mcp = FastMCP("notebook", log_level="WARNING")
    sessions = SessionStore()
    notebooks = NotebookStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="notebook_append")
    async def notebook_append(
        ctx: Context, text: str
    ) -> types.CallToolResult:
        """Append a note to the session's notebook."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        count = notebooks.append(session_id, text)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Note added (#{count}): {text}",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="notebook_read")
    async def notebook_read(ctx: Context) -> types.CallToolResult:
        """Read all notes in the session's notebook."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        notes = notebooks.read(session_id)
        if not notes:
            text = "(notebook is empty)"
        else:
            lines = []
            for i, note in enumerate(notes, 1):
                lines.append(f"  {i}. [{note['timestamp']}] {note['text']}")
            text = f"Notebook ({len(notes)} notes):\n" + "\n".join(lines)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="notebook_clear")
    async def notebook_clear(ctx: Context) -> types.CallToolResult:
        """Clear all notes in the session's notebook."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        removed = notebooks.clear(session_id)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Notebook cleared ({removed} notes removed).",
                )
            ],
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="notebook_status")
    async def notebook_status(ctx: Context) -> types.CallToolResult:
        """Show notebook status and session info."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        count = notebooks.count(session_id)
        info = {
            "session_id": session_id,
            "note_count": count,
            "tool_calls": record.tool_calls if record else 0,
        }
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Session {session_id}: {count} notes, {info['tool_calls']} calls",
                )
            ],
            _meta=cookie_meta(cookie),
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
