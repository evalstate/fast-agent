"""Client-notes MCP server — note history encoded in session `state`.

This demo intentionally keeps notebook data client-side by round-tripping
notes through the protocol session state token:

- server validates session IDs server-side;
- note payload lives in `_meta[io.modelcontextprotocol/session].state`;
- server decodes incoming state, applies operation, returns updated state.
"""

from __future__ import annotations

import argparse
import base64
import json

import mcp.types as types
from _session_base import (
    CreateSessionRequest,
    SessionRecord,
    SessionStore,
    add_transport_args,
    register_session_handlers,
    require_session,
    run_server,
    session_metadata_from_meta,
)
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.exceptions import McpError


def _encode_notes_state(notes: list[str]) -> str:
    payload = json.dumps({"notes": notes}, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")



def _decode_notes_state(state: str | None) -> list[str]:
    if not isinstance(state, str) or not state:
        return []
    try:
        decoded = base64.b64decode(state.encode("ascii"), validate=True).decode("utf-8")
        payload = json.loads(decoded)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list):
        return []
    return [item for item in raw_notes if isinstance(item, str)]



def _incoming_state(ctx: Context, record: SessionRecord) -> str | None:
    metadata = session_metadata_from_meta(ctx.request_context.meta)
    if isinstance(metadata, dict):
        value = metadata.get("state")
        if isinstance(value, str) and value:
            return value
    return record.state if isinstance(record.state, str) else None



def _session_meta(record: SessionRecord, state: str) -> dict[str, dict[str, str]]:
    record.state = state
    return {
        "io.modelcontextprotocol/session": {
            "sessionId": record.session_id,
            "state": state,
            "expiresAt": record.expires_at,
        }
    }



def build_server() -> FastMCP:
    mcp = FastMCP("client-notes", log_level="WARNING")
    sessions = SessionStore(include_state=True)
    lowlevel = mcp._mcp_server
    register_session_handlers(lowlevel, sessions)

    async def _handle_session_create(_request: CreateSessionRequest) -> types.ServerResult:
        if session_metadata_from_meta(lowlevel.request_context.meta) is not None:
            raise McpError(
                types.ErrorData(
                    code=-32602,
                    message=(
                        "sessions/create MUST NOT include "
                        "_meta['io.modelcontextprotocol/session']"
                    ),
                )
            )

        record = sessions.create(reason="sessions/create")
        record.state = _encode_notes_state([])
        return types.ServerResult(types.EmptyResult(session=sessions.to_metadata(record)))

    lowlevel.request_handlers[CreateSessionRequest] = _handle_session_create

    @mcp.tool(name="client_notes_add")
    async def client_notes_add(ctx: Context, text: str) -> types.CallToolResult:
        """Append a note in client-managed session state."""
        record = require_session(ctx, sessions)
        notes = _decode_notes_state(_incoming_state(ctx, record))
        notes.append(text)
        record.tool_calls += 1
        state = _encode_notes_state(notes)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Added note #{len(notes)}: {text}")],
            _meta=_session_meta(record, state),
        )

    @mcp.tool(name="client_notes_list")
    async def client_notes_list(ctx: Context) -> types.CallToolResult:
        """List notes from client-managed session state."""
        record = require_session(ctx, sessions)
        notes = _decode_notes_state(_incoming_state(ctx, record))
        record.tool_calls += 1
        if notes:
            rendered = "\n".join(f"  {idx}. {note}" for idx, note in enumerate(notes, 1))
            text = f"Client notes ({len(notes)} items):\n{rendered}"
        else:
            text = "(client notes are empty)"
        state = _encode_notes_state(notes)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=_session_meta(record, state),
        )

    @mcp.tool(name="client_notes_clear")
    async def client_notes_clear(ctx: Context) -> types.CallToolResult:
        """Clear notes stored in client-managed session state."""
        record = require_session(ctx, sessions)
        notes = _decode_notes_state(_incoming_state(ctx, record))
        removed = len(notes)
        record.tool_calls += 1
        state = _encode_notes_state([])
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Cleared {removed} client notes.")],
            _meta=_session_meta(record, state),
        )

    @mcp.tool(name="client_notes_status")
    async def client_notes_status(ctx: Context) -> types.CallToolResult:
        """Show session status and decoded client note count."""
        record = require_session(ctx, sessions)
        notes = _decode_notes_state(_incoming_state(ctx, record))
        record.tool_calls += 1
        state = _encode_notes_state(notes)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"Session {record.session_id}: "
                        f"{len(notes)} client notes, {record.tool_calls} calls"
                    ),
                )
            ],
            _meta=_session_meta(record, state),
        )

    return mcp



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Client-notes MCP demo server (state token notebook)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
