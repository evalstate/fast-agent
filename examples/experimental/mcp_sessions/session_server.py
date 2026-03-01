"""Reference MCP data-layer sessions server (draft).

This implementation intentionally stays small and focused:

- Advertises `capabilities.sessions`.
- Implements `sessions/create` and `sessions/delete`.
- Requires `_meta["io.modelcontextprotocol/session"]` on tool calls.
- Echoes updated SessionMetadata on successful tool responses.
- Returns `-32043` when a session is unknown.

Notes:
- This is a draft-spec demo, not production hardening.
- It uses in-memory session records for clarity.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import anyio
import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import InitializationState, ServerSession
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.shared.session import BaseSession
from pydantic import BaseModel

if TYPE_CHECKING:
    from mcp.server.models import InitializationOptions
    from mcp.shared.message import SessionMessage

SESSION_META_KEY = "io.modelcontextprotocol/session"
SESSION_NOT_FOUND = -32043


class SessionMetadata(BaseModel):
    sessionId: str
    expiresAt: str | None = None
    state: str | None = None


class CreateSessionRequest(types.Request):
    method: Literal["sessions/create"] = "sessions/create"
    params: types.RequestParams | None = None


class DeleteSessionRequest(types.Request):
    method: Literal["sessions/delete"] = "sessions/delete"
    params: types.RequestParams | None = None


class DataLayerClientRequest(types.ClientRequest):
    root: types.ClientRequestType | CreateSessionRequest | DeleteSessionRequest


class DataLayerServerSession(ServerSession):
    """ServerSession variant that accepts sessions/* requests."""

    def __init__(
        self,
        read_stream: anyio.streams.memory.MemoryObjectReceiveStream[
            SessionMessage | Exception
        ],
        write_stream: anyio.streams.memory.MemoryObjectSendStream[SessionMessage],
        init_options: InitializationOptions,
    ) -> None:
        BaseSession.__init__(
            self,
            read_stream,
            write_stream,
            DataLayerClientRequest,
            types.ClientNotification,
        )
        self._initialization_state = InitializationState.NotInitialized
        self._init_options = init_options
        (
            self._incoming_message_stream_writer,
            self._incoming_message_stream_reader,
        ) = anyio.create_memory_object_stream(0)
        self._exit_stack.push_async_callback(
            lambda: self._incoming_message_stream_reader.aclose()
        )


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    state: str
    expires_at: str
    call_count: int = 0


@dataclass(slots=True)
class SessionStore:
    _sessions: dict[str, SessionRecord] = field(default_factory=dict)

    def create(self) -> SessionRecord:
        session_id = f"sess-{secrets.token_hex(6)}"
        record = SessionRecord(
            session_id=session_id,
            state=_encode_state(call_count=0),
            expires_at=_future_expiry(minutes=30),
            call_count=0,
        )
        self._sessions[session_id] = record
        return record

    def get(self, session_id: str | None) -> SessionRecord | None:
        if not isinstance(session_id, str) or not session_id:
            return None
        return self._sessions.get(session_id)

    def delete(self, session_id: str | None) -> bool:
        if not isinstance(session_id, str) or not session_id:
            return False
        return self._sessions.pop(session_id, None) is not None

    def metadata(self, record: SessionRecord) -> dict[str, str]:
        return {
            "sessionId": record.session_id,
            "state": record.state,
            "expiresAt": record.expires_at,
        }

    def touch(self, record: SessionRecord) -> None:
        record.call_count += 1
        record.state = _encode_state(call_count=record.call_count)
        record.expires_at = _future_expiry(minutes=30)



def _future_expiry(*, minutes: int) -> str:
    expiry = datetime.now(UTC) + timedelta(minutes=minutes)
    return expiry.replace(microsecond=0).isoformat().replace("+00:00", "Z")



def _encode_state(*, call_count: int) -> str:
    payload = json.dumps({"callCount": call_count}, separators=(",", ":"), sort_keys=True)
    # Use padded base64 to match the draft examples and keep tokens copy-pastable.
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")



def _meta_to_dict(meta: types.RequestParams.Meta | None) -> dict[str, Any]:
    if meta is None:
        return {}
    dumped = meta.model_dump(by_alias=True, exclude_none=False)
    return dumped if isinstance(dumped, dict) else {}



def _session_meta(meta: types.RequestParams.Meta | None) -> dict[str, Any] | None:
    payload = _meta_to_dict(meta)
    raw = payload.get(SESSION_META_KEY)
    if isinstance(raw, dict):
        return dict(raw)
    return None



def _session_id_from_meta(meta: types.RequestParams.Meta | None) -> str | None:
    session = _session_meta(meta)
    if not isinstance(session, dict):
        return None
    raw = session.get("sessionId")
    if isinstance(raw, str) and raw:
        return raw
    return None



def _raise_session_not_found(session_id: str | None) -> None:
    data = {"sessionId": session_id or ""}
    raise McpError(
        types.ErrorData(
            code=SESSION_NOT_FOUND,
            message="Session not found",
            data=data,
        )
    )



def _register_session_handlers(lowlevel: Any, store: SessionStore) -> None:
    async def handle_create(request: CreateSessionRequest) -> types.ServerResult:
        # Draft rule: create request must not include session metadata.
        if _session_meta(lowlevel.request_context.meta) is not None:
            raise McpError(
                types.ErrorData(
                    code=-32602,
                    message=(
                        "sessions/create MUST NOT include "
                        "_meta['io.modelcontextprotocol/session']"
                    ),
                )
            )

        record = store.create()
        return types.ServerResult(
            types.EmptyResult(session=store.metadata(record))
        )

    async def handle_delete(_request: DeleteSessionRequest) -> types.ServerResult:
        session_id = _session_id_from_meta(lowlevel.request_context.meta)
        deleted = store.delete(session_id)
        if not deleted:
            _raise_session_not_found(session_id)
        return types.ServerResult(types.EmptyResult())

    lowlevel.request_handlers[CreateSessionRequest] = handle_create
    lowlevel.request_handlers[DeleteSessionRequest] = handle_delete



def build_server() -> FastMCP:
    mcp = FastMCP("mcp-data-layer-sessions", log_level="WARNING")
    sessions = SessionStore()
    lowlevel = mcp._mcp_server
    _register_session_handlers(lowlevel, sessions)

    @mcp.tool(name="session_probe")
    async def session_probe(
        ctx: Context,
        note: str | None = None,
    ) -> types.CallToolResult:
        session_id = _session_id_from_meta(ctx.request_context.meta)
        record = sessions.get(session_id)
        if record is None:
            _raise_session_not_found(session_id)

        sessions.touch(record)
        metadata = sessions.metadata(record)
        detail = note.strip() if isinstance(note, str) and note.strip() else "none"
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"session active id={record.session_id}, "
                        f"calls={record.call_count}, note={detail}"
                    ),
                )
            ],
            _meta={SESSION_META_KEY: metadata},
        )

    return mcp


async def run_stdio_server(mcp: FastMCP) -> None:
    lowlevel = mcp._mcp_server
    init_options = lowlevel.create_initialization_options(experimental_capabilities={})
    capabilities_extra = init_options.capabilities.model_extra
    if isinstance(capabilities_extra, dict):
        capabilities_extra["sessions"] = {}

    async with stdio_server() as (read_stream, write_stream):
        async with AsyncExitStack() as stack:
            lifespan_context = await stack.enter_async_context(lowlevel.lifespan(lowlevel))
            session = await stack.enter_async_context(
                DataLayerServerSession(
                    read_stream,
                    write_stream,
                    init_options,
                )
            )

            async with anyio.create_task_group() as tg:
                async for message in session.incoming_messages:
                    tg.start_soon(
                        lowlevel._handle_message,
                        message,
                        session,
                        lifespan_context,
                        False,
                    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference MCP data-layer sessions server"
    )
    _ = parser.parse_args()
    anyio.run(run_stdio_server, build_server())


if __name__ == "__main__":
    main()
