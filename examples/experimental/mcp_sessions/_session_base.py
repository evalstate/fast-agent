"""Shared draft data-layer session helpers for experimental MCP demos.

This module is intentionally small and stdio-focused.
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
from mcp.server.session import InitializationState, ServerSession
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.shared.session import BaseSession
from pydantic import BaseModel

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.models import InitializationOptions
    from mcp.shared.message import SessionMessage

SESSION_META_KEY = "io.modelcontextprotocol/session"
SESSION_NOT_FOUND_ERROR_CODE = -32043


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
    state: str | None
    expires_at: str
    label: str
    tool_calls: int = 0


@dataclass(slots=True)
class SessionStore:
    """Small in-memory store used by experimental demo servers."""

    _sessions: dict[str, SessionRecord] = field(default_factory=dict)
    include_state: bool = True

    def create(self, title: str = "experimental-session", *, reason: str = "") -> SessionRecord:
        del reason
        session_id = f"sess-{secrets.token_hex(6)}"
        initial_state = _encode_state(call_count=0) if self.include_state else None
        record = SessionRecord(
            session_id=session_id,
            state=initial_state,
            expires_at=_future_expiry(minutes=30),
            label=title,
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

    def to_metadata(self, record: SessionRecord) -> dict[str, str]:
        metadata: dict[str, str] = {
            "sessionId": record.session_id,
            "expiresAt": record.expires_at,
        }
        if self.include_state and isinstance(record.state, str) and record.state:
            metadata["state"] = record.state
        return metadata

    def touch(self, record: SessionRecord) -> None:
        record.tool_calls += 1
        if self.include_state:
            record.state = _encode_state(call_count=record.tool_calls)
        record.expires_at = _future_expiry(minutes=30)



def _future_expiry(*, minutes: int) -> str:
    expiry = datetime.now(UTC) + timedelta(minutes=minutes)
    return expiry.replace(microsecond=0).isoformat().replace("+00:00", "Z")



def _encode_state(*, call_count: int) -> str:
    payload = json.dumps({"callCount": call_count}, separators=(",", ":"), sort_keys=True)
    # Use padded base64 to match the draft examples and keep tokens copy-pastable.
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")



def meta_dict(meta: types.RequestParams.Meta | None) -> dict[str, Any]:
    if meta is None:
        return {}
    dumped = meta.model_dump(by_alias=True, exclude_none=False)
    return dumped if isinstance(dumped, dict) else {}



def session_metadata_from_meta(
    meta: types.RequestParams.Meta | None,
) -> dict[str, Any] | None:
    payload = meta_dict(meta)
    raw = payload.get(SESSION_META_KEY)
    if isinstance(raw, dict):
        return dict(raw)
    return None



def session_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("sessionId")
    if isinstance(raw, str) and raw:
        return raw
    return None



def session_id_from_meta(meta: types.RequestParams.Meta | None) -> str | None:
    return session_id_from_metadata(session_metadata_from_meta(meta))



def session_meta(record: SessionRecord, store: SessionStore) -> dict[str, Any]:
    return {SESSION_META_KEY: store.to_metadata(record)}



def raise_session_not_found(session_id: str | None) -> None:
    raise McpError(
        types.ErrorData(
            code=SESSION_NOT_FOUND_ERROR_CODE,
            message="Session not found",
            data={"sessionId": session_id or ""},
        )
    )



def require_session(ctx: Context, store: SessionStore) -> SessionRecord:
    session_id = session_id_from_meta(ctx.request_context.meta)
    record = store.get(session_id)
    if record is None:
        raise_session_not_found(session_id)
    return record



def optional_session(ctx: Context, store: SessionStore) -> SessionRecord | None:
    metadata = session_metadata_from_meta(ctx.request_context.meta)
    if metadata is None:
        return None
    session_id = session_id_from_metadata(metadata)
    record = store.get(session_id)
    if record is None:
        raise_session_not_found(session_id)
    return record



def register_session_handlers(
    lowlevel: Any,
    store: SessionStore,
) -> None:
    """Register draft `sessions/create` and `sessions/delete` handlers."""

    async def handle_session_create(_request: CreateSessionRequest) -> types.ServerResult:
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

        record = store.create(reason="sessions/create")
        return types.ServerResult(types.EmptyResult(session=store.to_metadata(record)))

    async def handle_session_delete(_request: DeleteSessionRequest) -> types.ServerResult:
        session_id = session_id_from_meta(lowlevel.request_context.meta)
        deleted = store.delete(session_id)
        if not deleted:
            raise_session_not_found(session_id)
        return types.ServerResult(types.EmptyResult())

    lowlevel.request_handlers[CreateSessionRequest] = handle_session_create
    lowlevel.request_handlers[DeleteSessionRequest] = handle_session_delete


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



def add_transport_args(parser: argparse.ArgumentParser) -> None:
    """Kept for compatibility with existing demo entrypoints."""
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="Server transport mode (default: stdio)",
    )



def run_server(mcp: FastMCP, args: argparse.Namespace | None = None) -> None:
    """Run a FastMCP server with draft data-layer session support."""
    if args is None:
        parser = argparse.ArgumentParser()
        add_transport_args(parser)
        args = parser.parse_args()

    if args.transport != "stdio":
        raise ValueError("Only stdio transport is supported by this draft demo.")

    anyio.run(run_stdio_server, mcp)
