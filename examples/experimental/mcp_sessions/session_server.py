"""Experimental MCP Sessions v2 demo server.

This server demonstrates the fast-agent experimental session flow:

1. Advertises ``experimental.session`` capability (v2)
2. Implements ``session/create`` / ``session/list`` / ``session/delete``
3. Echoes a session cookie in ``_meta["mcp/session"]`` on tool responses
4. Supports explicit revocation with ``_meta["mcp/session"] = null``

Why this uses a small low-level bridge:
--------------------------------------
Current MCP SDK typed client/server request unions do not include ``session/*``
methods yet. For this example we extend the server-side request union locally and
run the FastMCP low-level server loop with a custom session parser.
"""

from __future__ import annotations

import argparse
import logging
import secrets
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import anyio
import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.session import InitializationState, ServerSession
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import (
    INVALID_REQUEST,
    MCP_SESSION_ID_HEADER,
    ErrorData,
    JSONRPCError,
    StreamableHTTPSessionManager,
)
from mcp.shared.session import BaseSession
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anyio.abc import TaskStatus
    from mcp.server.models import InitializationOptions
    from mcp.shared.message import SessionMessage
    from starlette.types import Receive, Scope, Send

SESSION_META_KEY = "mcp/session"
SESSION_CAPABILITIES: dict[str, dict[str, Any]] = {
    "session": {
        "version": 2,
        "features": ["create", "list", "delete"],
    }
}


class SessionCreateHints(BaseModel):
    label: str | None = None
    data: dict[str, str] | None = None


class SessionCreateParams(types.RequestParams):
    hints: SessionCreateHints | None = None


class SessionCreateRequest(types.Request):
    method: Literal["session/create"] = "session/create"
    params: SessionCreateParams | None = None


class SessionListRequest(types.Request):
    method: Literal["session/list"] = "session/list"
    params: types.RequestParams | None = None


class SessionCookie(BaseModel):
    id: str
    expiry: str | None = None
    data: dict[str, str] | None = None


class SessionDeleteParams(types.RequestParams):
    id: str | None = None


class SessionDeleteRequest(types.Request):
    method: Literal["session/delete"] = "session/delete"
    params: SessionDeleteParams | None = None


class ExperimentalClientRequest(types.ClientRequest):
    root: (
        types.ClientRequestType
        | SessionCreateRequest
        | SessionListRequest
        | SessionDeleteRequest
    )


class ExperimentalServerSession(ServerSession):
    """ServerSession variant that accepts ``ExperimentalClientRequest``."""

    def __init__(
        self,
        read_stream: anyio.streams.memory.MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: anyio.streams.memory.MemoryObjectSendStream[SessionMessage],
        init_options: InitializationOptions,
        stateless: bool = False,
    ) -> None:
        BaseSession.__init__(
            self,
            read_stream,
            write_stream,
            ExperimentalClientRequest,
            types.ClientNotification,
        )
        self._initialization_state = (
            InitializationState.Initialized
            if stateless
            else InitializationState.NotInitialized
        )
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
    expiry: str
    data: dict[str, str]
    tool_calls: int = 0


@dataclass(slots=True)
class SessionStore:
    """Tiny in-memory store for demo session cookies."""

    _sessions: dict[str, SessionRecord] = field(default_factory=dict)

    def create(self, title: str, *, reason: str) -> SessionRecord:
        session_id = f"sess-{secrets.token_hex(6)}"
        expiry = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
        data = {
            "title": title,
            "reason": reason,
            "createdAt": datetime.now(UTC).isoformat(),
        }
        record = SessionRecord(session_id=session_id, expiry=expiry, data=data)
        self._sessions[session_id] = record
        return record

    def get(self, session_id: str | None) -> SessionRecord | None:
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def ensure_from_cookie(self, cookie: dict[str, Any] | None) -> SessionRecord | None:
        if not cookie:
            return None
        session_id = cookie.get("id")
        if not isinstance(session_id, str) or not session_id:
            return None
        return self._sessions.get(session_id)

    def list_cookies(self) -> list[SessionCookie]:
        return [
            SessionCookie(
                id=record.session_id,
                expiry=record.expiry,
                data=dict(record.data),
            )
            for record in self._sessions.values()
        ]

    def delete(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        return self._sessions.pop(session_id, None) is not None

    def to_cookie(self, record: SessionRecord) -> dict[str, Any]:
        return {
            "id": record.session_id,
            "expiry": record.expiry,
            "data": dict(record.data),
        }


def _meta_dict(meta: types.RequestParams.Meta | None) -> dict[str, Any]:
    if meta is None:
        return {}
    dumped = meta.model_dump(by_alias=True, exclude_none=False)
    if isinstance(dumped, dict):
        return dumped
    return {}


def _session_cookie_from_meta(meta: types.RequestParams.Meta | None) -> dict[str, Any] | None:
    payload = _meta_dict(meta)
    raw_cookie = payload.get(SESSION_META_KEY)
    if isinstance(raw_cookie, dict):
        return dict(raw_cookie)
    return None


def _session_id_from_cookie(cookie: dict[str, Any] | None) -> str | None:
    if not cookie:
        return None
    raw_id = cookie.get("id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    return None


def _title_from_create_request(request: SessionCreateRequest) -> str:
    hints = request.params.hints if request.params else None
    if hints is not None and hints.data is not None:
        title = hints.data.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    if hints is not None and hints.label is not None and hints.label.strip():
        return hints.label.strip()
    return "experimental-session-demo"


def _cookie_meta(cookie: dict[str, Any] | None) -> dict[str, Any]:
    return {SESSION_META_KEY: cookie}


def _cookie_text(cookie: dict[str, Any]) -> str:
    session_id = _session_id_from_cookie(cookie) or "unknown"
    data = cookie.get("data")
    title = None
    if isinstance(data, dict):
        raw_title = data.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            title = raw_title.strip()
    if title:
        return f"id={session_id}, title={title}"
    return f"id={session_id}"


def build_demo_server() -> FastMCP:
    mcp = FastMCP("experimental-mcp-sessions", log_level="WARNING")
    store = SessionStore()
    lowlevel = mcp._mcp_server

    async def handle_session_create(request: SessionCreateRequest) -> types.ServerResult:
        title = _title_from_create_request(request)
        record = store.create(title=title, reason="session/create")
        cookie = store.to_cookie(record)
        return types.ServerResult(
            types.EmptyResult(
                _meta=_cookie_meta(cookie),
                id=record.session_id,
                expiry=record.expiry,
                data=dict(record.data),
            )
        )

    async def handle_session_list(_request: SessionListRequest) -> types.ServerResult:
        cookies = [cookie.model_dump(exclude_none=True) for cookie in store.list_cookies()]
        return types.ServerResult(types.EmptyResult(sessions=cookies))

    async def handle_session_delete(request: SessionDeleteRequest) -> types.ServerResult:
        cookie = _session_cookie_from_meta(lowlevel.request_context.meta)
        cookie_id = _session_id_from_cookie(cookie)
        requested_id = request.params.id if request.params else None
        target_id = requested_id or cookie_id
        deleted = store.delete(target_id)
        result_meta: dict[str, Any] | None = None
        if deleted and target_id == cookie_id:
            result_meta = _cookie_meta(None)
        elif cookie is not None:
            result_meta = _cookie_meta(cookie)

        kwargs: dict[str, Any] = {}
        if result_meta is not None:
            kwargs["_meta"] = result_meta
        return types.ServerResult(types.EmptyResult(deleted=deleted, **kwargs))

    lowlevel.request_handlers[SessionCreateRequest] = handle_session_create
    lowlevel.request_handlers[SessionListRequest] = handle_session_list
    lowlevel.request_handlers[SessionDeleteRequest] = handle_session_delete

    @mcp.tool(name="session_probe")
    async def session_probe(
        ctx: Context,
        action: Literal["status", "revoke", "new"] = "status",
        note: str | None = None,
    ) -> types.CallToolResult:
        cookie = _session_cookie_from_meta(ctx.request_context.meta)
        record = store.ensure_from_cookie(cookie)

        if record is None:
            record = store.create(
                title="lazy-tool-session",
                reason="lazy/tools/call",
            )

        if action == "revoke":
            store.delete(record.session_id)
            detail = note or "revocation requested"
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=(
                            "session revoked by tool; returning _meta['mcp/session']=null "
                            f"({detail})"
                        ),
                    )
                ],
                _meta=_cookie_meta(None),
            )

        if action == "new":
            store.delete(record.session_id)
            record = store.create(title="rotated-session", reason="tool/new")

        record.tool_calls += 1
        cookie = store.to_cookie(record)
        detail = note or "none"
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        "session active "
                        f"({_cookie_text(cookie)}), calls={record.tool_calls}, note={detail}"
                    ),
                )
            ],
            _meta=_cookie_meta(cookie),
        )

    return mcp


async def _run_lowlevel_with_experimental_session(
    lowlevel: Any,
    read_stream: anyio.streams.memory.MemoryObjectReceiveStream[SessionMessage | Exception],
    write_stream: anyio.streams.memory.MemoryObjectSendStream[SessionMessage],
    *,
    stateless: bool,
) -> None:
    init_options = lowlevel.create_initialization_options(
        experimental_capabilities=SESSION_CAPABILITIES,
    )

    async with AsyncExitStack() as stack:
        lifespan_context = await stack.enter_async_context(lowlevel.lifespan(lowlevel))
        session = await stack.enter_async_context(
            ExperimentalServerSession(
                read_stream,
                write_stream,
                init_options,
                stateless=stateless,
            )
        )

        task_support = (
            lowlevel._experimental_handlers.task_support
            if lowlevel._experimental_handlers is not None
            else None
        )
        if task_support is not None:
            task_support.configure_session(session)
            await stack.enter_async_context(task_support.run())

        async with anyio.create_task_group() as tg:
            async for message in session.incoming_messages:
                tg.start_soon(
                    lowlevel._handle_message,
                    message,
                    session,
                    lifespan_context,
                    False,
                )


async def run_stdio_server(mcp: FastMCP) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await _run_lowlevel_with_experimental_session(
            mcp._mcp_server,
            read_stream,
            write_stream,
            stateless=False,
        )


class ExperimentalStreamableHTTPSessionManager(StreamableHTTPSessionManager):
    """StreamableHTTP manager variant that uses ExperimentalServerSession."""

    async def _handle_stateless_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        http_transport = StreamableHTTPServerTransport(
            mcp_session_id=None,
            is_json_response_enabled=self.json_response,
            event_store=None,
            security_settings=self.security_settings,
        )

        async def run_stateless_server(
            *,
            task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
        ) -> None:
            async with http_transport.connect() as streams:
                read_stream, write_stream = streams
                task_status.started()
                try:
                    await _run_lowlevel_with_experimental_session(
                        self.app,
                        read_stream,
                        write_stream,
                        stateless=True,
                    )
                except Exception:
                    LOGGER.exception("Stateless session crashed")

        assert self._task_group is not None
        await self._task_group.start(run_stateless_server)
        await http_transport.handle_request(scope, receive, send)
        await http_transport.terminate()

    async def _handle_stateful_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        request = Request(scope, receive)
        request_mcp_session_id = request.headers.get(MCP_SESSION_ID_HEADER)

        if request_mcp_session_id is not None and request_mcp_session_id in self._server_instances:
            transport = self._server_instances[request_mcp_session_id]
            await transport.handle_request(scope, receive, send)
            return

        if request_mcp_session_id is None:
            async with self._session_creation_lock:
                new_session_id = uuid4().hex
                http_transport = StreamableHTTPServerTransport(
                    mcp_session_id=new_session_id,
                    is_json_response_enabled=self.json_response,
                    event_store=self.event_store,
                    security_settings=self.security_settings,
                    retry_interval=self.retry_interval,
                )

                assert http_transport.mcp_session_id is not None
                self._server_instances[http_transport.mcp_session_id] = http_transport

                async def run_server(
                    *,
                    task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
                ) -> None:
                    async with http_transport.connect() as streams:
                        read_stream, write_stream = streams
                        task_status.started()
                        try:
                            await _run_lowlevel_with_experimental_session(
                                self.app,
                                read_stream,
                                write_stream,
                                stateless=False,
                            )
                        except Exception as exc:
                            LOGGER.exception(
                                "Session %s crashed: %s",
                                http_transport.mcp_session_id,
                                exc,
                            )
                        finally:
                            if (
                                http_transport.mcp_session_id
                                and http_transport.mcp_session_id in self._server_instances
                                and not http_transport.is_terminated
                            ):
                                del self._server_instances[http_transport.mcp_session_id]

                assert self._task_group is not None
                await self._task_group.start(run_server)
                await http_transport.handle_request(scope, receive, send)
        else:
            error_response = JSONRPCError(
                jsonrpc="2.0",
                id="server-error",
                error=ErrorData(
                    code=INVALID_REQUEST,
                    message="Session not found",
                ),
            )
            response = Response(
                content=error_response.model_dump_json(by_alias=True, exclude_none=True),
                status_code=404,
                media_type="application/json",
            )
            await response(scope, receive, send)


async def run_streamable_http_server(
    mcp: FastMCP,
    *,
    host: str,
    port: int,
    path: str,
) -> None:
    import uvicorn

    manager = ExperimentalStreamableHTTPSessionManager(
        app=mcp._mcp_server,
        event_store=None,
        json_response=False,
        stateless=False,
        security_settings=None,
        retry_interval=None,
    )
    app = Starlette(
        debug=False,
        routes=[Route(path, endpoint=StreamableHTTPASGIApp(manager))],
        lifespan=lambda _app: manager.run(),
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental MCP Sessions demo server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="Server transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--path", default="/mcp", help="HTTP MCP path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mcp = build_demo_server()
    if args.transport == "stdio":
        anyio.run(run_stdio_server, mcp)
        return

    async def _run_http() -> None:
        await run_streamable_http_server(
            mcp,
            host=args.host,
            port=args.port,
            path=args.path,
        )

    anyio.run(_run_http)


if __name__ == "__main__":
    main()
