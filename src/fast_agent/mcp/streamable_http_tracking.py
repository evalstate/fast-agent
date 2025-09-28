from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Awaitable, Callable

import anyio
import httpx
from httpx_sse import EventSource, ServerSentEvent, aconnect_sse
from mcp.client.streamable_http import (
    RequestContext,
    RequestId,
    StreamableHTTPTransport,
    StreamWriter,
)
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName

if TYPE_CHECKING:
    from datetime import timedelta

    from anyio.abc import ObjectReceiveStream, ObjectSendStream

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


class ChannelTrackingStreamableHTTPTransport(StreamableHTTPTransport):
    """Streamable HTTP transport that emits channel events before dispatching."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | timedelta = 30,
        sse_read_timeout: float | timedelta = 60 * 5,
        auth: httpx.Auth | None = None,
        channel_hook: ChannelHook | None = None,
    ) -> None:
        super().__init__(
            url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            auth=auth,
        )
        self._channel_hook = channel_hook

    def _emit_channel_event(
        self,
        channel: ChannelName,
        event_type: str,
        *,
        message: JSONRPCMessage | None = None,
        raw_event: str | None = None,
        detail: str | None = None,
        status_code: int | None = None,
    ) -> None:
        if self._channel_hook is None:
            return
        try:
            self._channel_hook(
                ChannelEvent(
                    channel=channel,
                    event_type=event_type,  # type: ignore[arg-type]
                    message=message,
                    raw_event=raw_event,
                    detail=detail,
                    status_code=status_code,
                )
            )
        except Exception:  # pragma: no cover - hook errors must not break transport
            logger.exception("Channel hook raised an exception")

    async def _handle_json_response(  # type: ignore[override]
        self,
        response: httpx.Response,
        read_stream_writer: StreamWriter,
        is_initialization: bool = False,
    ) -> None:
        try:
            content = await response.aread()
            message = JSONRPCMessage.model_validate_json(content)

            if is_initialization:
                self._maybe_extract_protocol_version_from_message(message)

            self._emit_channel_event("post-json", "message", message=message)
            await read_stream_writer.send(SessionMessage(message))
        except Exception as exc:  # pragma: no cover - propagate to session
            logger.exception("Error parsing JSON response")
            await read_stream_writer.send(exc)

    async def _handle_sse_event_with_channel(
        self,
        channel: ChannelName,
        sse: ServerSentEvent,
        read_stream_writer: StreamWriter,
        original_request_id: RequestId | None = None,
        resumption_callback: Callable[[str], Awaitable[None]] | None = None,
        is_initialization: bool = False,
    ) -> bool:
        if sse.event != "message":
            # Treat non-message events (e.g. ping) as keepalive notifications
            self._emit_channel_event(channel, "keepalive", raw_event=sse.event or "keepalive")
            return False

        try:
            message = JSONRPCMessage.model_validate_json(sse.data)
            if is_initialization:
                self._maybe_extract_protocol_version_from_message(message)

            if original_request_id is not None and isinstance(
                message.root, (JSONRPCResponse, JSONRPCError)
            ):
                message.root.id = original_request_id

            self._emit_channel_event(channel, "message", message=message)
            await read_stream_writer.send(SessionMessage(message))

            if sse.id and resumption_callback:
                await resumption_callback(sse.id)

            return isinstance(message.root, (JSONRPCResponse, JSONRPCError))
        except Exception as exc:  # pragma: no cover - propagate to session
            logger.exception("Error parsing SSE message")
            await read_stream_writer.send(exc)
            return False

    async def handle_get_stream(  # type: ignore[override]
        self,
        client: httpx.AsyncClient,
        read_stream_writer: StreamWriter,
    ) -> None:
        if not self.session_id:
            return

        headers = self._prepare_request_headers(self.request_headers)
        connected = False
        try:
            async with aconnect_sse(
                client,
                "GET",
                self.url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, read=self.sse_read_timeout),
            ) as event_source:
                event_source.response.raise_for_status()
                self._emit_channel_event("get", "connect")
                connected = True
                async for sse in event_source.aiter_sse():
                    await self._handle_sse_event_with_channel(
                        "get",
                        sse,
                        read_stream_writer,
                    )
        except Exception as exc:  # pragma: no cover - non fatal stream errors
            logger.debug("GET stream error (non-fatal): %s", exc)
            status_code = None
            detail = str(exc)
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response is not None:
                    status_code = exc.response.status_code
                    reason = exc.response.reason_phrase or ""
                    if not reason:
                        try:
                            reason = (exc.response.text or "").strip()
                        except Exception:
                            reason = ""
                    detail = f"HTTP {status_code}: {reason or 'response'}"
                else:
                    status_code = exc.response.status_code if hasattr(exc, "response") else None
            self._emit_channel_event("get", "error", detail=detail, status_code=status_code)
        finally:
            if connected:
                self._emit_channel_event("get", "disconnect")

    async def _handle_resumption_request(  # type: ignore[override]
        self,
        ctx: RequestContext,
    ) -> None:
        headers = self._prepare_request_headers(ctx.headers)
        if ctx.metadata and ctx.metadata.resumption_token:
            headers["last-event-id"] = ctx.metadata.resumption_token
        else:  # pragma: no cover - defensive
            raise ValueError("Resumption request requires a resumption token")

        original_request_id: RequestId | None = None
        if isinstance(ctx.session_message.message.root, JSONRPCRequest):
            original_request_id = ctx.session_message.message.root.id

        async with aconnect_sse(
            ctx.client,
            "GET",
            self.url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, read=self.sse_read_timeout),
        ) as event_source:
            event_source.response.raise_for_status()
            async for sse in event_source.aiter_sse():
                is_complete = await self._handle_sse_event_with_channel(
                    "resumption",
                    sse,
                    ctx.read_stream_writer,
                    original_request_id,
                    ctx.metadata.on_resumption_token_update if ctx.metadata else None,
                )
                if is_complete:
                    await event_source.response.aclose()
                    break

    async def _handle_sse_response(  # type: ignore[override]
        self,
        response: httpx.Response,
        ctx: RequestContext,
        is_initialization: bool = False,
    ) -> None:
        try:
            event_source = EventSource(response)
            async for sse in event_source.aiter_sse():
                is_complete = await self._handle_sse_event_with_channel(
                    "post-sse",
                    sse,
                    ctx.read_stream_writer,
                    resumption_callback=(
                        ctx.metadata.on_resumption_token_update if ctx.metadata else None
                    ),
                    is_initialization=is_initialization,
                )
                if is_complete:
                    await response.aclose()
                    break
        except Exception as exc:  # pragma: no cover - propagate to session
            logger.exception("Error reading SSE stream")
            await ctx.read_stream_writer.send(exc)


@asynccontextmanager
async def tracking_streamablehttp_client(
    url: str,
    headers: dict[str, str] | None = None,
    *,
    timeout: float | timedelta = 30,
    sse_read_timeout: float | timedelta = 60 * 5,
    terminate_on_close: bool = True,
    httpx_client_factory: McpHttpClientFactory = create_mcp_http_client,
    auth: httpx.Auth | None = None,
    channel_hook: ChannelHook | None = None,
) -> AsyncGenerator[
    tuple[
        ObjectReceiveStream[SessionMessage | Exception],
        ObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ],
    None,
]:
    """Context manager mirroring streamablehttp_client with channel tracking."""

    transport = ChannelTrackingStreamableHTTPTransport(
        url,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=sse_read_timeout,
        auth=auth,
        channel_hook=channel_hook,
    )

    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        0
    )
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    async with anyio.create_task_group() as tg:
        try:
            async with httpx_client_factory(
                headers=transport.request_headers,
                timeout=httpx.Timeout(transport.timeout, read=transport.sse_read_timeout),
                auth=transport.auth,
            ) as client:

                def start_get_stream() -> None:
                    tg.start_soon(transport.handle_get_stream, client, read_stream_writer)

                tg.start_soon(
                    transport.post_writer,
                    client,
                    write_stream_reader,
                    read_stream_writer,
                    write_stream,
                    start_get_stream,
                    tg,
                )

                try:
                    yield read_stream, write_stream, transport.get_session_id
                finally:
                    if transport.session_id and terminate_on_close:
                        await transport.terminate_session(client)
                    tg.cancel_scope.cancel()
        finally:
            await read_stream_writer.aclose()
            await read_stream.aclose()
            await write_stream_reader.aclose()
            await write_stream.aclose()
