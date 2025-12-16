from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Awaitable, Callable

import anyio
import httpx
from httpx_sse import EventSource, ServerSentEvent, aconnect_sse
from mcp.client.streamable_http import (
    DEFAULT_RECONNECTION_DELAY_MS,
    LAST_EVENT_ID,
    MAX_RECONNECTION_ATTEMPTS,
    RequestContext,
    RequestId,
    ResumptionError,
    StreamableHTTPTransport,
    StreamWriter,
)
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName

if TYPE_CHECKING:

    from anyio.abc import ObjectReceiveStream, ObjectSendStream

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


class ChannelTrackingStreamableHTTPTransport(StreamableHTTPTransport):
    """Streamable HTTP transport that emits channel events before dispatching."""

    def __init__(
        self,
        url: str,
        *,
        channel_hook: ChannelHook | None = None,
    ) -> None:
        super().__init__(url)
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
            self._emit_channel_event("post-json", "error", detail=str(exc))

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

        # Handle priming events (empty data with ID) for resumability
        if not sse.data:
            if sse.id and resumption_callback:
                await resumption_callback(sse.id)
            self._emit_channel_event(channel, "keepalive", raw_event="priming")
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
            self._emit_channel_event(channel, "error", detail=str(exc))
            return False

    async def handle_get_stream(  # type: ignore[override]
        self,
        client: httpx.AsyncClient,
        read_stream_writer: StreamWriter,
    ) -> None:
        last_event_id: str | None = None
        retry_interval_ms: int | None = None
        attempt: int = 0

        while attempt < MAX_RECONNECTION_ATTEMPTS:  # pragma: no branch
            connected = False
            try:
                if not self.session_id:
                    return

                headers = self._prepare_headers()
                if last_event_id:
                    headers[LAST_EVENT_ID] = last_event_id  # pragma: no cover

                async with aconnect_sse(
                    client,
                    "GET",
                    self.url,
                    headers=headers,
                ) as event_source:
                    event_source.response.raise_for_status()
                    self._emit_channel_event("get", "connect")
                    connected = True

                    async for sse in event_source.aiter_sse():
                        if sse.id:
                            last_event_id = sse.id  # pragma: no cover
                        if sse.retry is not None:
                            retry_interval_ms = sse.retry  # pragma: no cover

                        await self._handle_sse_event_with_channel(
                            "get",
                            sse,
                            read_stream_writer,
                        )

                    attempt = 0

            except Exception as exc:  # pragma: no cover - non fatal stream errors
                logger.debug("GET stream error: %s", exc)
                attempt += 1
                status_code = None
                detail = str(exc)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    status_code = exc.response.status_code
                    reason = exc.response.reason_phrase or ""
                    if not reason:
                        try:
                            reason = (exc.response.text or "").strip()
                        except Exception:
                            reason = ""
                    detail = f"HTTP {status_code}: {reason or 'response'}"
                self._emit_channel_event("get", "error", detail=detail, status_code=status_code)
            finally:
                if connected:
                    self._emit_channel_event("get", "disconnect")

            if attempt >= MAX_RECONNECTION_ATTEMPTS:  # pragma: no cover
                return

            delay_ms = (
                retry_interval_ms if retry_interval_ms is not None else DEFAULT_RECONNECTION_DELAY_MS
            )
            logger.info("GET stream disconnected, reconnecting in %sms...", delay_ms)
            await anyio.sleep(delay_ms / 1000.0)

    async def _handle_resumption_request(  # type: ignore[override]
        self,
        ctx: RequestContext,
    ) -> None:
        headers = self._prepare_headers()
        if ctx.metadata and ctx.metadata.resumption_token:
            headers[LAST_EVENT_ID] = ctx.metadata.resumption_token
        else:  # pragma: no cover - defensive
            raise ResumptionError("Resumption request requires a resumption token")

        original_request_id: RequestId | None = None
        if isinstance(ctx.session_message.message.root, JSONRPCRequest):
            original_request_id = ctx.session_message.message.root.id

        async with aconnect_sse(
            ctx.client,
            "GET",
            self.url,
            headers=headers,
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
        last_event_id: str | None = None
        retry_interval_ms: int | None = None

        try:
            event_source = EventSource(response)
            async for sse in event_source.aiter_sse():
                if sse.id:
                    last_event_id = sse.id
                if sse.retry is not None:
                    retry_interval_ms = sse.retry

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
                    return
        except Exception as exc:  # pragma: no cover - propagate to session
            logger.exception("Error reading SSE stream")
            await ctx.read_stream_writer.send(exc)
            self._emit_channel_event("post-sse", "error", detail=str(exc))

        if last_event_id is not None:  # pragma: no branch
            await self._handle_reconnection(ctx, "post-sse", last_event_id, retry_interval_ms)

    async def _handle_reconnection(
        self,
        ctx: RequestContext,
        channel: ChannelName,
        last_event_id: str,
        retry_interval_ms: int | None = None,
        attempt: int = 0,
    ) -> None:
        if attempt >= MAX_RECONNECTION_ATTEMPTS:  # pragma: no cover
            logger.debug(
                "Max reconnection attempts (%s) exceeded", MAX_RECONNECTION_ATTEMPTS
            )  # pragma: no cover
            return

        delay_ms = retry_interval_ms if retry_interval_ms is not None else DEFAULT_RECONNECTION_DELAY_MS
        await anyio.sleep(delay_ms / 1000.0)

        headers = self._prepare_headers()
        headers[LAST_EVENT_ID] = last_event_id

        original_request_id = None
        if isinstance(ctx.session_message.message.root, JSONRPCRequest):  # pragma: no branch
            original_request_id = ctx.session_message.message.root.id

        try:
            async with aconnect_sse(
                ctx.client,
                "GET",
                self.url,
                headers=headers,
            ) as event_source:
                event_source.response.raise_for_status()
                logger.info("Reconnected to SSE stream")

                reconnect_last_event_id: str = last_event_id
                reconnect_retry_ms = retry_interval_ms

                async for sse in event_source.aiter_sse():
                    if sse.id:  # pragma: no branch
                        reconnect_last_event_id = sse.id
                    if sse.retry is not None:
                        reconnect_retry_ms = sse.retry

                    is_complete = await self._handle_sse_event_with_channel(
                        channel,
                        sse,
                        ctx.read_stream_writer,
                        original_request_id,
                        ctx.metadata.on_resumption_token_update if ctx.metadata else None,
                    )
                    if is_complete:
                        await event_source.response.aclose()
                        return

                await self._handle_reconnection(
                    ctx,
                    channel,
                    reconnect_last_event_id,
                    reconnect_retry_ms,
                    0,
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("Reconnection failed: %s", exc)
            self._emit_channel_event(channel, "error", detail=str(exc))
            await self._handle_reconnection(
                ctx,
                channel,
                last_event_id,
                retry_interval_ms,
                attempt + 1,
            )


@asynccontextmanager
async def tracking_streamablehttp_client(
    url: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    terminate_on_close: bool = True,
    channel_hook: ChannelHook | None = None,
) -> AsyncGenerator[
    tuple[
        ObjectReceiveStream[SessionMessage | Exception],
        ObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ],
    None,
]:
    """Context manager mirroring streamable_http_client with channel tracking."""

    transport = ChannelTrackingStreamableHTTPTransport(url, channel_hook=channel_hook)

    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        0
    )
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    client_provided = http_client is not None
    client = http_client or create_mcp_http_client()

    async with anyio.create_task_group() as tg:
        try:
            async with AsyncExitStack() as stack:
                if not client_provided:
                    await stack.enter_async_context(client)

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
