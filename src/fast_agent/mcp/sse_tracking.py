"""SSE transport wrapper that emits channel events for UI display."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urljoin, urlparse

import anyio
import httpx
import mcp.types as types
from httpx_sse import aconnect_sse
from httpx_sse._exceptions import SSEError
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client
from mcp.shared.message import SessionMessage

from fast_agent.mcp.http_errors import format_http_error_detail
from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName, EventType
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from anyio.abc import TaskStatus
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


@dataclass(slots=True)
class _TrackingSSERuntime:
    url: str
    channel_hook: ChannelHook | None
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]
    session_id: str | None = None

    def get_session_id(self) -> str | None:
        return self.session_id

    async def sse_reader(
        self,
        event_source: Any,
        task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        try:
            async for sse in event_source.aiter_sse():
                if sse.event == "endpoint":
                    task_status.started(self._endpoint_url_from_event(sse.data))
                elif sse.event == "message":
                    await self._handle_message_event(sse.data)
                else:
                    _emit_channel_event(
                        self.channel_hook,
                        "get",
                        "keepalive",
                        raw_event=sse.event or "keepalive",
                    )
        except SSEError as sse_exc:
            logger.exception("Encountered SSE exception")
            _emit_channel_event(self.channel_hook, "get", "error", detail=str(sse_exc))
            raise
        except Exception as exc:
            logger.exception("Error in sse_reader")
            _emit_channel_event(self.channel_hook, "get", "error", detail=str(exc))
            await self.read_stream_writer.send(exc)
        finally:
            await self.read_stream_writer.aclose()

    def _endpoint_url_from_event(self, data: str) -> str:
        endpoint_url = urljoin(self.url, data)
        logger.debug("Received SSE endpoint URL: %s", endpoint_url)

        if not _same_origin(self.url, endpoint_url):
            error_msg = f"Endpoint origin does not match connection origin: {endpoint_url}"
            logger.error(error_msg)
            _emit_channel_event(self.channel_hook, "get", "error", detail=error_msg)
            raise ValueError(error_msg)

        self.session_id = _extract_session_id(endpoint_url)
        return endpoint_url

    async def _handle_message_event(self, data: str) -> None:
        try:
            message = types.JSONRPCMessage.model_validate_json(data)
        except Exception as exc:
            logger.exception("Error parsing server message")
            _emit_channel_event(
                self.channel_hook,
                "get",
                "error",
                detail="Error parsing server message",
            )
            await self.read_stream_writer.send(exc)
            return

        _emit_channel_event(self.channel_hook, "get", "message", message=message)
        await self.read_stream_writer.send(SessionMessage(message))

    async def post_writer(self, client: httpx.AsyncClient, endpoint_url: str) -> None:
        try:
            async with self.write_stream_reader:
                async for session_message in self.write_stream_reader:
                    payload = self._payload_for_session_message(session_message)
                    if payload is None:
                        continue

                    _emit_channel_event(
                        self.channel_hook,
                        "post-sse",
                        "message",
                        message=session_message.message,
                    )
                    await self._post_payload(client, endpoint_url, payload)
        except httpx.HTTPStatusError:
            logger.exception("HTTP error in post_writer")
        except Exception:
            logger.exception("Error in post_writer")
            _emit_channel_event(
                self.channel_hook,
                "post-sse",
                "error",
                detail="Error sending client message",
            )
        finally:
            await self.write_stream.aclose()

    @staticmethod
    def _payload_for_session_message(session_message: SessionMessage) -> dict[str, Any] | None:
        try:
            return session_message.message.model_dump(
                by_alias=True,
                mode="json",
                exclude_none=True,
            )
        except Exception:
            logger.exception("Invalid session message payload")
            return None

    async def _post_payload(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            response = await client.post(endpoint_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_detail = _format_http_error(exc)
            _emit_channel_event(
                self.channel_hook,
                "post-sse",
                "error",
                detail=error_detail.detail,
                status_code=error_detail.status_code,
            )
            raise


def _extract_session_id(endpoint_url: str) -> str | None:
    parsed = urlparse(endpoint_url)
    query_params = parse_qs(parsed.query)
    for key in ("sessionId", "session_id", "session"):
        values = query_params.get(key)
        if values:
            return values[0]
    return None


def _emit_channel_event(
    channel_hook: ChannelHook | None,
    channel: ChannelName,
    event_type: EventType,
    *,
    message: types.JSONRPCMessage | None = None,
    raw_event: str | None = None,
    detail: str | None = None,
    status_code: int | None = None,
) -> None:
    if channel_hook is None:
        return
    try:
        channel_hook(
            ChannelEvent(
                channel=channel,
                event_type=event_type,
                message=message,
                raw_event=raw_event,
                detail=detail,
                status_code=status_code,
            )
        )
    except Exception:
        logger.debug("Channel hook raised an exception", exc_info=True)


_format_http_error = format_http_error_detail


def _origin_port(scheme: str, port: int | None) -> int | None:
    if scheme == "http" and port == 80:
        return None
    if scheme == "https" and port == 443:
        return None
    return port


def _origin_parts(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlparse(url)
    scheme = strip_casefold(parsed.scheme)
    hostname = strip_casefold(parsed.hostname) if parsed.hostname else None
    return scheme, hostname, _origin_port(scheme, parsed.port)


def _same_origin(base_url: str, endpoint_url: str) -> bool:
    return _origin_parts(base_url) == _origin_parts(endpoint_url)


def _raise_for_sse_status(response: httpx.Response, channel_hook: ChannelHook | None) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error_detail = _format_http_error(exc)
        _emit_channel_event(
            channel_hook,
            "get",
            "error",
            detail=error_detail.detail,
            status_code=error_detail.status_code,
        )
        raise


@asynccontextmanager
async def tracking_sse_client(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5,
    sse_read_timeout: float = 60 * 5,
    httpx_client_factory: McpHttpClientFactory = create_mcp_http_client,
    auth: httpx.Auth | None = None,
    channel_hook: ChannelHook | None = None,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
        Callable[[], str | None],
    ],
    None,
]:
    """
    Client transport for SSE with channel activity tracking.
    """

    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        0
    )
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)
    runtime = _TrackingSSERuntime(
        url=url,
        channel_hook=channel_hook,
        read_stream_writer=read_stream_writer,
        write_stream=write_stream,
        write_stream_reader=write_stream_reader,
    )

    async with anyio.create_task_group() as tg:
        try:
            logger.debug("Connecting to SSE endpoint: %s", url)
            async with httpx_client_factory(
                headers=headers,
                auth=auth,
                timeout=httpx.Timeout(timeout, read=sse_read_timeout),
            ) as client:
                connected = False
                post_connected = False

                try:
                    async with aconnect_sse(
                        client,
                        "GET",
                        url,
                    ) as event_source:
                        _raise_for_sse_status(event_source.response, channel_hook)

                        _emit_channel_event(channel_hook, "get", "connect")
                        connected = True

                        endpoint_url = await tg.start(runtime.sse_reader, event_source)
                        _emit_channel_event(channel_hook, "post-sse", "connect")
                        post_connected = True
                        tg.start_soon(runtime.post_writer, client, endpoint_url)

                        try:
                            yield read_stream, write_stream, runtime.get_session_id
                        finally:
                            tg.cancel_scope.cancel()
                except Exception:
                    raise
                finally:
                    if connected:
                        _emit_channel_event(channel_hook, "get", "disconnect")
                    if post_connected:
                        _emit_channel_event(channel_hook, "post-sse", "disconnect")
        finally:
            await read_stream_writer.aclose()
            await read_stream.aclose()
            await write_stream_reader.aclose()
            await write_stream.aclose()
