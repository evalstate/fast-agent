"""SSE transport wrapper that emits channel events for UI display."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable
from urllib.parse import parse_qs, urljoin, urlparse

import anyio
import httpx
import mcp.types as types
from httpx_sse import aconnect_sse
from httpx_sse._exceptions import SSEError
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client
from mcp.shared.message import SessionMessage

from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName

if TYPE_CHECKING:
    from anyio.abc import TaskStatus
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


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
    event_type: str,
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
                event_type=event_type,  # type: ignore[arg-type]
                message=message,
                raw_event=raw_event,
                detail=detail,
                status_code=status_code,
            )
        )
    except Exception:
        logger.debug("Channel hook raised an exception", exc_info=True)


def _format_http_error(exc: httpx.HTTPStatusError) -> tuple[int | None, str]:
    status_code: int | None = None
    detail = str(exc)
    if exc.response is not None:
        status_code = exc.response.status_code
        reason = exc.response.reason_phrase or ""
        if not reason:
            try:
                reason = (exc.response.text or "").strip()
            except Exception:
                reason = ""
        detail = f"HTTP {status_code}: {reason or 'response'}"
    return status_code, detail


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

    session_id: str | None = None

    def get_session_id() -> str | None:
        return session_id

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

                async def sse_reader(
                    task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED,
                ):
                    try:
                        async for sse in event_source.aiter_sse():
                            if sse.event == "endpoint":
                                endpoint_url = urljoin(url, sse.data)
                                logger.debug("Received SSE endpoint URL: %s", endpoint_url)

                                url_parsed = urlparse(url)
                                endpoint_parsed = urlparse(endpoint_url)
                                if (
                                    url_parsed.scheme != endpoint_parsed.scheme
                                    or url_parsed.netloc != endpoint_parsed.netloc
                                ):
                                    error_msg = (
                                        "Endpoint origin does not match connection origin: "
                                        f"{endpoint_url}"
                                    )
                                    logger.error(error_msg)
                                    _emit_channel_event(
                                        channel_hook,
                                        "get",
                                        "error",
                                        detail=error_msg,
                                    )
                                    raise ValueError(error_msg)

                                nonlocal session_id
                                session_id = _extract_session_id(endpoint_url)
                                task_status.started(endpoint_url)
                            elif sse.event == "message":
                                try:
                                    message = types.JSONRPCMessage.model_validate_json(sse.data)
                                except Exception as exc:
                                    logger.exception("Error parsing server message")
                                    _emit_channel_event(
                                        channel_hook,
                                        "get",
                                        "error",
                                        detail="Error parsing server message",
                                    )
                                    await read_stream_writer.send(exc)
                                    continue

                                _emit_channel_event(channel_hook, "get", "message", message=message)
                                await read_stream_writer.send(SessionMessage(message))
                            else:
                                _emit_channel_event(
                                    channel_hook,
                                    "get",
                                    "keepalive",
                                    raw_event=sse.event or "keepalive",
                                )
                    except SSEError as sse_exc:
                        logger.exception("Encountered SSE exception")
                        _emit_channel_event(
                            channel_hook,
                            "get",
                            "error",
                            detail=str(sse_exc),
                        )
                        raise
                    except Exception as exc:
                        logger.exception("Error in sse_reader")
                        _emit_channel_event(
                            channel_hook,
                            "get",
                            "error",
                            detail=str(exc),
                        )
                        await read_stream_writer.send(exc)
                    finally:
                        await read_stream_writer.aclose()

                async def post_writer(endpoint_url: str):
                    try:
                        async with write_stream_reader:
                            async for session_message in write_stream_reader:
                                try:
                                    payload = session_message.message.model_dump(
                                        by_alias=True,
                                        mode="json",
                                        exclude_none=True,
                                    )
                                except Exception:
                                    logger.exception("Invalid session message payload")
                                    continue

                                _emit_channel_event(
                                    channel_hook,
                                    "post-sse",
                                    "message",
                                    message=session_message.message,
                                )

                                try:
                                    response = await client.post(endpoint_url, json=payload)
                                    response.raise_for_status()
                                except httpx.HTTPStatusError as exc:
                                    status_code, detail = _format_http_error(exc)
                                    _emit_channel_event(
                                        channel_hook,
                                        "post-sse",
                                        "error",
                                        detail=detail,
                                        status_code=status_code,
                                    )
                                    raise
                    except httpx.HTTPStatusError:
                        logger.exception("HTTP error in post_writer")
                    except Exception:
                        logger.exception("Error in post_writer")
                        _emit_channel_event(
                            channel_hook,
                            "post-sse",
                            "error",
                            detail="Error sending client message",
                        )
                    finally:
                        await write_stream.aclose()

                try:
                    async with aconnect_sse(
                        client,
                        "GET",
                        url,
                    ) as event_source:
                        try:
                            event_source.response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            status_code, detail = _format_http_error(exc)
                            _emit_channel_event(
                                channel_hook,
                                "get",
                                "error",
                                detail=detail,
                                status_code=status_code,
                            )
                            raise

                        _emit_channel_event(channel_hook, "get", "connect")
                        connected = True

                        endpoint_url = await tg.start(sse_reader)
                        _emit_channel_event(channel_hook, "post-sse", "connect")
                        post_connected = True
                        tg.start_soon(post_writer, endpoint_url)

                        try:
                            yield read_stream, write_stream, get_session_id
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
