from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from mcp.client.stdio import StdioServerParameters, stdio_client

from fast_agent.mcp.transport_tracking import ChannelEvent

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream
    from mcp.shared.message import SessionMessage

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


@asynccontextmanager
async def tracking_stdio_client(
    server_params: StdioServerParameters,
    *,
    channel_hook: ChannelHook | None = None,
    errlog: Callable[[str], None] | None = None,
) -> AsyncGenerator[
    tuple[ObjectReceiveStream[SessionMessage | Exception], ObjectSendStream[SessionMessage]], None
]:
    """Context manager for stdio client with basic connection tracking."""

    def emit_channel_event(event_type: str, detail: str | None = None) -> None:
        if channel_hook is None:
            return
        try:
            channel_hook(
                ChannelEvent(
                    channel="stdio",
                    event_type=event_type,  # type: ignore[arg-type]
                    detail=detail,
                )
            )
        except Exception:  # pragma: no cover - hook errors must not break transport
            logger.exception("Channel hook raised an exception")

    try:
        # Emit connection event
        emit_channel_event("connect")

        # Use the original stdio_client without stream interception
        async with stdio_client(server_params, errlog=errlog) as (read_stream, write_stream):
            yield read_stream, write_stream

    except Exception as exc:
        # Emit error event
        emit_channel_event("error", detail=str(exc))
        raise
    finally:
        # Emit disconnection event
        emit_channel_event("disconnect")
