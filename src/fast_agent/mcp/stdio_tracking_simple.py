from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TextIO

from mcp.client.stdio import StdioServerParameters, stdio_client

from fast_agent.mcp.transport_tracking import ChannelEvent, EventType

if TYPE_CHECKING:
    from mcp.client import Transport

logger = logging.getLogger(__name__)

ChannelHook = Callable[[ChannelEvent], None]


def tracking_stdio_client(
    server_params: StdioServerParameters,
    *,
    channel_hook: ChannelHook | None = None,
    errlog: TextIO | None = None,
) -> Transport:
    """Context manager for stdio client with basic connection tracking."""

    @asynccontextmanager
    async def tracked():
        def emit_channel_event(event_type: EventType, detail: str | None = None) -> None:
            if channel_hook is None:
                return
            try:
                channel_hook(
                    ChannelEvent(
                        channel="stdio",
                        event_type=event_type,
                        detail=detail,
                    )
                )
            except Exception:  # pragma: no cover - hook errors must not break transport
                logger.exception("Channel hook raised an exception")

        try:
            emit_channel_event("connect")

            if errlog is None:
                async with stdio_client(server_params) as streams:
                    yield streams
            else:
                async with stdio_client(server_params, errlog=errlog) as streams:
                    yield streams
        except Exception as exc:
            emit_channel_event("error", detail=str(exc))
            raise
        finally:
            emit_channel_event("disconnect")

    return tracked()
