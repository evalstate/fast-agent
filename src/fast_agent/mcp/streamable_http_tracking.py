from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage

from fast_agent.mcp.transport_tracking import ChannelEvent

if TYPE_CHECKING:
    import httpx2

ChannelHook = Callable[[ChannelEvent], None]


@asynccontextmanager
async def tracking_streamablehttp_client(
    url: str,
    *,
    http_client: httpx2.AsyncClient | None = None,
    channel_hook: ChannelHook | None = None,
) -> AsyncIterator[tuple[object, object, None]]:
    """Compatibility adapter over the SDK v2 transport.

    SDK v2 owns protocol-era routing, HTTP channels, cancellation, and legacy
    session behavior. Low-level channel callbacks are intentionally unavailable
    until the SDK exposes a public diagnostics hook.
    """
    del channel_hook
    async with streamable_http_client(url, http_client=http_client) as streams:
        read_stream, write_stream = streams
        yield read_stream, write_stream, None


__all__ = ["SessionMessage", "tracking_streamablehttp_client"]
