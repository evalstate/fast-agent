from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp.client.sse import sse_client

if TYPE_CHECKING:
    import httpx2

from fast_agent.mcp.transport_tracking import ChannelEvent

ChannelHook = Callable[[ChannelEvent], None]


@asynccontextmanager
async def tracking_sse_client(
    url: str,
    headers: dict[str, Any] | None = None,
    *,
    timeout: float = 5,
    sse_read_timeout: float = 300,
    auth: httpx2.Auth | None = None,
    channel_hook: ChannelHook | None = None,
) -> AsyncIterator[tuple[object, object, None]]:
    """Compatibility adapter for the deprecated HTTP+SSE transport."""
    del channel_hook
    async with sse_client(
        url,
        headers,
        timeout=timeout,
        sse_read_timeout=sse_read_timeout,
        auth=auth,
    ) as streams:
        read_stream, write_stream = streams
        yield read_stream, write_stream, None
