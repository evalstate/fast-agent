"""Client-side streams for MCP Events push delivery (SEP-3415 ``events/stream``)."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from mcp.client.extension import ClientExtension, NotificationBinding
from mcp.shared.dispatcher import as_request_id, coerce_request_id
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY

from fast_agent.mcp.events_extension import (
    EVENT_STREAM_NOTIFICATIONS,
    EVENTS_EXTENSION,
    EventStreamFrame,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp_types import RequestId

STREAM_OPEN_TIMEOUT_SECONDS = 30.0


class EventStream:
    """One open ``events/stream`` request, as an async iterator of its messages.

    Iteration ends when the server closes the stream, and raises what the
    request raised when it failed. ``aclose`` cancels the request, which ends
    the subscription on the server.
    """

    def __init__(self, request: asyncio.Future[object]) -> None:
        self.request = request
        self.frames: deque[EventStreamFrame] = deque()
        self.changed = asyncio.Event()
        request.add_done_callback(lambda _: self.changed.set())

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> EventStreamFrame:
        while not self.frames:
            if self.request.done():
                if not self.request.cancelled():
                    self.request.result()
                raise StopAsyncIteration
            self.changed.clear()
            await self.changed.wait()
        return self.frames.popleft()

    async def aclose(self) -> None:
        self.request.cancel()
        await asyncio.gather(self.request, return_exceptions=True)


class EventStreamRouter(ClientExtension):
    """Advertise the Events extension and give each open stream its messages.

    A message names its stream by the id of the ``events/stream`` request.
    ``ClientSession.send_request`` does not report the id it assigns, so
    streams open one at a time and the first message with an unknown id
    belongs to the stream that is opening.

    The SDK queues each notification method separately, so messages of
    different kinds can arrive out of wire order.
    """

    identifier = EVENTS_EXTENSION

    def __init__(self, *, open_timeout_seconds: float = STREAM_OPEN_TIMEOUT_SECONDS) -> None:
        self._open_timeout_seconds = open_timeout_seconds
        self._streams: dict[RequestId, EventStream] = {}
        self._opening: EventStream | None = None
        self._open_lock = asyncio.Lock()

    def notifications(self) -> list[NotificationBinding[Any]]:
        return [
            NotificationBinding(method=method, params_type=frame_type, handler=self._receive)
            for method, frame_type in EVENT_STREAM_NOTIFICATIONS.items()
        ]

    async def _receive(self, frame: EventStreamFrame) -> None:
        raw_id = as_request_id((frame.meta or {}).get(SUBSCRIPTION_ID_META_KEY))
        if raw_id is None:
            return
        subscription_id = coerce_request_id(raw_id)
        stream = self._streams.get(subscription_id)
        if stream is None:
            stream, self._opening = self._opening, None
            if stream is None:
                return
            self._streams[subscription_id] = stream
            stream.request.add_done_callback(lambda _: self._streams.pop(subscription_id, None))
        stream.frames.append(frame)
        stream.changed.set()

    async def open(self, send: Callable[[], Awaitable[object]]) -> EventStream:
        """Send an ``events/stream`` request and return its stream once a message arrives.

        Raises what the request raises when the server rejects it, and
        ``TimeoutError`` when the server stays silent.
        """
        async with self._open_lock:
            stream = self._opening = EventStream(asyncio.ensure_future(send()))
            try:
                async with asyncio.timeout(self._open_timeout_seconds):
                    await stream.changed.wait()
                if not stream.frames:
                    await stream.request
                    raise RuntimeError("The server closed the event stream before any message")
            except BaseException:
                await stream.aclose()
                raise
            finally:
                self._opening = None
        return stream
