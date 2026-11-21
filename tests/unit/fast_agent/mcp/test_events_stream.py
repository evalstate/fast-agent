from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY

from fast_agent.mcp.events_extension import (
    EventStreamActive,
    EventStreamFrame,
    EventStreamHeartbeat,
)
from fast_agent.mcp.events_stream import EventStreamRouter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp_types import RequestId


async def _notify(router: EventStreamRouter, method: str, frame: EventStreamFrame) -> None:
    handlers = {binding.method: binding.handler for binding in router.notifications()}
    await handlers[f"notifications/events/{method}"](frame)


def _active(subscription_id: RequestId) -> EventStreamActive:
    return EventStreamActive(cursor="c0", meta={SUBSCRIPTION_ID_META_KEY: subscription_id})


def _heartbeat(subscription_id: RequestId, cursor: str) -> EventStreamHeartbeat:
    return EventStreamHeartbeat(cursor=cursor, meta={SUBSCRIPTION_ID_META_KEY: subscription_id})


@pytest.mark.asyncio
async def test_streams_open_one_at_a_time_and_each_receives_its_own_messages() -> None:
    router = EventStreamRouter()
    server_closes = asyncio.Event()
    sent: list[str] = []

    def request(label: str) -> Callable[[], Awaitable[None]]:
        async def send() -> None:
            sent.append(label)
            await server_closes.wait()

        return send

    opening_one = asyncio.create_task(router.open(request("one")))
    opening_two = asyncio.create_task(router.open(request("two")))
    for _ in range(5):
        await asyncio.sleep(0)
    assert sent == ["one"]

    await _notify(router, "active", _active(7))
    one = await opening_one
    while sent != ["one", "two"]:
        await asyncio.sleep(0)
    await _notify(router, "active", _active("stream-8"))
    two = await opening_two

    await _notify(router, "heartbeat", _heartbeat("7", "c1"))
    await _notify(router, "heartbeat", _heartbeat("stream-8", "c2"))
    await _notify(router, "heartbeat", _heartbeat(99, "c3"))
    await _notify(router, "heartbeat", EventStreamHeartbeat(cursor="c4"))
    server_closes.set()

    assert [frame async for frame in one] == [_active(7), _heartbeat("7", "c1")]
    assert [frame async for frame in two] == [_active("stream-8"), _heartbeat("stream-8", "c2")]


@pytest.mark.asyncio
async def test_open_fails_when_the_server_is_silent_or_rejects_the_request() -> None:
    router = EventStreamRouter(open_timeout_seconds=0.01)
    cancelled = asyncio.Event()

    async def silent() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await router.open(silent)
    assert cancelled.is_set()

    async def rejects() -> None:
        raise MCPError(-32602, "InvalidParams")

    with pytest.raises(MCPError):
        await router.open(rejects)


@pytest.mark.asyncio
async def test_stream_raises_when_its_request_fails_and_closing_cancels_the_request() -> None:
    router = EventStreamRouter()
    connection_drops = asyncio.Event()
    cancelled = asyncio.Event()

    async def drops() -> None:
        await connection_drops.wait()
        raise ConnectionError("gone")

    async def held() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    opening = asyncio.create_task(router.open(drops))
    await asyncio.sleep(0)
    await _notify(router, "active", _active(1))
    dropped = await opening
    connection_drops.set()
    assert await anext(dropped) == _active(1)
    with pytest.raises(ConnectionError):
        await anext(dropped)

    opening = asyncio.create_task(router.open(held))
    await asyncio.sleep(0)
    await _notify(router, "active", _active(2))
    await (await opening).aclose()
    assert cancelled.is_set()
