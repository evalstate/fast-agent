"""Tests for AsyncEventBus._process_events task_done() on the happy path."""

import asyncio

import pytest

from fast_agent.core.logging.events import Event
from fast_agent.core.logging.listeners import EventListener
from fast_agent.core.logging.transport import AsyncEventBus


class CountingListener(EventListener):
    """A listener that counts handled events."""

    def __init__(self) -> None:
        self.count = 0

    async def handle_event(self, event: Event) -> None:
        self.count += 1


@pytest.mark.asyncio
async def test_task_done_called_on_happy_path() -> None:
    """After processing events, queue.join() should complete promptly
    instead of timing out (which was the pre-fix behaviour)."""
    # Use a fresh bus instance (don't pollute the singleton)
    bus = AsyncEventBus.__new__(AsyncEventBus)
    bus.transport = type("T", (), {"send_event": staticmethod(lambda e: asyncio.sleep(0))})()
    bus.listeners = {}
    bus._queue = None
    bus._task = None
    bus._running = False

    listener = CountingListener()
    bus.add_listener("counter", listener)
    await bus.start()

    n_events = 10
    for i in range(n_events):
        await bus.emit(
            Event(
                type="info",
                namespace="test",
                message=f"event-{i}",
            )
        )

    # Give the processing loop a moment to drain
    for _ in range(50):
        if listener.count >= n_events:
            break
        await asyncio.sleep(0.02)

    assert listener.count == n_events, f"Expected {n_events} events, got {listener.count}"

    # The key test: join() should complete quickly if task_done() was called.
    # Before the fix this would always time out.
    try:
        assert bus._queue is not None
        await asyncio.wait_for(bus._queue.join(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail(
            "queue.join() timed out — task_done() is likely missing on the happy path"
        )

    await bus.stop()
