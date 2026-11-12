from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.shared.exceptions import MCPError

from fast_agent.mcp.events_extension import (
    EVENTS_FORBIDDEN,
    EVENTS_NOT_FOUND,
    EventDescriptor,
    EventOccurrence,
    ListEventsResult,
    PollEventsResult,
)
from fast_agent.mcp.events_subscriptions import (
    EventInboxItem,
    EventSubscriptionManager,
    JsonEventSubscriptionStore,
    format_event_messages,
    subscription_key,
)

if TYPE_CHECKING:
    from pathlib import Path

EVENT = "job.status_changed"
ARGS = {"job_id": "job-1"}


def _occurrence(stage: str) -> EventOccurrence:
    return EventOccurrence(
        event_id=f"job-1:{stage}",
        name=EVENT,
        timestamp="2026-11-12T10:00:00Z",
        data={"job_id": "job-1", "stage": stage},
    )


class _Poller:
    """Answer polls from a script, then block."""

    def __init__(
        self,
        script: list[PollEventsResult | Exception],
        *,
        delivery: list[str] | None = None,
    ) -> None:
        self.script = list(script)
        self.delivery = ["poll"] if delivery is None else delivery
        self.cursors: list[str | None] = []
        self.list_calls = 0
        self.offered = True
        self.exhausted = asyncio.Event()

    async def list_events(self, server_name: str, cursor: str | None = None) -> ListEventsResult:
        del server_name, cursor
        self.list_calls += 1
        events = [EventDescriptor(name=EVENT, delivery=self.delivery)] if self.offered else []
        return ListEventsResult(events=events)

    async def poll_events(
        self,
        server_name: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        cursor: str | None = None,
        max_age_ms: int | None = None,
        max_events: int | None = None,
    ) -> PollEventsResult:
        del server_name, name, arguments, max_age_ms, max_events
        if not self.script:
            self.exhausted.set()
            await asyncio.Event().wait()
        self.cursors.append(cursor)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0)


def _manager(poller: _Poller, sleeps: _Sleeps, **kwargs: Any) -> EventSubscriptionManager:
    return EventSubscriptionManager(poller, sleep=sleeps, **kwargs)


@pytest.mark.asyncio
async def test_poll_loop_delivers_events_and_follows_server_pacing() -> None:
    poller = _Poller(
        [
            PollEventsResult(cursor="c0", next_poll_ms=5_000),
            PollEventsResult(
                events=[_occurrence("RUNNING")], cursor="c1", has_more=True, next_poll_ms=5_000
            ),
            PollEventsResult(events=[_occurrence("COMPLETED")], cursor="c2", next_poll_ms=10),
        ]
    )
    sleeps = _Sleeps()
    manager = _manager(poller, sleeps)
    arrivals: list[int] = []
    manager.add_listener(lambda item: arrivals.append(manager.pending_count))

    subscription = await manager.subscribe("demo", EVENT, ARGS)
    await poller.exhausted.wait()

    assert poller.cursors == [None, "c0", "c1"]
    # 5 s after bootstrap, no wait while hasMore, then the 10 ms hint raised to the floor.
    assert sleeps.delays == [5.0, 1.0]
    assert arrivals == [1, 2]
    assert subscription.key == subscription_key("demo", EVENT, ARGS)
    assert subscription.delivered == 2
    # The bootstrap poll left nothing pending, so its cursor counts as consumed.
    assert subscription.cursor == "c0"

    items = manager.take()
    assert [item.occurrence.event_id for item in items if item.occurrence] == [
        "job-1:RUNNING",
        "job-1:COMPLETED",
    ]
    assert subscription.cursor == "c2"
    assert manager.pending_count == 0
    await manager.aclose()


@pytest.mark.asyncio
async def test_duplicate_event_ids_are_dropped_and_truncation_reports_a_gap() -> None:
    poller = _Poller(
        [
            PollEventsResult(cursor="c0"),
            PollEventsResult(events=[_occurrence("RUNNING")], cursor="c1"),
            PollEventsResult(events=[_occurrence("RUNNING")], cursor="c2", truncated=True),
            # No replay cursor, so truncated carries no meaning here.
            PollEventsResult(cursor=None, truncated=True),
        ]
    )
    manager = _manager(poller, _Sleeps())

    await manager.subscribe("demo", EVENT, ARGS)
    await poller.exhausted.wait()

    assert [item.kind for item in manager.take()] == ["event", "gap"]
    await manager.aclose()


@pytest.mark.asyncio
async def test_subscribe_rejects_unknown_and_non_poll_event_types() -> None:
    push_only = _manager(_Poller([], delivery=["push", "webhook"]), _Sleeps())
    with pytest.raises(ValueError, match="does not support poll delivery"):
        await push_only.subscribe("demo", EVENT)

    poller = _Poller([])
    poller.offered = False
    with pytest.raises(ValueError, match="does not offer event"):
        await _manager(poller, _Sleeps()).subscribe("demo", EVENT)

    rejecting = _manager(_Poller([MCPError(-32602, "InvalidParams")]), _Sleeps())
    with pytest.raises(MCPError):
        await rejecting.subscribe("demo", EVENT, {"job_id": 1})
    assert rejecting.subscriptions() == []


@pytest.mark.asyncio
async def test_forbidden_terminates_the_subscription() -> None:
    poller = _Poller([PollEventsResult(cursor="c0"), MCPError(EVENTS_FORBIDDEN, "Forbidden")])
    manager = _manager(poller, _Sleeps())

    subscription = await manager.subscribe("demo", EVENT, ARGS)
    while subscription.status == "active":
        await asyncio.sleep(0)

    assert subscription.last_error == "Forbidden"
    assert [item.kind for item in manager.take()] == ["terminated"]
    assert manager.subscriptions() == [subscription]
    assert await manager.unsubscribe(subscription.key)
    assert manager.subscriptions() == []


@pytest.mark.asyncio
async def test_not_found_rediscovers_once_then_terminates() -> None:
    poller = _Poller(
        [
            PollEventsResult(cursor="c0"),
            MCPError(EVENTS_NOT_FOUND, "NotFound"),
            PollEventsResult(cursor="c9"),
            MCPError(EVENTS_NOT_FOUND, "NotFound"),
            MCPError(EVENTS_NOT_FOUND, "NotFound"),
        ]
    )
    manager = _manager(poller, _Sleeps())

    subscription = await manager.subscribe("demo", EVENT, ARGS)
    while subscription.status == "active":
        await asyncio.sleep(0)

    # The poll after each rediscovery starts from now.
    assert poller.cursors == [None, "c0", None, "c9", None]
    assert poller.list_calls == 3
    assert [item.kind for item in manager.take()] == ["gap", "gap", "terminated"]


@pytest.mark.asyncio
async def test_transport_errors_back_off_and_keep_the_cursor() -> None:
    poller = _Poller(
        [
            PollEventsResult(cursor="c0", next_poll_ms=2_000),
            ConnectionError("down"),
            ConnectionError("down"),
            PollEventsResult(events=[_occurrence("COMPLETED")], cursor="c1", next_poll_ms=2_000),
        ]
    )
    sleeps = _Sleeps()
    manager = _manager(poller, sleeps, default_poll_ms=4_000)

    subscription = await manager.subscribe("demo", EVENT, ARGS)
    await poller.exhausted.wait()

    assert poller.cursors == [None, "c0", "c0", "c0"]
    assert sleeps.delays == [2.0, 4.0, 8.0, 2.0]
    assert subscription.last_error is None
    assert subscription.delivered == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_store_keeps_the_consumed_cursor_and_restores_from_it(tmp_path: Path) -> None:
    store = JsonEventSubscriptionStore(tmp_path / "events" / "subscriptions.json")
    poller = _Poller(
        [
            PollEventsResult(cursor="c0"),
            PollEventsResult(events=[_occurrence("COMPLETED")], cursor="c1"),
        ]
    )
    manager = _manager(poller, _Sleeps(), store=store)

    subscription = await manager.subscribe("demo", EVENT, ARGS)
    await poller.exhausted.wait()
    # The event is still in the inbox, so a restart must replay it.
    assert store.load() == [
        {"server_name": "demo", "name": EVENT, "arguments": ARGS, "cursor": "c0"}
    ]
    await manager.aclose()

    replay = _Poller([PollEventsResult(events=[_occurrence("COMPLETED")], cursor="c1")])
    restarted = _manager(replay, _Sleeps(), store=store)
    restored = restarted.restore()
    await replay.exhausted.wait()

    assert [item.key for item in restored] == [subscription.key]
    assert replay.cursors == ["c0"]
    assert len(restarted.take()) == 1
    assert store.load()[0]["cursor"] == "c1"

    assert await restarted.unsubscribe(subscription.key)
    assert store.load() == []
    assert not await restarted.unsubscribe(subscription.key)
    await restarted.aclose()


def test_event_messages_mark_payloads_as_untrusted() -> None:
    text = format_event_messages(
        [
            EventInboxItem("event", "k", "demo", EVENT, occurrence=_occurrence("COMPLETED")),
            EventInboxItem("gap", "k", "demo", EVENT),
            EventInboxItem("terminated", "k", "demo", EVENT, detail="Access revoked"),
        ]
    )

    assert text.startswith("MCP event notifications")
    assert "untrusted data" in text
    assert '<mcp-event server="demo" name="job.status_changed" id="job-1:COMPLETED"' in text
    assert '"stage": "COMPLETED"' in text
    assert "<mcp-event-gap" in text
    assert "Access revoked" in text
