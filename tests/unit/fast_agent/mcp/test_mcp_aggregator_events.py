from __future__ import annotations

from typing import Any

import pytest
from mcp_types import ServerCapabilities

from fast_agent.context import Context
from fast_agent.mcp.events_extension import ListEventsResult, PollEventsResult
from fast_agent.mcp.mcp_aggregator import MCPAggregator


class _BaseAggregator(MCPAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialized = True

    async def validate_server(self, server_name: str) -> bool:
        return server_name in self.server_names


@pytest.mark.asyncio
async def test_events_extension_routes_requests_to_the_named_server() -> None:
    calls: list[tuple[str, str, str, str, dict[str, Any] | None]] = []

    class _EventsAggregator(_BaseAggregator):
        async def get_capabilities(self, server_name: str) -> ServerCapabilities:
            assert server_name == "demo"
            return ServerCapabilities.model_validate(
                {"extensions": {"io.modelcontextprotocol/events": {}}}
            )

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del error_factory, progress_callback
            calls.append((server_name, operation_type, operation_name, method_name, method_args))
            if method_name == "list_events":
                return ListEventsResult(events=[])
            return PollEventsResult(cursor="c1")

    aggregator = _EventsAggregator(
        server_names=["demo"], connection_persistence=False, context=Context()
    )

    assert await aggregator.supports_events("demo")
    assert not await aggregator.supports_events("other")
    await aggregator.list_events("demo")
    await aggregator.list_events("demo", cursor="page-2")
    await aggregator.poll_events("demo", "job.status_changed")
    polled = await aggregator.poll_events(
        "demo",
        "job.status_changed",
        {"job_id": "job-1"},
        cursor="c0",
        max_age_ms=300_000,
        max_events=50,
    )

    assert polled.cursor == "c1"
    assert calls == [
        ("demo", "events/list", "", "list_events", None),
        ("demo", "events/list", "", "list_events", {"cursor": "page-2"}),
        (
            "demo",
            "events/poll",
            "job.status_changed",
            "poll_events",
            {"name": "job.status_changed", "arguments": {}},
        ),
        (
            "demo",
            "events/poll",
            "job.status_changed",
            "poll_events",
            {
                "name": "job.status_changed",
                "arguments": {"job_id": "job-1"},
                "cursor": "c0",
                "max_age_ms": 300_000,
                "max_events": 50,
            },
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["list_events", "poll_events"])
@pytest.mark.parametrize(
    "capabilities",
    [
        None,
        {},
        {"extensions": {"unrelated/extension": {}}},
    ],
)
async def test_events_guards_reject_before_dispatch(method: str, capabilities) -> None:
    class _GuardAggregator(_BaseAggregator):
        async def get_capabilities(self, server_name: str) -> ServerCapabilities | None:
            assert server_name == "demo"
            return (
                ServerCapabilities.model_validate(capabilities)
                if capabilities is not None
                else None
            )

        async def _execute_on_server(self, *args, **kwargs):
            raise AssertionError("Unsupported request dispatched")

    aggregator = _GuardAggregator(
        server_names=["demo"], connection_persistence=False, context=Context()
    )
    assert not await aggregator.supports_events("demo")
    with pytest.raises(ValueError, match="does not support the events extension"):
        if method == "list_events":
            await aggregator.list_events("demo")
        else:
            await aggregator.poll_events("demo", "job.status_changed")


@pytest.mark.asyncio
async def test_take_event_prompt_messages_clears_toolbar_pending_count() -> None:
    from fast_agent.mcp.events_extension import EventDescriptor, EventOccurrence
    from fast_agent.mcp.events_subscriptions import is_mcp_events_message
    from fast_agent.ui import notification_tracker

    class _SubscribedAggregator(_BaseAggregator):
        async def list_events(
            self, server_name: str, cursor: str | None = None
        ) -> ListEventsResult:
            del server_name, cursor
            return ListEventsResult(
                events=[EventDescriptor(name="job.status_changed", delivery=["poll"])]
            )

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
            if cursor is None:
                return PollEventsResult(
                    events=[
                        EventOccurrence(
                            event_id="ev-1",
                            name="job.status_changed",
                            timestamp="2026-11-12T10:00:00Z",
                            data={"stage": "DONE"},
                        )
                    ],
                    cursor="c1",
                    next_poll_ms=60_000,
                )
            return PollEventsResult(cursor="c1", next_poll_ms=60_000)

    notification_tracker.clear()
    notification_tracker.add_tool_update("other")
    aggregator = _SubscribedAggregator(
        server_names=["demo"], connection_persistence=False, context=Context()
    )
    try:
        await aggregator.events.subscribe("demo", "job.status_changed")
        assert notification_tracker.get_counts_by_type() == {"tool_update": 1, "mcp_event": 1}

        messages = aggregator.take_event_prompt_messages()
        assert len(messages) == 1
        assert is_mcp_events_message(messages[0])
        assert notification_tracker.get_counts_by_type() == {"tool_update": 1}
    finally:
        await aggregator.events.aclose()
        notification_tracker.clear()


@pytest.mark.asyncio
async def test_collect_server_status_populates_mcp_events_enabled() -> None:
    class _StatusAggregator(_BaseAggregator):
        async def _capabilities_for_status(self, server_name: str) -> ServerCapabilities | None:
            if server_name == "with-events":
                return ServerCapabilities.model_validate(
                    {"extensions": {"io.modelcontextprotocol/events": {}}}
                )
            return ServerCapabilities()

    aggregator = _StatusAggregator(
        server_names=["with-events", "plain"],
        connection_persistence=False,
        context=Context(),
    )
    status_map = await aggregator.collect_server_status()
    assert status_map["with-events"].mcp_events_enabled is True
    assert status_map["plain"].mcp_events_enabled is False


@pytest.mark.asyncio
async def test_toolbar_pending_event_count_is_scoped_per_aggregator_and_cleared_on_close() -> None:
    from fast_agent.mcp.events_extension import EventDescriptor, EventOccurrence
    from fast_agent.ui import notification_tracker

    class _SubscribedAggregator(_BaseAggregator):
        def __init__(self, event_id: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._event_id = event_id

        async def list_events(
            self, server_name: str, cursor: str | None = None
        ) -> ListEventsResult:
            del server_name, cursor
            return ListEventsResult(
                events=[EventDescriptor(name="job.status_changed", delivery=["poll"])]
            )

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
            if cursor is None:
                return PollEventsResult(
                    events=[
                        EventOccurrence(
                            event_id=self._event_id,
                            name="job.status_changed",
                            timestamp="2026-11-12T10:00:00Z",
                            data={"stage": "DONE"},
                        )
                    ],
                    cursor="c1",
                    next_poll_ms=60_000,
                )
            return PollEventsResult(cursor="c1", next_poll_ms=60_000)

    notification_tracker.clear()
    agg_one = _SubscribedAggregator(
        "ev-1", server_names=["demo"], connection_persistence=False, context=Context()
    )
    agg_two = _SubscribedAggregator(
        "ev-2", server_names=["demo"], connection_persistence=False, context=Context()
    )
    try:
        await agg_one.events.subscribe("demo", "job.status_changed")
        await agg_two.events.subscribe("demo", "job.status_changed")
        assert notification_tracker.get_counts_by_type() == {"mcp_event": 2}

        # Draining agg_one keeps agg_two's pending count intact.
        assert len(agg_one.take_event_prompt_messages()) == 1
        assert notification_tracker.get_counts_by_type() == {"mcp_event": 1}

        # Closing agg_two without draining clears its unread event notification.
        await agg_two.close()
        assert notification_tracker.get_counts_by_type() == {}
    finally:
        await agg_one.close()
        await agg_two.close()
        notification_tracker.clear()
