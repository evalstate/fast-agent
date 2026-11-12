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
