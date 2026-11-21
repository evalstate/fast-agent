"""Unit tests for the built-in model-controlled ``events`` tool."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp_types import TextContent

from fast_agent.mcp.events_extension import (
    EventDescriptor,
    ListEventsResult,
    PollEventsResult,
)
from fast_agent.mcp.events_subscriptions import EventSubscriptionManager
from fast_agent.ptc.vocabulary import is_builtin_tool
from fast_agent.tools.events_tool import (
    EVENTS_TOOL_NAME,
    build_events_tool,
    execute_events_tool,
)


class _FakeEventsAggregator:
    def __init__(
        self,
        descriptors_by_server: dict[str, list[EventDescriptor]],
        *,
        non_events_servers: list[str] | None = None,
        max_consecutive_wakes: int = 2,
    ) -> None:
        self._descriptors = descriptors_by_server
        self._non_events = set(non_events_servers or [])
        self.server_names = [*descriptors_by_server.keys(), *(non_events_servers or [])]
        self._events = EventSubscriptionManager(
            self,
            min_poll_ms=10,
            default_poll_ms=10_000,
            max_consecutive_wakes=max_consecutive_wakes,
        )

    @property
    def events(self) -> EventSubscriptionManager:
        return self._events

    async def supports_events(self, server_name: str) -> bool:
        return server_name in self._descriptors and server_name not in self._non_events

    async def list_events(self, server_name: str, cursor: str | None = None) -> ListEventsResult:
        return ListEventsResult(events=list(self._descriptors.get(server_name, [])))

    def can_hold_event_stream(self, server_name: str) -> bool:
        return True

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
        return PollEventsResult(events=[], cursor=cursor or "c1", next_poll_ms=60_000)


def _sample_aggregator() -> _FakeEventsAggregator:
    return _FakeEventsAggregator(
        {
            "jobs": [
                EventDescriptor(
                    name="job.completed",
                    description="Emitted when a build job completes.",
                    delivery=["poll"],
                    input_schema={
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                ),
                EventDescriptor(
                    name="job.push_only",
                    description="Push only event.",
                    delivery=["push"],
                ),
            ],
            "alerts": [
                EventDescriptor(
                    name="alert.fired",
                    description="Emitted on alert.",
                    delivery=["poll"],
                )
            ],
        },
        non_events_servers=["plain"],
    )


def test_build_events_tool_contract() -> None:
    tool = build_events_tool()
    assert tool.name == EVENTS_TOOL_NAME == "events"
    assert is_builtin_tool(tool)
    description = tool.description or ""
    for phrase in (
        "list",
        "status",
        "subscribe",
        "unsubscribe",
        "<mcp-event subscription=",
        "wake",
    ):
        assert phrase in description
    props = tool.input_schema["properties"]
    assert props["action"]["enum"] == ["list", "status", "subscribe", "unsubscribe"]
    for prop in ("server", "name", "arguments", "wake", "once", "every", "id"):
        assert prop in props


@pytest.mark.asyncio
async def test_events_tool_list_single_and_all_servers() -> None:
    agg = _sample_aggregator()
    try:
        one = await execute_events_tool(agg, {"action": "list", "server": "jobs"})
        assert not one.is_error
        assert isinstance(one.structured_content, dict)
        events = one.structured_content["events"]
        assert [e["name"] for e in events] == ["job.completed", "job.push_only"]
        assert events[0]["server"] == "jobs"
        assert events[0]["required_arguments"] == ["job_id"]
        assert events[0]["delivery"] == ["poll"]
        assert events[0]["input_schema"]["required"] == ["job_id"]
        assert isinstance(one.content[0], TextContent)
        assert json.loads(one.content[0].text) == one.structured_content

        all_res = await execute_events_tool(agg, {"action": "list"})
        assert not all_res.is_error
        assert isinstance(all_res.structured_content, dict)
        servers = {e["server"] for e in all_res.structured_content["events"]}
        assert servers == {"jobs", "alerts"}

        unsupported = await execute_events_tool(agg, {"action": "list", "server": "plain"})
        assert unsupported.is_error
    finally:
        await agg.events.aclose()


@pytest.mark.asyncio
async def test_events_tool_subscribe_status_and_unsubscribe() -> None:
    agg = _sample_aggregator()
    try:
        sub_res = await execute_events_tool(
            agg,
            {
                "action": "subscribe",
                "server": "jobs",
                "name": "job.completed",
                "arguments": {"job_id": "42"},
                "wake": True,
                "once": True,
                "every": "5m",
            },
        )
        assert not sub_res.is_error
        assert isinstance(sub_res.structured_content, dict)
        sub_id = sub_res.structured_content["id"]
        assert sub_res.structured_content["server"] == "jobs"
        assert sub_res.structured_content["name"] == "job.completed"
        assert sub_res.structured_content["wake"] is True
        assert sub_res.structured_content["once"] is True
        assert sub_res.structured_content["every"] == "5m"
        assert sub_res.structured_content["status"] == "active"

        status_res = await execute_events_tool(agg, {"action": "status"})
        assert not status_res.is_error
        assert isinstance(status_res.structured_content, dict)
        assert status_res.structured_content["wake_paused"] is False
        subs = status_res.structured_content["subscriptions"]
        assert len(subs) == 1
        assert subs[0]["id"] == sub_id
        assert subs[0]["origin"] == "model"
        assert subs[0]["arguments"] == {"job_id": "42"}
        assert subs[0]["every"] == "5m"
        assert subs[0]["delivered"] == 0

        agg.events.record_wake_turn()
        agg.events.record_wake_turn()
        paused_status = await execute_events_tool(agg, {"action": "status"})
        assert isinstance(paused_status.structured_content, dict)
        assert paused_status.structured_content["wake_paused"] is True

        unsub_res = await execute_events_tool(agg, {"action": "unsubscribe", "id": sub_id})
        assert not unsub_res.is_error
        assert unsub_res.structured_content == {"id": sub_id, "unsubscribed": True}

        missing = await execute_events_tool(agg, {"action": "unsubscribe", "id": sub_id})
        assert missing.is_error
    finally:
        await agg.events.aclose()


@pytest.mark.asyncio
async def test_events_tool_respects_model_cap_and_user_ownership() -> None:
    agg = _sample_aggregator()
    try:
        user_sub = await agg.events.subscribe(
            "jobs", "job.completed", {"job_id": "user-owned"}, origin="user"
        )

        # Model cannot mutate or unsubscribe a user-owned subscription.
        mutate_err = await execute_events_tool(
            agg,
            {
                "action": "subscribe",
                "server": "jobs",
                "name": "job.completed",
                "arguments": {"job_id": "user-owned"},
                "wake": True,
            },
        )
        assert mutate_err.is_error

        unsub_err = await execute_events_tool(agg, {"action": "unsubscribe", "id": user_sub.key})
        assert unsub_err.is_error

        # Model subscription cap of 8 is enforced.
        for idx in range(8):
            res = await execute_events_tool(
                agg,
                {
                    "action": "subscribe",
                    "server": "jobs",
                    "name": "job.completed",
                    "arguments": {"job_id": f"m-{idx}"},
                },
            )
            assert not res.is_error

        ninth = await execute_events_tool(
            agg,
            {
                "action": "subscribe",
                "server": "jobs",
                "name": "job.completed",
                "arguments": {"job_id": "m-overflow"},
            },
        )
        assert ninth.is_error
        assert isinstance(ninth.content[0], TextContent)
        assert "limit" in ninth.content[0].text.lower()
    finally:
        await agg.events.aclose()


@pytest.mark.asyncio
async def test_events_tool_validation_errors_never_raise() -> None:
    agg = _sample_aggregator()
    try:
        for bad_args in (
            None,
            {},
            {"action": "bogus"},
            {"action": "subscribe"},
            {"action": "subscribe", "server": "jobs"},
            {"action": "subscribe", "server": "jobs", "name": "job.completed", "every": "soon"},
            {"action": "subscribe", "server": "jobs", "name": "job.completed", "arguments": "bad"},
            {"action": "subscribe", "server": "jobs", "name": "job.completed", "wake": "yes"},
            {"action": "unsubscribe"},
        ):
            res = await execute_events_tool(agg, bad_args)
            assert res.is_error
    finally:
        await agg.events.aclose()


@pytest.mark.asyncio
async def test_events_tool_list_refreshes_descriptors() -> None:
    agg = _sample_aggregator()
    try:
        # Warm cache first.
        await agg.events.list_descriptors("jobs")
        first = await execute_events_tool(agg, {"action": "list", "server": "jobs"})
        assert not first.is_error
        # Add a new event type on the server and verify action="list" refreshes the cache.
        agg._descriptors["jobs"].append(
            EventDescriptor(name="job.restarted", description="Restarted", delivery=["poll"])
        )
        second = await execute_events_tool(agg, {"action": "list", "server": "jobs"})
        assert not second.is_error
        assert second.structured_content is not None
        names = [item["name"] for item in second.structured_content["events"]]
        assert "job.restarted" in names
    finally:
        await agg.events.aclose()


@pytest.mark.asyncio
async def test_local_function_named_events_runs_without_events_server() -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.mcp_agent import McpAgent

    def events(topic: str = "general") -> str:
        """Return user-defined local events summary."""
        return f"local-events:{topic}"

    agent = McpAgent(config=AgentConfig("test-events-fn"), tools=[events])
    listed = await agent.list_tools()
    assert any(tool.name == "events" for tool in listed.tools)

    result = await agent.call_tool("events", {"topic": "deploy"})
    assert not result.is_error
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "local-events:deploy"
