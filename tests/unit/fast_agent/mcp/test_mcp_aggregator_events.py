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


@pytest.mark.asyncio
async def test_declared_events_pass_once_every_and_origin_and_aggregator_wake_helpers() -> None:
    from pydantic import ValidationError

    from fast_agent.config import MCPEventSubscriptionSettings, MCPServerSettings
    from fast_agent.mcp.events_extension import EventDescriptor

    with pytest.raises(ValidationError):
        MCPEventSubscriptionSettings(name="job.status_changed", every="invalid")
    with pytest.raises(ValidationError):
        MCPEventSubscriptionSettings(name="job.status_changed", every=True)  # type: ignore[arg-type]

    class _DeclaredAggregator(_BaseAggregator):
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
            del server_name, name, arguments, cursor, max_age_ms, max_events
            return PollEventsResult(cursor="c1", next_poll_ms=60_000)

    aggregator = _DeclaredAggregator(
        server_names=["demo"],
        connection_persistence=False,
        context=Context(),
    )
    aggregator._attached_server_names.append("demo")
    aggregator._attachment_configs["demo"] = MCPServerSettings(
        name="demo",
        command="echo",
        events=[
            MCPEventSubscriptionSettings(
                name="job.status_changed",
                arguments={"job_id": "1"},
                wake=True,
                once=True,
                every="5m",
            )
        ],
    )
    try:
        await aggregator.start_configured_events()
        [sub] = aggregator.events.subscriptions()
        assert sub.origin == "declared"
        assert sub.wake is True
        assert sub.once is True
        assert sub.min_poll_ms == 300_000
        assert aggregator.active_wake_subscriptions() == [sub]

        # Model cannot modify or unsubscribe declared subscriptions.
        with pytest.raises(ValueError, match="declared-owned"):
            await aggregator.events.unsubscribe(sub.key, origin="model")

        # Wake budget helpers pass through to EventSubscriptionManager.
        for _ in range(aggregator.events.max_consecutive_wakes):
            aggregator.record_event_wake_turn()
        assert aggregator.event_wake_paused is True
        aggregator.reset_event_wake_budget()
        assert aggregator.event_wake_paused is False
    finally:
        await aggregator.close()


@pytest.mark.asyncio
async def test_interactive_prompt_does_not_reset_wake_budget_on_hc_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import cast

    from fast_agent.agents.agent_types import AgentType
    from fast_agent.core.agent_app import AgentRefreshResult
    from fast_agent.ui import interactive_prompt
    from fast_agent.ui.interactive_prompt import InteractivePrompt

    class _BudgetAgent:
        def __init__(self) -> None:
            self.message_history: list[Any] = []
            self.resets = 0
            self.records = 0

        def record_event_wake_turn(self) -> None:
            self.records += 1

        def reset_event_wake_budget(self) -> None:
            self.resets += 1

    class _App:
        def __init__(self) -> None:
            self.agent = _BudgetAgent()
            self.no_home_mode = False
            self.missing_shell_cwd_policy_override = None

        async def refresh_if_needed(self) -> bool:
            return False

        def latest_refresh_result(self) -> AgentRefreshResult:
            return AgentRefreshResult(changed=False)

        def _agent(self, _name: str) -> _BudgetAgent:
            return self.agent

        def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
            del force_include
            return ["agent1"]

        def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
            del force_include
            return {"agent1": AgentType.BASIC}

        def registered_agent_names(self) -> list[str]:
            return ["agent1"]

        def registered_agents(self) -> dict[str, _BudgetAgent]:
            return {"agent1": self.agent}

        def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
            return agent_name or "agent1"

    app = _App()
    inputs = iter(['<hc from="peer">automated ping</hc>', "hello human", "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **_kwargs: Any) -> str:
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(_payload: Any, _agent_name: str) -> str:
        return "ok"

    await InteractivePrompt().prompt_loop(
        send_func=fake_send,
        default_agent="agent1",
        available_agents=["agent1"],
        prompt_provider=cast("Any", app),
    )

    # Only the human-typed turn reset the budget; the <hc> turn did not.
    assert app.agent.resets == 1


@pytest.mark.asyncio
async def test_clear_agent_histories_calls_reset_events() -> None:
    from fast_agent.commands.handlers.shared import clear_agent_histories

    class _ResettableAgent:
        def __init__(self) -> None:
            self.name = "main"
            self.usage_accumulator = None
            self.cleared = False
            self.events_reset = False

        def clear(self, *, clear_prompts: bool = False) -> None:
            del clear_prompts
            self.cleared = True

        def pop_last_message(self) -> Any:
            return None

        @property
        def message_history(self) -> list[Any]:
            return []

        def load_message_history(self, messages: list[Any]) -> None:
            del messages

        async def reset_events(self) -> None:
            self.events_reset = True

    agent = _ResettableAgent()
    cleared = await clear_agent_histories({"main": agent})

    assert cleared == ["main"]
    assert agent.cleared is True
    assert agent.events_reset is True
