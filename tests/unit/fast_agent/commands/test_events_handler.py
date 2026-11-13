from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.shared.exceptions import MCPError

from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.events import handle_events, parse_event_arguments
from fast_agent.mcp.events_extension import (
    EventDescriptor,
    EventOccurrence,
    ListEventsResult,
    PollEventsResult,
)
from fast_agent.mcp.events_subscriptions import EventSubscriptionManager, subscription_key
from fast_agent.ui.prompt.parser import parse_special_input

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage, CommandOutcome


class _Aggregator:
    """Offer one poll event type on ``jobs`` and none on ``plain``."""

    def __init__(self) -> None:
        self.server_names = ["jobs", "plain"]
        self.polls: list[tuple[str, str, dict[str, Any], str | None]] = []
        self.next_events: list[EventOccurrence] = []
        self.events = EventSubscriptionManager(self, sleep=self._sleep)

    @staticmethod
    async def _sleep(seconds: float) -> None:
        del seconds
        await asyncio.sleep(0)

    async def supports_events(self, server_name: str) -> bool:
        return server_name == "jobs"

    async def list_events(self, server_name: str, cursor: str | None = None) -> ListEventsResult:
        del server_name, cursor
        return ListEventsResult(
            events=[
                EventDescriptor(
                    name="job.status_changed",
                    description="Fires when a job changes stage",
                    delivery=["poll"],
                    input_schema={
                        "type": "object",
                        "properties": {"job_id": {}, "verbose": {}},
                        "required": ["job_id"],
                    },
                ),
                EventDescriptor(name="job.log_line", delivery=["push"]),
            ]
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
        del max_age_ms, max_events
        arguments = arguments or {}
        if "job_id" not in arguments:
            raise MCPError(-32602, "job_id is required")
        if cursor is not None and not self.next_events:
            await asyncio.Event().wait()
        self.polls.append((server_name, name, arguments, cursor))
        events, self.next_events = self.next_events, []
        return PollEventsResult(events=events, cursor=f"c{len(self.polls)}")


class _Agent:
    def __init__(self) -> None:
        self.aggregator = _Aggregator()


class _IO(NonInteractiveCommandIOBase):
    async def emit(self, message: CommandMessage) -> None:
        del message


def _context(agent: object) -> CommandContext:
    return CommandContext(
        agent_provider=StaticAgentProvider({"main": agent}),
        current_agent_name="main",
        io=_IO(),
        no_home=True,
    )


def _text(outcome: CommandOutcome) -> str:
    assert len(outcome.messages) == 1
    return outcome.messages[0].plain_text()


@pytest.mark.parametrize(
    ("command", "argument"),
    [
        ("/events", None),
        ("/events list jobs", "list jobs"),
        (
            "/events subscribe jobs job.status_changed job_id=abc",
            "subscribe jobs job.status_changed job_id=abc",
        ),
    ],
)
def test_parse_special_input_events_command(command: str, argument: str | None) -> None:
    from fast_agent.ui.command_payloads import EventsCommand

    actual = parse_special_input(command)

    assert isinstance(actual, EventsCommand)
    assert actual.argument == argument


def test_parse_event_arguments() -> None:
    assert parse_event_arguments([]) == {}
    assert parse_event_arguments(["job_id=abc", "limit=5", "redact=true", "q=a=b"]) == {
        "job_id": "abc",
        "limit": 5,
        "redact": True,
        "q": "a=b",
    }
    assert parse_event_arguments(['{"job_id": "abc", "tags": ["x"]}']) == {
        "job_id": "abc",
        "tags": ["x"],
    }
    with pytest.raises(ValueError, match="key=value"):
        parse_event_arguments(["job_id"])
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_event_arguments(['{"job_id": }'])


@pytest.mark.asyncio
async def test_events_list_shows_offered_types_and_arguments() -> None:
    context = _context(_Agent())

    text = _text(await handle_events(context, agent_name="main", argument="list"))

    assert "## jobs" in text
    assert "plain" not in text
    assert "`job.status_changed` (poll): Fires when a job changes stage" in text
    assert "arguments: `job_id` (required), `verbose`" in text
    assert "`job.log_line` (push)" in text
    assert "Not subscribable here" in text

    outcome = await handle_events(context, agent_name="main", argument="list plain")
    assert outcome.messages[0].channel == "error"
    assert "does not support the events extension" in _text(outcome)


@pytest.mark.asyncio
async def test_events_subscribe_status_and_unsubscribe() -> None:
    agent = _Agent()
    context = _context(agent)
    manager = agent.aggregator.events

    empty = _text(await handle_events(context, agent_name="main"))
    assert "No event subscriptions" in empty

    key = subscription_key("jobs", "job.status_changed", {"job_id": "abc"})
    subscribed = _text(
        await handle_events(
            context,
            agent_name="main",
            argument="subscribe jobs job.status_changed job_id=abc",
        )
    )
    assert f"as `{key}`" in subscribed
    assert agent.aggregator.polls == [("jobs", "job.status_changed", {"job_id": "abc"}, None)]

    agent.aggregator.next_events = [
        EventOccurrence(
            event_id="abc:COMPLETED",
            name="job.status_changed",
            timestamp="2026-11-12T10:00:00Z",
            data={"stage": "COMPLETED"},
        )
    ]
    while not manager.pending_count:
        await asyncio.sleep(0)

    status = _text(await handle_events(context, agent_name="main", argument="status"))
    assert f"`{key}`" in status
    assert "| jobs | job.status_changed |" in status
    assert "| active | 1 |" in status
    assert "1 waiting for the next model call." in status

    removed = _text(await handle_events(context, agent_name="main", argument=f"unsubscribe {key}"))
    assert f"Removed subscription `{key}`" in removed
    assert manager.subscriptions() == []

    missing = await handle_events(context, agent_name="main", argument=f"unsubscribe {key}")
    assert missing.messages[0].channel == "error"
    await manager.aclose()


@pytest.mark.asyncio
async def test_events_subscribe_with_wake_and_poll() -> None:
    agent = _Agent()
    context = _context(agent)
    manager = agent.aggregator.events

    subscribed = _text(
        await handle_events(
            context,
            agent_name="main",
            argument="subscribe jobs job.status_changed --wake job_id=abc",
        )
    )
    assert "start a turn" in subscribed
    [subscription] = manager.subscriptions()
    assert subscription.wake
    assert subscription.arguments == {"job_id": "abc"}
    assert "| yes | active |" in _text(await handle_events(context, agent_name="main"))

    polled = _text(
        await handle_events(context, agent_name="main", argument=f"poll {subscription.key}")
    )
    assert f"Polling `{subscription.key}` now" in polled

    missing = await handle_events(context, agent_name="main", argument="poll nope")
    assert missing.messages[0].channel == "error"
    await manager.aclose()


@pytest.mark.asyncio
async def test_events_subscribe_reports_rejections_and_bad_usage() -> None:
    agent = _Agent()
    context = _context(agent)

    for argument, expected in [
        ("subscribe jobs job.deleted", "does not offer event"),
        ("subscribe jobs job.log_line", "does not support poll delivery"),
        ("subscribe jobs job.status_changed", "job_id is required"),
        ("subscribe jobs job.status_changed job_id", "Expected key=value"),
        ("subscribe jobs", "Usage: /events"),
        ("frobnicate", "Usage: /events"),
    ]:
        outcome = await handle_events(context, agent_name="main", argument=argument)
        assert outcome.messages[0].channel == "error", argument
        assert expected in _text(outcome), argument

    assert agent.aggregator.events.subscriptions() == []


@pytest.mark.asyncio
async def test_events_requires_an_mcp_agent() -> None:
    outcome = await handle_events(_context(object()), agent_name="main")

    assert outcome.messages[0].channel == "warning"
    assert "not available for this agent" in _text(outcome)
