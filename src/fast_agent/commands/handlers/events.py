"""Shared /events command handler for MCP Events (SEP-3415) subscriptions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from mcp.shared.exceptions import MCPError

from fast_agent.commands.results import CommandOutcome
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.events_subscriptions import format_poll_interval, parse_poll_interval_ms
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import escape_markdown_table_cell, markdown_code_span
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.events_extension import EventDescriptor
    from fast_agent.mcp.events_subscriptions import EventSubscriptionManager

logger = get_logger(__name__)

EVENTS_USAGE = (
    "Usage: /events [select|list [server]|subscribe <server> <event> [key=value ...] "
    "[--no-wake] [--once] [--every <duration>]|poll <id>|unsubscribe <id>|unsubscribe all]"
)
WAKE_FLAG = "--wake"
NO_WAKE_FLAG = "--no-wake"
ONCE_FLAG = "--once"
EVERY_FLAG = "--every"


@dataclass(frozen=True, slots=True)
class _SubscribeOptions:
    argument_tokens: list[str]
    wake: bool = True
    once: bool = False
    min_poll_ms: int | None = None


def _parse_subscribe_options(tokens: list[str]) -> _SubscribeOptions:
    argument_tokens: list[str] = []
    wake = True
    once = False
    min_poll_ms: int | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == WAKE_FLAG:
            wake = True
        elif token == NO_WAKE_FLAG:
            wake = False
        elif token == ONCE_FLAG:
            once = True
        elif token == EVERY_FLAG:
            index += 1
            if index >= len(tokens):
                raise ValueError("Expected a duration after '--every' (e.g. --every 5m)")
            min_poll_ms = parse_poll_interval_ms(tokens[index])
        elif token.startswith(f"{EVERY_FLAG}="):
            min_poll_ms = parse_poll_interval_ms(token.removeprefix(f"{EVERY_FLAG}="))
        else:
            argument_tokens.append(token)
        index += 1
    return _SubscribeOptions(
        argument_tokens=argument_tokens,
        wake=wake,
        once=once,
        min_poll_ms=min_poll_ms,
    )


class _EventsAggregator(Protocol):
    server_names: list[str]

    @property
    def events(self) -> EventSubscriptionManager: ...

    async def supports_events(self, server_name: str) -> bool: ...


@runtime_checkable
class _EventsAgent(Protocol):
    @property
    def aggregator(self) -> _EventsAggregator: ...


@dataclass(frozen=True, slots=True)
class EventsPickerResult:
    """Outcome returned when the interactive events picker closes."""

    buffer_prefill: str | None = None
    message: str | None = None


@runtime_checkable
class EventsPickerIO(Protocol):
    """Command IO that can run the interactive MCP events picker."""

    async def pick_events(
        self,
        *,
        manager: EventSubscriptionManager,
        descriptors_by_server: Mapping[str, Sequence[EventDescriptor]],
    ) -> EventsPickerResult | None:
        """Return the picker outcome, or None when cancelled."""
        ...


def parse_event_arguments(tokens: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` tokens, or one JSON object, into subscription arguments.

    Values that parse as JSON keep their type, so ``limit=5`` is a number and
    ``redact=true`` a boolean. Anything else is a string.
    """
    if len(tokens) == 1 and tokens[0].lstrip().startswith("{"):
        try:
            parsed = json.loads(tokens[0])
        except ValueError as exc:
            raise ValueError(f"Arguments are not valid JSON: {exc}") from exc
        return parsed

    arguments: dict[str, Any] = {}
    for token in tokens:
        key, separator, raw = token.partition("=")
        if not separator or not key:
            raise ValueError(f"Expected key=value, got '{token}'")
        try:
            arguments[key] = json.loads(raw)
        except ValueError:
            arguments[key] = raw
    return arguments


async def handle_events(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, _EventsAgent):
        outcome.add_message(
            "MCP events are not available for this agent.",
            channel="warning",
            right_info="events",
            agent_name=agent_name,
        )
        return outcome
    aggregator = agent.aggregator

    try:
        tokens = split_commandline(argument or "", syntax="posix")
    except ValueError:
        outcome.add_message(EVENTS_USAGE, channel="error", right_info="events")
        return outcome

    action = strip_casefold(tokens[0]) if tokens else "status"
    rest = tokens[1:]
    try:
        if action == "status" and not rest:
            _add_markdown(outcome, _render_status(aggregator), "events")
        elif action == "select" and not rest:
            if await _handle_select(ctx, outcome, aggregator):
                await _persist_session_events(
                    ctx,
                    cast("AgentProtocol", agent),
                    create_if_missing=bool(aggregator.events.persistable_subscriptions()),
                )
        elif action == "list" and len(rest) <= 1:
            text = await _render_event_types(aggregator, rest[0] if rest else None)
            _add_markdown(outcome, text, "events list")
        elif action == "subscribe" and len(rest) >= 2:
            options = _parse_subscribe_options(rest[2:])
            subscription = await aggregator.events.subscribe(
                rest[0],
                rest[1],
                parse_event_arguments(options.argument_tokens),
                wake=options.wake,
                once=options.once,
                min_poll_ms=options.min_poll_ms,
            )
            await _persist_session_events(ctx, cast("AgentProtocol", agent), create_if_missing=True)
            delivery = (
                "Events start a turn when the prompt is idle."
                if options.wake
                else "Events are added to the next model call."
            )
            extras: list[str] = []
            if options.once:
                extras.append("Stops after the first event.")
            if options.min_poll_ms is not None:
                extras.append(
                    f"Minimum poll interval: {format_poll_interval(options.min_poll_ms)}."
                )
            extra_text = (" " + " ".join(extras)) if extras else ""
            display_server = aggregator.events.display_server_name(subscription.server_name)
            _add_markdown(
                outcome,
                f"Subscribed to `{subscription.name}` on `{display_server}` "
                f"as `{subscription.key}`. {delivery}{extra_text}",
                "events subscribe",
            )
        elif action == "poll" and len(rest) == 1:
            if not aggregator.events.poll_now(rest[0]):
                raise ValueError(f"No active subscription with id '{rest[0]}'")
            _add_markdown(outcome, f"Polling `{rest[0]}` now.", "events poll")
        elif action == "unsubscribe" and len(rest) == 1:
            message = await _unsubscribe(aggregator, rest[0])
            await _persist_session_events(ctx, cast("AgentProtocol", agent))
            _add_markdown(outcome, message, "events unsubscribe")
        else:
            outcome.add_message(EVENTS_USAGE, channel="error", right_info="events")
    except (ValueError, MCPError) as exc:
        outcome.add_message(str(exc), channel="error", right_info=f"events {action}")
    return outcome


def _add_markdown(outcome: CommandOutcome, text: str, right_info: str) -> None:
    outcome.add_message(text, right_info=right_info, render_markdown=True)


def _active_subscriptions_signature(
    manager: EventSubscriptionManager,
) -> tuple[tuple[str, bool, bool, int | None, str], ...]:
    return tuple(
        (sub.key, sub.wake, sub.once, sub.min_poll_ms, sub.origin)
        for sub in manager.subscriptions()
        if sub.status == "active"
    )


async def _persist_session_events(
    ctx: CommandContext,
    agent: AgentProtocol,
    *,
    create_if_missing: bool = False,
) -> None:
    if not ctx.sessions_enabled:
        return
    try:
        manager = ctx.resolve_session_manager()
        session = manager.current_session
        if session is None:
            if not create_if_missing:
                return
            session = manager.create_session()
        await session.save_history(agent, checkpoint=True)
        from fast_agent.integrations.herdr_session import report_current_session

        report_current_session(manager, agent=agent)
    except Exception:
        logger.warning("Failed to persist session event subscriptions", exc_info=True)


async def _handle_select(
    ctx: CommandContext,
    outcome: CommandOutcome,
    aggregator: _EventsAggregator,
) -> bool:
    if not isinstance(ctx.io, EventsPickerIO):
        status = _render_status(aggregator)
        _add_markdown(outcome, f"{status}\n\n{EVENTS_USAGE}", "events select")
        return False

    descriptors_by_server: dict[str, list[EventDescriptor]] = {}
    for name in aggregator.server_names:
        if await aggregator.supports_events(name):
            display_name = aggregator.events.display_server_name(name)
            descriptors_by_server[display_name] = await aggregator.events.list_descriptors(
                name, refresh=True
            )

    if not descriptors_by_server and not aggregator.events.subscriptions():
        _add_markdown(
            outcome,
            "No attached server supports the MCP events extension.",
            "events select",
        )
        return False

    before = _active_subscriptions_signature(aggregator.events)
    result = await ctx.io.pick_events(
        manager=aggregator.events,
        descriptors_by_server=descriptors_by_server,
    )
    mutated = _active_subscriptions_signature(aggregator.events) != before
    if result is None:
        return mutated
    if result.buffer_prefill:
        outcome.buffer_prefill = result.buffer_prefill
    if result.message:
        _add_markdown(outcome, result.message, "events select")
    return mutated


def _render_status(aggregator: _EventsAggregator) -> str:
    manager = aggregator.events
    subscriptions = manager.subscriptions()
    if not subscriptions:
        return "No event subscriptions. Use `/events list` to see what servers offer."

    lines = [
        "| id | server | event | arguments | wake | origin | poll | status | delivered |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for subscription in subscriptions:
        status = subscription.status
        if subscription.last_error:
            status = f"{status}: {subscription.last_error}"
        arguments = json.dumps(subscription.arguments, sort_keys=True)
        if subscription.poll_interval_ms is not None:
            poll_text = format_poll_interval(subscription.poll_interval_ms)
            if (
                subscription.min_poll_ms is not None
                and subscription.min_poll_ms >= subscription.poll_interval_ms
            ):
                poll_text = f"{poll_text} (min)"
        elif subscription.min_poll_ms is not None:
            poll_text = f"{format_poll_interval(subscription.min_poll_ms)} (min)"
        else:
            poll_text = "-"
        display_server = manager.display_server_name(subscription.server_name)
        lines.append(
            f"| `{subscription.key}` "
            f"| {escape_markdown_table_cell(display_server)} "
            f"| {escape_markdown_table_cell(subscription.name)} "
            f"| {escape_markdown_table_cell(markdown_code_span(arguments))} "
            f"| {'yes' if subscription.wake else 'no'} "
            f"| {escape_markdown_table_cell(subscription.origin)} "
            f"| {escape_markdown_table_cell(poll_text)} "
            f"| {escape_markdown_table_cell(status)} "
            f"| {subscription.delivered} |"
        )
    if manager.pending_count:
        lines.extend(["", f"{manager.pending_count} waiting for the next model call."])
    lines.extend(
        [
            "",
            "Use `/events select`, `/events poll <id>`, or `/events unsubscribe <id>`.",
        ]
    )
    return "\n".join(lines)


async def _render_event_types(aggregator: _EventsAggregator, server_name: str | None) -> str:
    if server_name is not None:
        if not await aggregator.supports_events(server_name):
            raise ValueError(f"Server '{server_name}' does not support the events extension")
        server_names = [server_name]
    else:
        server_names = [
            name for name in aggregator.server_names if await aggregator.supports_events(name)
        ]
        if not server_names:
            return "No attached server supports the MCP events extension."

    sections: list[str] = []
    for name in server_names:
        descriptors = await aggregator.events.list_descriptors(name, refresh=True)
        display_name = aggregator.events.display_server_name(name)
        lines = [f"## {display_name}", ""]
        if not descriptors:
            lines.append("No event types offered.")
        for descriptor in descriptors:
            lines.append(_render_descriptor(descriptor))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _render_descriptor(descriptor: EventDescriptor) -> str:
    delivery = ", ".join(descriptor.delivery)
    line = f"- `{descriptor.name}` ({delivery})"
    if descriptor.description:
        line += f": {descriptor.description}"
    if not descriptor.supports("poll"):
        line += " Not subscribable here: poll delivery only."

    schema = descriptor.input_schema or {}
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        required = schema.get("required")
        required_names = set(required) if isinstance(required, list) else set()
        names = [
            f"`{key}`" + (" (required)" if key in required_names else "") for key in properties
        ]
        line += f"\n  - arguments: {', '.join(names)}"
    return line


async def _unsubscribe(aggregator: _EventsAggregator, target: str) -> str:
    manager = aggregator.events
    if strip_casefold(target) == "all":
        keys = [subscription.key for subscription in manager.subscriptions()]
        for key in keys:
            await manager.unsubscribe(key)
        return f"Removed {len(keys)} subscription(s)."
    if not await manager.unsubscribe(target):
        raise ValueError(f"No subscription with id '{target}'")
    return f"Removed subscription `{target}`."
