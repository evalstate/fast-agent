"""Shared /events command handler for MCP Events (SEP-3415) subscriptions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp.shared.exceptions import MCPError

from fast_agent.commands.results import CommandOutcome
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import escape_markdown_table_cell, markdown_code_span
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.mcp.events_extension import EventDescriptor
    from fast_agent.mcp.events_subscriptions import EventSubscriptionManager

EVENTS_USAGE = (
    "Usage: /events [list [server]|subscribe <server> <event> [key=value ...] [--wake]|"
    "poll <id>|unsubscribe <id>|unsubscribe all]"
)
WAKE_FLAG = "--wake"


class _EventsAggregator(Protocol):
    server_names: list[str]

    @property
    def events(self) -> EventSubscriptionManager: ...

    async def supports_events(self, server_name: str) -> bool: ...


@runtime_checkable
class _EventsAgent(Protocol):
    @property
    def aggregator(self) -> _EventsAggregator: ...


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
        elif action == "list" and len(rest) <= 1:
            text = await _render_event_types(aggregator, rest[0] if rest else None)
            _add_markdown(outcome, text, "events list")
        elif action == "subscribe" and len(rest) >= 2:
            wake = WAKE_FLAG in rest
            arguments = [token for token in rest[2:] if token != WAKE_FLAG]
            subscription = await aggregator.events.subscribe(
                rest[0], rest[1], parse_event_arguments(arguments), wake=wake
            )
            delivery = (
                "Events start a turn when the prompt is idle."
                if wake
                else "Events are added to the next model call."
            )
            _add_markdown(
                outcome,
                f"Subscribed to `{subscription.name}` on `{subscription.server_name}` "
                f"as `{subscription.key}`. {delivery}",
                "events subscribe",
            )
        elif action == "poll" and len(rest) == 1:
            if not aggregator.events.poll_now(rest[0]):
                raise ValueError(f"No active subscription with id '{rest[0]}'")
            _add_markdown(outcome, f"Polling `{rest[0]}` now.", "events poll")
        elif action == "unsubscribe" and len(rest) == 1:
            _add_markdown(outcome, await _unsubscribe(aggregator, rest[0]), "events unsubscribe")
        else:
            outcome.add_message(EVENTS_USAGE, channel="error", right_info="events")
    except (ValueError, MCPError) as exc:
        outcome.add_message(str(exc), channel="error", right_info=f"events {action}")
    return outcome


def _add_markdown(outcome: CommandOutcome, text: str, right_info: str) -> None:
    outcome.add_message(text, right_info=right_info, render_markdown=True)


def _render_status(aggregator: _EventsAggregator) -> str:
    manager = aggregator.events
    subscriptions = manager.subscriptions()
    if not subscriptions:
        return "No event subscriptions. Use `/events list` to see what servers offer."

    lines = [
        "| id | server | event | arguments | wake | status | delivered |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for subscription in subscriptions:
        status = subscription.status
        if subscription.last_error:
            status = f"{status}: {subscription.last_error}"
        arguments = json.dumps(subscription.arguments, sort_keys=True)
        lines.append(
            f"| `{subscription.key}` "
            f"| {escape_markdown_table_cell(subscription.server_name)} "
            f"| {escape_markdown_table_cell(subscription.name)} "
            f"| {escape_markdown_table_cell(markdown_code_span(arguments))} "
            f"| {'yes' if subscription.wake else 'no'} "
            f"| {escape_markdown_table_cell(status)} "
            f"| {subscription.delivered} |"
        )
    if manager.pending_count:
        lines.extend(["", f"{manager.pending_count} waiting for the next model call."])
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
        descriptors = await aggregator.events.list_event_types(name)
        lines = [f"## {name}", ""]
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
