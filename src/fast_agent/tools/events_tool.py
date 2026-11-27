"""Model-controlled built-in tool for MCP Events (SEP-3415) subscriptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol

from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.mcp.events_subscriptions import format_poll_interval, parse_poll_interval_ms

if TYPE_CHECKING:
    from fast_agent.mcp.events_extension import EventDescriptor
    from fast_agent.mcp.events_subscriptions import EventSubscription, EventSubscriptionManager

EVENTS_TOOL_NAME: Final = "events"
EVENTS_TOOL_METADATA: Final[dict[str, Any]] = {"fast_agent": {"builtin": "harness"}}

_VALID_ACTIONS: Final = frozenset({"list", "status", "subscribe", "unsubscribe"})

_EVENTS_TOOL_DESCRIPTION: Final = (
    "Discover and manage MCP event subscriptions on attached servers that support the "
    "MCP Events extension.\n"
    "Actions:\n"
    "- `list`: list available event types (optionally filtered by `server`).\n"
    "- `status`: show active/completed/terminated subscriptions and whether wake turns are paused.\n"
    "- `subscribe`: subscribe to `(server, name)` with optional filter `arguments`, `wake` "
    "(defaults to `True`, starting a turn when an event arrives while idle; set `False` for "
    "passive inbox-only buffering), `once=True` (stop after the first event), and `every` "
    "(minimum poll interval such as `'30s'` or `'5m'`).\n"
    "- `unsubscribe`: stop a subscription by `id`.\n"
    'Subscribed events arrive later as `<mcp-event subscription="...">` blocks in a user message. '
    "Unsubscribe when you no longer need updates (for example, after a watched job finishes)."
)

_EVENTS_INPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "status", "subscribe", "unsubscribe"],
            "description": (
                "Action to perform: 'list' event types, 'status' of subscriptions, "
                "'subscribe' to an event type, or 'unsubscribe' by subscription id."
            ),
        },
        "server": {
            "type": "string",
            "description": "MCP server name (optional for 'list', required for 'subscribe').",
        },
        "name": {
            "type": "string",
            "description": "Event type name (required for 'subscribe').",
        },
        "arguments": {
            "type": "object",
            "description": "Optional subscription filter arguments.",
        },
        "wake": {
            "type": "boolean",
            "default": True,
            "description": (
                "Start a turn when an event arrives while idle "
                "(default: true; set false for passive inbox-only buffering)."
            ),
        },
        "once": {
            "type": "boolean",
            "default": False,
            "description": "Stop polling automatically after the first event arrives.",
        },
        "every": {
            "type": "string",
            "description": "Optional minimum poll interval (e.g. '30s', '5m').",
        },
        "id": {
            "type": "string",
            "description": "Subscription id to cancel (required for 'unsubscribe').",
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


class EventsAggregatorProtocol(Protocol):
    """Minimal aggregator surface required by ``execute_events_tool``."""

    server_names: list[str]

    @property
    def events(self) -> EventSubscriptionManager: ...

    async def supports_events(self, server_name: str) -> bool: ...


def build_events_tool() -> Tool:
    """Build the static, prompt-cache-friendly built-in ``events`` MCP tool."""
    return Tool(
        name=EVENTS_TOOL_NAME,
        description=_EVENTS_TOOL_DESCRIPTION,
        input_schema=_EVENTS_INPUT_SCHEMA,
        meta=EVENTS_TOOL_METADATA,
    )


def _ok_result(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, sort_keys=True))],
        structured_content=payload,
        is_error=False,
    )


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=True,
    )


def _required_arguments(descriptor: EventDescriptor) -> list[str]:
    schema = descriptor.input_schema
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [item for item in required if isinstance(item, str)]


def _serialize_descriptor(server: str, descriptor: EventDescriptor) -> dict[str, Any]:
    return {
        "server": server,
        "name": descriptor.name,
        "description": descriptor.description,
        "delivery": list(descriptor.delivery),
        "required_arguments": _required_arguments(descriptor),
        "input_schema": descriptor.input_schema,
    }


def _serialize_subscription(
    subscription: EventSubscription,
    *,
    server_display_name: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": subscription.key,
        "server": server_display_name or subscription.server_name,
        "name": subscription.name,
        "arguments": dict(subscription.arguments),
        "wake": subscription.wake,
        "once": subscription.once,
        "every": (
            format_poll_interval(subscription.min_poll_ms)
            if subscription.min_poll_ms is not None
            else None
        ),
        "origin": subscription.origin,
        "status": subscription.status,
        "delivered": subscription.delivered,
    }
    if subscription.last_error is not None:
        entry["last_error"] = subscription.last_error
    return entry


def _require_non_empty_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Parameter '{field}' is required and must be a non-empty string.")
    return value.strip()


def _optional_bool(payload: Mapping[str, Any], field: str, *, default: bool = False) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"Parameter '{field}' must be a boolean.")
    return value


async def execute_events_tool(
    aggregator: EventsAggregatorProtocol,
    arguments: Mapping[str, Any] | None = None,
) -> CallToolResult:
    """Execute the built-in ``events`` tool against ``aggregator``."""
    if arguments is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(arguments, Mapping):
        payload = arguments
    else:
        return _error_result("Arguments must be an object.")

    action = payload.get("action")
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        return _error_result(
            "Parameter 'action' is required and must be one of: list, status, subscribe, unsubscribe."
        )

    try:
        if action == "list":
            return await _handle_list(aggregator, payload)
        if action == "status":
            return _handle_status(aggregator)
        if action == "subscribe":
            return await _handle_subscribe(aggregator, payload)
        return await _handle_unsubscribe(aggregator, payload)
    except (ValueError, LookupError, MCPError) as exc:
        return _error_result(str(exc) or type(exc).__name__)


async def _handle_list(
    aggregator: EventsAggregatorProtocol,
    payload: Mapping[str, Any],
) -> CallToolResult:
    raw_server = payload.get("server")
    if raw_server is not None:
        if not isinstance(raw_server, str) or not raw_server.strip():
            raise ValueError("Parameter 'server' must be a non-empty string when provided.")
        server = raw_server.strip()
        if not await aggregator.supports_events(server):
            raise ValueError(f"Server '{server}' does not support MCP events.")
        target_servers = [server]
    else:
        target_servers = [
            name for name in aggregator.server_names if await aggregator.supports_events(name)
        ]

    items: list[dict[str, Any]] = []
    for server_name in target_servers:
        descriptors = await aggregator.events.list_descriptors(server_name, refresh=True)
        display_name = aggregator.events.display_server_name(server_name)
        items.extend(_serialize_descriptor(display_name, descriptor) for descriptor in descriptors)
    return _ok_result({"events": items})


def _handle_status(aggregator: EventsAggregatorProtocol) -> CallToolResult:
    manager = aggregator.events
    subscriptions = [
        _serialize_subscription(
            sub,
            server_display_name=manager.display_server_name(sub.server_name),
        )
        for sub in manager.subscriptions()
    ]
    return _ok_result(
        {
            "subscriptions": subscriptions,
            "wake_paused": manager.wake_paused,
        }
    )


async def _handle_subscribe(
    aggregator: EventsAggregatorProtocol,
    payload: Mapping[str, Any],
) -> CallToolResult:
    server = _require_non_empty_str(payload, "server")
    name = _require_non_empty_str(payload, "name")
    raw_args = payload.get("arguments")
    if raw_args is None:
        sub_args: dict[str, Any] = {}
    elif isinstance(raw_args, Mapping):
        sub_args = dict(raw_args)
    else:
        raise ValueError("Parameter 'arguments' must be an object.")

    wake = _optional_bool(payload, "wake", default=True)
    once = _optional_bool(payload, "once", default=False)

    raw_every = payload.get("every")
    min_poll_ms: int | None = None
    if raw_every is not None:
        if not isinstance(raw_every, str) or not raw_every.strip():
            raise ValueError("Parameter 'every' must be a duration string like '30s' or '5m'.")
        min_poll_ms = parse_poll_interval_ms(raw_every)

    subscription = await aggregator.events.subscribe(
        server,
        name,
        sub_args,
        wake=wake,
        once=once,
        min_poll_ms=min_poll_ms,
        origin="model",
    )
    return _ok_result(
        {
            "id": subscription.key,
            "server": aggregator.events.display_server_name(subscription.server_name),
            "name": subscription.name,
            "wake": subscription.wake,
            "once": subscription.once,
            "every": (
                format_poll_interval(subscription.min_poll_ms)
                if subscription.min_poll_ms is not None
                else None
            ),
            "status": subscription.status,
        }
    )


async def _handle_unsubscribe(
    aggregator: EventsAggregatorProtocol,
    payload: Mapping[str, Any],
) -> CallToolResult:
    sub_id = _require_non_empty_str(payload, "id")
    removed = await aggregator.events.unsubscribe(sub_id, origin="model")
    if not removed:
        raise LookupError(f"No subscription found with id '{sub_id}'.")
    return _ok_result({"id": sub_id, "unsubscribed": True})
