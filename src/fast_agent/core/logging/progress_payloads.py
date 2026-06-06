"""Helpers for building normalized progress payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fast_agent.event_progress import ProgressAction


_PROGRESS_PAYLOAD_FIELDS = (
    "agent_name",
    "tool_name",
    "server_name",
    "tool_use_id",
    "tool_call_id",
    "tool_event",
    "tool_state",
    "tool_terminal",
    "progress",
    "total",
    "details",
)


def build_progress_payload(
    *,
    action: "ProgressAction",
    agent_name: str | None = None,
    tool_name: str | None = None,
    server_name: str | None = None,
    tool_use_id: str | None = None,
    tool_call_id: str | None = None,
    tool_event: str | None = None,
    tool_state: str | None = None,
    tool_terminal: bool | None = None,
    progress: float | None = None,
    total: float | None = None,
    details: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a normalized payload for progress logger events."""
    values = locals()
    payload: dict[str, Any] = {
        "progress_action": action,
        **{field: values[field] for field in _PROGRESS_PAYLOAD_FIELDS if values[field] is not None},
    }

    if extra:
        payload.update(extra)

    return payload
