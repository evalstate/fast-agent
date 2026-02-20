"""Helpers for building normalized progress payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fast_agent.event_progress import ProgressAction


def build_progress_payload(
    *,
    action: "ProgressAction",
    agent_name: str | None = None,
    tool_name: str | None = None,
    server_name: str | None = None,
    tool_use_id: str | None = None,
    tool_call_id: str | None = None,
    tool_event: str | None = None,
    progress: float | None = None,
    total: float | None = None,
    details: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a normalized payload for progress logger events."""
    payload: dict[str, Any] = {"progress_action": action}

    if agent_name is not None:
        payload["agent_name"] = agent_name
    if tool_name is not None:
        payload["tool_name"] = tool_name
    if server_name is not None:
        payload["server_name"] = server_name
    if tool_use_id is not None:
        payload["tool_use_id"] = tool_use_id
    if tool_call_id is not None:
        payload["tool_call_id"] = tool_call_id
    if tool_event is not None:
        payload["tool_event"] = tool_event
    if progress is not None:
        payload["progress"] = progress
    if total is not None:
        payload["total"] = total
    if details is not None:
        payload["details"] = details

    if extra:
        payload.update(extra)

    return payload
