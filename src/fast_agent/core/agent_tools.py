"""Helpers for wiring agents as tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def add_tools_for_agents(add_tool_fn: Callable[[Any], str], agents: Iterable[Any]) -> list[str]:
    """Register child agents as tools and return tool names."""
    added_tools: list[str] = []
    for agent in agents:
        if agent is None:
            continue
        tool_name = add_tool_fn(agent)
        added_tools.append(tool_name)
    return added_tools
