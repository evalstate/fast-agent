from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fast_agent.interfaces import AgentProtocol

T = TypeVar("T")


def _never_tool_only(name: str, agent: T) -> bool:
    del name, agent
    return False


def agent_is_default(agent: "AgentProtocol") -> bool:
    return bool(agent.config.default)


def resolve_default_agent_name(
    agents: Mapping[str, T],
    *,
    is_default: Callable[[str, T], bool],
    is_tool_only: Callable[[str, T], bool] = _never_tool_only,
) -> str | None:
    for name, agent in agents.items():
        if is_tool_only(name, agent):
            continue
        if is_default(name, agent):
            return name

    for name, agent in agents.items():
        if not is_tool_only(name, agent):
            return name

    return next(iter(agents), None)
