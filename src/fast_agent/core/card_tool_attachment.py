"""Shared registry-level AgentCard tool attachment helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.default_agent import resolve_default_agent_name

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.core.agent_card_types import AgentCardData

_CARD_TOOL_ATTACHABLE_TYPES = frozenset({"basic", "smart", "custom"})


class CardToolAttachTarget(Protocol):
    agents: dict[str, AgentCardData]

    def load_agents(self, path: str | Path) -> list[str]: ...

    def attach_agent_tools(self, parent_name: str, child_names: list[str]) -> list[str]: ...


def resolve_card_tool_target(
    agents: dict[str, AgentCardData],
    preferred_agent_names: list[str | None],
) -> str | None:
    for agent_name in preferred_agent_names:
        if agent_name and _agent_supports_card_tools(agents, agent_name):
            return agent_name

    return resolve_default_agent_name(
        agents,
        is_default=lambda _name, agent_data: (
            isinstance(config_obj := agent_data.get("config"), AgentConfig) and config_obj.default
        ),
        is_tool_only=lambda _name, agent_data: (
            bool(agent_data.get("tool_only", False))
            or str(agent_data.get("type")) not in _CARD_TOOL_ATTACHABLE_TYPES
        ),
    )


def _agent_supports_card_tools(agents: dict[str, AgentCardData], agent_name: str) -> bool:
    agent_data = agents.get(agent_name)
    return agent_data is not None and str(agent_data.get("type")) in _CARD_TOOL_ATTACHABLE_TYPES


def attach_card_tool_agents(
    fast: CardToolAttachTarget,
    child_agent_names: list[str],
    *,
    preferred_agent_names: list[str | None],
) -> list[str]:
    if not child_agent_names:
        return []

    target_name = resolve_card_tool_target(fast.agents, preferred_agent_names)
    if target_name is None:
        return []

    return fast.attach_agent_tools(target_name, child_agent_names)


def load_and_attach_card_tool_agents(
    fast: CardToolAttachTarget,
    card_sources: list[str] | None,
    *,
    preferred_agent_names: list[str | None],
) -> list[str]:
    loaded_names: list[str] = []
    for card_source in card_sources or []:
        loaded_names.extend(fast.load_agents(card_source))

    attach_card_tool_agents(
        fast,
        loaded_names,
        preferred_agent_names=preferred_agent_names,
    )
    return loaded_names
