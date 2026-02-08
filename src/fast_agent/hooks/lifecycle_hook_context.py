"""Lifecycle hook context passed to agent lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.context import Context
    from fast_agent.interfaces import AgentProtocol


@dataclass
class AgentLifecycleContext:
    agent: AgentProtocol
    context: Context | None
    config: AgentConfig
    hook_type: Literal["on_start", "on_shutdown"]

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def has_context(self) -> bool:
        return self.context is not None

    @property
    def agent_registry(self) -> "Mapping[str, AgentProtocol] | None":
        """Return the active agent registry when configured."""
        return getattr(self.agent, "agent_registry", None)

    def get_agent(self, name: str) -> "AgentProtocol | None":
        """Lookup another agent by name when a registry is available."""
        getter = getattr(self.agent, "get_agent", None)
        if callable(getter):
            return getter(name)
        return None
