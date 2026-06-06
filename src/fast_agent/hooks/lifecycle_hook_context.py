"""Lifecycle hook context passed to agent lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.context import Context
    from fast_agent.hooks.hook_context import HookAgentProtocol
    from fast_agent.hooks.lifecycle_hook_types import LifecycleHookType
    from fast_agent.interfaces import AgentProtocol


@dataclass
class AgentLifecycleContext:
    agent: "HookAgentProtocol"
    context: Context | None
    config: AgentConfig
    hook_type: LifecycleHookType

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def has_context(self) -> bool:
        return self.context is not None

    @property
    def agent_registry(self) -> "Mapping[str, AgentProtocol] | None":
        """Return the active agent registry when configured."""
        return self.agent.agent_registry

    def get_agent(self, name: str) -> "AgentProtocol | None":
        """Lookup another agent by name when a registry is available."""
        return self.agent.get_agent(name)
