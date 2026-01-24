"""Lifecycle hook context passed to agent lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
