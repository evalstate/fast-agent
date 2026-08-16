"""Agent adapter for persisting an explicit message-history snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LlmAgentProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@runtime_checkable
class _AttachedMcpServerProvider(Protocol):
    def list_attached_mcp_servers(self) -> list[str]: ...


@runtime_checkable
class _AgentBackedToolProvider(Protocol):
    @property
    def agent_backed_tools(self) -> Mapping[str, "LlmAgentProtocol"]: ...


@dataclass
class HistoryAgent:
    """Delegate agent metadata while exposing an explicit history snapshot."""

    agent: AgentProtocol
    message_history: list[PromptMessageExtended]

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def config(self) -> AgentConfig:
        return self.agent.config

    @property
    def instruction(self) -> str:
        return self.agent.instruction

    @property
    def llm(self) -> FastAgentLLMProtocol | None:
        return self.agent.llm

    @property
    def usage_accumulator(self) -> UsageAccumulator | None:
        return self.agent.usage_accumulator

    def list_attached_mcp_servers(self) -> list[str]:
        if isinstance(self.agent, _AttachedMcpServerProvider):
            return self.agent.list_attached_mcp_servers()
        return []

    @property
    def agent_backed_tools(self) -> Mapping[str, LlmAgentProtocol]:
        if isinstance(self.agent, _AgentBackedToolProvider):
            return self.agent.agent_backed_tools
        return {}
