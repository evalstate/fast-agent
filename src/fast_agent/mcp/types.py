from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.interfaces import AgentProtocol

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.mcp.mcp_aggregator import MCPAggregator
    from fast_agent.ui.console_display import ConsoleDisplay


@runtime_checkable
class McpAgentProtocol(AgentProtocol, Protocol):
    """Agent protocol with MCP-specific surface area."""

    @property
    def aggregator(self) -> MCPAggregator: ...

    @property
    def display(self) -> "ConsoleDisplay": ...

    @property
    def context(self) -> "Context | None": ...
