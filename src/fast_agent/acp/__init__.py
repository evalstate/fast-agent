"""Agent Client Protocol (ACP) support for fast-agent."""

from typing import TYPE_CHECKING, Any

from fast_agent.acp.acp_aware_mixin import ACPAwareMixin, ACPCommand, ACPModeInfo
from fast_agent.acp.acp_context import ACPContext, ClientCapabilities, ClientInfo
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from fast_agent.acp.server.agent_acp_server import AgentACPServer as AgentACPServer

__all__ = [
    "ACPCommand",
    "ACPModeInfo",
    "ACPContext",
    "ACPAwareMixin",
    "ClientCapabilities",
    "ClientInfo",
    "AgentACPServer",
    "ACPFilesystemRuntime",
    "ACPTerminalRuntime",
]


def __getattr__(name: str) -> Any:
    if name == "AgentACPServer":
        from fast_agent.acp.server.agent_acp_server import AgentACPServer

        return AgentACPServer

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
