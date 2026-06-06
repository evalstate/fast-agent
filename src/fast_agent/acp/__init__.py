"""Agent Client Protocol (ACP) support for fast-agent."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from fast_agent.acp.acp_aware_mixin import ACPAwareMixin, ACPCommand, ACPModeInfo
from fast_agent.acp.acp_context import ACPContext, ClientCapabilities, ClientInfo
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from fast_agent.acp.server.agent_acp_server import AgentACPServer as AgentACPServer

__all__ = [
    "ACPAwareMixin",
    "ACPCommand",
    "ACPContext",
    "ACPFilesystemRuntime",
    "ACPModeInfo",
    "ACPTerminalRuntime",
    "AgentACPServer",
    "ClientCapabilities",
    "ClientInfo",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentACPServer": ("fast_agent.acp.server.agent_acp_server", "AgentACPServer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    return getattr(import_module(module_name), attr_name)
