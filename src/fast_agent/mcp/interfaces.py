"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)

from fast_agent.interfaces import (
    AgentProtocol,
    FastAgentLLMProtocol,
    LlmAgentProtocol,
    LLMFactoryProtocol,
    ModelFactoryFunctionProtocol,
    ModelT,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mcp_types import (
        ServerCapabilities,
    )

    from fast_agent.config import MCPServerSettings
__all__ = [
    "AgentProtocol",
    "FastAgentLLMProtocol",
    "LLMFactoryProtocol",
    "LlmAgentProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
    "ServerRegistryProtocol",
]


@runtime_checkable
class ServerRegistryProtocol(Protocol):
    """Configuration and capability storage used by MCP clients."""

    @property
    def registry(self) -> dict[str, "MCPServerSettings"]: ...

    @property
    def active_home(self) -> "str | Path | None": ...

    @property
    def no_home(self) -> bool: ...

    def get_server_config(self, server_name: str) -> "MCPServerSettings | None": ...

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server."""
        ...

    def set_server_capabilities(
        self, server_name: str, capabilities: "ServerCapabilities"
    ) -> None: ...
