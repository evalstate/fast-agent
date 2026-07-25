"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from contextlib import AbstractAsyncContextManager
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
    from mcp_types import (
        ServerCapabilities,
    )

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.client_connection import MCPClientConnection

__all__ = [
    "AgentProtocol",
    "FastAgentLLMProtocol",
    "LLMFactoryProtocol",
    "LlmAgentProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
    "ServerInitializerProtocol",
    "ServerRegistryProtocol",
]


@runtime_checkable
class ServerInitializerProtocol(Protocol):
    """Protocol for on-demand server clients used by gen_client."""

    def initialize_server(
        self,
        server_name: str,
        callback_runtime: "MCPClientCallbackRuntime | None" = None,
        trigger_oauth: bool | None = None,
    ) -> AbstractAsyncContextManager["MCPClientConnection"]:
        """Initialize a server and yield a client connection."""
        ...

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server, or None if not yet initialized."""
        ...


@runtime_checkable
class ServerRegistryProtocol(ServerInitializerProtocol, Protocol):
    """Protocol defining the minimal interface of ServerRegistry needed by gen_client."""

    @property
    def registry(self) -> dict[str, "MCPServerSettings"]: ...

    def get_server_config(self, server_name: str) -> "MCPServerSettings | None": ...
