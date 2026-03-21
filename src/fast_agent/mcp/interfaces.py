"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from typing import (
    TYPE_CHECKING,
    AsyncContextManager,
    Callable,
    Protocol,
    runtime_checkable,
)

from mcp import ClientSession

from fast_agent.interfaces import (
    AgentProtocol,
    FastAgentLLMProtocol,
    LlmAgentProtocol,
    LLMFactoryProtocol,
    ModelFactoryFunctionProtocol,
    ModelT,
)

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings

__all__ = [
    "MCPConnectionManagerProtocol",
    "ServerInitializerProtocol",
    "ServerRegistryProtocol",
    "ServerConnection",
    "FastAgentLLMProtocol",
    "AgentProtocol",
    "LlmAgentProtocol",
    "LLMFactoryProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
]


@runtime_checkable
class MCPConnectionManagerProtocol(Protocol):
    """Protocol for MCPConnectionManager functionality needed by ServerRegistry."""

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> "ServerConnection": ...

    async def disconnect_server(self, server_name: str) -> None: ...

    async def disconnect_all_servers(self) -> None: ...


@runtime_checkable
class ServerInitializerProtocol(Protocol):
    """Protocol defining the interface required by temporary-session helpers."""

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> AsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...


@runtime_checkable
class ServerRegistryProtocol(Protocol):
    """Protocol defining the interface required by persistent-session helpers."""

    @property
    def registry(self) -> dict[str, "MCPServerSettings"]: ...

    @property
    def connection_manager(self) -> MCPConnectionManagerProtocol: ...

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> AsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...

    def get_server_config(self, server_name: str) -> "MCPServerSettings | None": ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...
