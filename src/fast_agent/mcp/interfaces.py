"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
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
    from mcp.types import ServerCapabilities

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

__all__ = [
    "ClientSessionFactory",
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
class ClientSessionFactory(Protocol):
    """Protocol for creating client sessions across persistent and temporary connections."""

    def __call__(
        self,
        read_stream: MemoryObjectReceiveStream,
        write_stream: MemoryObjectSendStream,
        read_timeout: timedelta | None,
        *,
        server_config: "MCPServerSettings | None" = None,
        transport_metrics: "TransportChannelMetrics | None" = None,
    ) -> ClientSession: ...


@runtime_checkable
class MCPConnectionManagerProtocol(Protocol):
    """Protocol for MCPConnectionManager functionality needed by ServerRegistry."""

    async def get_server(
        self,
        server_name: str,
        client_session_factory: ClientSessionFactory | None = None,
    ) -> "ServerConnection": ...

    async def disconnect_server(self, server_name: str) -> None: ...

    async def disconnect_all_servers(self) -> None: ...


@runtime_checkable
class ServerInitializerProtocol(Protocol):
    """Protocol for temporary (non-persistent) server connections used by gen_client."""

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: ClientSessionFactory | None = None,
    ) -> AbstractAsyncContextManager[ClientSession]:
        """Initialize a server and yield a client session."""
        ...

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server, or None if not yet initialized."""
        ...


@runtime_checkable
class ServerRegistryProtocol(ServerInitializerProtocol, Protocol):
    """Protocol defining the minimal interface of ServerRegistry needed by gen_client."""

    @property
    def registry(self) -> dict[str, "MCPServerSettings"]: ...

    @property
    def connection_manager(self) -> MCPConnectionManagerProtocol: ...

    def get_server_config(self, server_name: str) -> "MCPServerSettings | None": ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...
