"""
Interface definitions to prevent circular imports.
This module defines protocols (interfaces) that can be used to break circular dependencies.
"""

from contextlib import AbstractAsyncContextManager
from typing import (
    TYPE_CHECKING,
    Any,
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
    from mcp.shared.dispatcher import ProgressFnT
    from mcp_types import (
        CallToolResult,
        GetPromptResult,
        ReadResourceResult,
        RequestParamsMeta,
        ServerCapabilities,
    )

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

__all__ = [
    "AgentProtocol",
    "ClientSessionFactory",
    "CompletingClientSession",
    "FastAgentLLMProtocol",
    "LLMFactoryProtocol",
    "LlmAgentProtocol",
    "ModelFactoryFunctionProtocol",
    "ModelT",
    "ServerConnection",
    "ServerInitializerProtocol",
    "ServerRegistryProtocol",
]


@runtime_checkable
class CompletingClientSession(Protocol):
    """Session operations that resolve modern input-required rounds."""

    async def call_tool_complete(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        progress_callback: "ProgressFnT | None" = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> "CallToolResult": ...

    async def read_resource_complete(
        self,
        uri: str,
        *,
        meta: "RequestParamsMeta | None" = None,
    ) -> "ReadResourceResult": ...

    async def get_prompt_complete(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        meta: "RequestParamsMeta | None" = None,
    ) -> "GetPromptResult": ...


@runtime_checkable
class ClientSessionFactory(Protocol):
    """Protocol for creating client sessions across persistent and temporary connections."""

    def __call__(
        self,
        read_stream: MemoryObjectReceiveStream,
        write_stream: MemoryObjectSendStream,
        read_timeout: float | None,
        *,
        server_config: "MCPServerSettings | None" = None,
        transport_metrics: "TransportChannelMetrics | None" = None,
    ) -> ClientSession: ...


@runtime_checkable
class ServerInitializerProtocol(Protocol):
    """Protocol for temporary (non-persistent) server connections used by gen_client."""

    def initialize_server(
        self,
        server_name: str,
        client_session_factory: ClientSessionFactory | None = None,
        trigger_oauth: bool | None = None,
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

    def get_server_config(self, server_name: str) -> "MCPServerSettings | None": ...


class ServerConnection(Protocol):
    """Protocol for server connection objects returned by MCPConnectionManager."""

    @property
    def session(self) -> ClientSession: ...
