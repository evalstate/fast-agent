"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Callable

from mcp import ClientSession
from mcp.types import ServerCapabilities

from fast_agent.config import (
    MCPServerSettings,
    Settings,
    get_settings,
)
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class ServerRegistry:
    """
    Maps MCP Server configurations to names; can be populated from a YAML file (other formats soon)

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (dict[str, MCPServerSettings]): Loaded server configurations.
    """

    registry: dict[str, MCPServerSettings] = {}

    def __init__(
        self,
        config: Settings | None = None,
    ) -> None:
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config (Settings): The Settings object containing the server configurations.
            config_path (str): Path to the YAML configuration file.
        """
        if config is not None and config.mcp is not None:
            self.registry = config.mcp.servers or {}

    ## TODO-- leaving this here to support more file formats to add servers
    def load_registry_from_file(
        self, config_path: str | None = None
    ) -> dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            dict[str, MCPServerSettings]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """
        servers = {}

        settings = get_settings(config_path)

        if (
            settings.mcp is not None
            and hasattr(settings.mcp, "servers")
            and settings.mcp.servers is not None
        ):
            return settings.mcp.servers

        return servers

    def get_server_config(self, server_name: str) -> MCPServerSettings | None:
        """
        Get the configuration for a specific server.

        Args:
            server_name (str): The name of the server.

        Returns:
            MCPServerSettings: The server configuration.
        """

        server_config = self.registry.get(server_name)
        if server_config is None:
            logger.warning(f"Server '{server_name}' not found in registry.")
            return None
        elif server_config.name is None:
            server_config.name = server_name
        return server_config

    @asynccontextmanager
    async def _initialize_server_session(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> AsyncGenerator[tuple[ClientSession, ServerCapabilities | None], None]:
        """
        Create a temporary connection to a server and yield an initialized session.
        """
        config = self.get_server_config(server_name)
        if config is None:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
        from fast_agent.mcp.mcp_connection_manager import create_transport_context

        factory = client_session_factory or MCPAgentClientSession
        transport_ctx = create_transport_context(server_name=server_name, config=config)

        async with transport_ctx as (read_stream, write_stream, _get_session_id):
            read_timeout = (
                timedelta(seconds=config.read_timeout_seconds)
                if config.read_timeout_seconds
                else None
            )
            session = factory(
                read_stream,
                write_stream,
                read_timeout,
                server_config=config,
            )
            async with session:
                result = await session.initialize()
                yield session, result.capabilities

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """Create a temporary connection to a server, initialize the session, and yield it."""

        async with self._initialize_server_session(
            server_name=server_name,
            client_session_factory=client_session_factory,
        ) as (session, _):
            yield session

    async def get_server_capabilities(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] | None = None,
    ) -> ServerCapabilities | None:
        """Create a temporary connection and return the server capabilities."""

        async with self._initialize_server_session(
            server_name=server_name,
            client_session_factory=client_session_factory,
        ) as (_, capabilities):
            return capabilities
