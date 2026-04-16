"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, AsyncIterator

from mcp import ClientSession

from fast_agent.config import (
    MCPServerSettings,
    Settings,
    get_settings,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.interfaces import ClientSessionFactory

if TYPE_CHECKING:
    from mcp.types import InitializeResult, ServerCapabilities

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
        self._init_results: dict[str, "InitializeResult"] = {}
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

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server, or None if not yet initialized."""
        init_result = self._init_results.get(server_name)
        return init_result.capabilities if init_result else None

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: ClientSessionFactory | None = None,
        trigger_oauth: bool | None = None,
    ) -> AsyncIterator[ClientSession]:
        """
        Create a temporary connection to a server, initialize the session, and yield it.

        Delegates transport creation to the shared create_transport_context helper.
        Capabilities are stored internally and retrievable via get_server_capabilities().

        Note: transport_metrics and OAuth event handlers are intentionally omitted
        for temporary connections -- they are short-lived probes, not managed lifecycles.

        Args:
            server_name: Name of the server to initialize.
            client_session_factory: Optional factory for creating the ClientSession.
        """
        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
        from fast_agent.mcp.mcp_connection_manager import (
            _is_http_auth_challenge_error,
            _resolve_oauth_mode,
            create_transport_context,
        )

        config = self.get_server_config(server_name)
        if config is None:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        oauth_mode = _resolve_oauth_mode(config, trigger_oauth=trigger_oauth)

        @asynccontextmanager
        async def _initialized_session(oauth_enabled: bool) -> AsyncIterator[ClientSession]:
            transport_context = create_transport_context(
                server_name=server_name,
                config=config,
                trigger_oauth=oauth_enabled,
            )

            async with transport_context as (read_stream, write_stream, _get_session_id_cb):
                read_timeout = (
                    timedelta(seconds=config.read_timeout_seconds)
                    if config.read_timeout_seconds
                    else None
                )
                if client_session_factory is not None:
                    session = client_session_factory(
                        read_stream,
                        write_stream,
                        read_timeout,
                        server_config=config,
                    )
                else:
                    session = MCPAgentClientSession(
                        read_stream, write_stream, read_timeout, server_config=config
                    )

                async with session:
                    result: "InitializeResult" = await session.initialize()
                    self._init_results[server_name] = result
                    yield session

        try:
            async with _initialized_session(oauth_mode == 'force') as session:
                yield session
        except Exception as exc:
            if oauth_mode == 'auto' and _is_http_auth_challenge_error(exc):
                logger.info(
                    '%s: Received authentication challenge during probe; retrying with OAuth enabled',
                    server_name,
                )
                async with _initialized_session(True) as session:
                    yield session
            else:
                raise
