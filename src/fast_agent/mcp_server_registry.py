"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.client_connection import MCPClientConnection

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp_types import ServerCapabilities

    from fast_agent.config import MCPServerSettings, Settings
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime

logger = get_logger(__name__)


class ServerRegistry:
    """
    Maps MCP Server configurations to names; can be populated from a YAML file (other formats soon)

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (dict[str, MCPServerSettings]): Loaded server configurations.
    """

    registry: dict[str, MCPServerSettings]

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
        self._capabilities: dict[str, ServerCapabilities] = {}
        self._config = config
        self.registry = config.mcp.servers if config is not None and config.mcp is not None else {}

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
        if server_config.name is None:
            server_config.name = server_name
        return server_config

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server, or None if not yet initialized."""
        return self._capabilities.get(server_name)

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        callback_runtime: MCPClientCallbackRuntime | None = None,
        trigger_oauth: bool | None = None,
    ) -> AsyncIterator[MCPClientConnection]:
        """
        Create a temporary connection to a server, initialize the session, and yield it.

        Delegates transport creation to the shared create_transport_context helper.
        Capabilities are stored internally and retrievable via get_server_capabilities().

        Note: transport_metrics and OAuth event handlers are intentionally omitted
        for temporary connections -- they are short-lived probes, not managed lifecycles.

        Args:
            server_name: Name of the server to initialize.
            callback_runtime: Optional fast-agent callback configuration.
        """
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
        async def _initialized_session(
            oauth_enabled: bool,
        ) -> AsyncIterator[MCPClientConnection]:
            transport = create_transport_context(
                server_name=server_name,
                config=config,
                trigger_oauth=oauth_enabled,
                active_home=getattr(self._config, "_fast_agent_home", None),
                no_home=bool(getattr(self._config, "_fast_agent_no_home", False)),
            )

            # Import lazily to keep the registry usable while Context is being
            # constructed; the callback runtime depends on agent configuration.
            from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime

            callbacks = callback_runtime or MCPClientCallbackRuntime(
                server_name=server_name, server_config=config
            )
            connection = MCPClientConnection(
                transport,
                callbacks,
                read_timeout_seconds=config.read_timeout_seconds,
                cache=False,
            )
            async with connection:
                if connection.server_capabilities is not None:
                    self._capabilities[server_name] = connection.server_capabilities
                yield connection

        try:
            async with _initialized_session(oauth_mode == "force") as session:
                yield session
        except Exception as exc:
            if oauth_mode == "auto" and _is_http_auth_challenge_error(exc):
                logger.info(
                    "%s: Received authentication challenge during probe; retrying with OAuth enabled",
                    server_name,
                )
                async with _initialized_session(True) as session:
                    yield session
            else:
                raise
