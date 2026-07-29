"""
This module defines a `ServerRegistry` for MCP configuration and capabilities.

Client construction and lifecycle ownership live in the MCP client gateway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_types import ServerCapabilities

    from fast_agent.config import MCPServerSettings, Settings

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
        return server_config

    @property
    def active_home(self) -> str | None:
        return self._config._fast_agent_home if self._config is not None else None

    @property
    def no_home(self) -> bool:
        return self._config._fast_agent_no_home if self._config is not None else False

    def get_server_capabilities(self, server_name: str) -> "ServerCapabilities | None":
        """Return cached capabilities for a server, or None if not yet initialized."""
        return self._capabilities.get(server_name)

    def set_server_capabilities(
        self, server_name: str, capabilities: "ServerCapabilities"
    ) -> None:
        self._capabilities[server_name] = capabilities
