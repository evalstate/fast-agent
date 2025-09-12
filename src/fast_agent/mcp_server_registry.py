"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from typing import Dict

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
        registry (Dict[str, MCPServerSettings]): Loaded server configurations.
    """

    registry: Dict[str, MCPServerSettings] = {}

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
    ) -> Dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, MCPServerSettings]: A dictionary of server configurations.

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
