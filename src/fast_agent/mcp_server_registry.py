"""
This module defines a `ServerRegistry` for MCP configuration and capabilities.

Client construction and lifecycle ownership live in the MCP client gateway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_types import ServerCapabilities

    from fast_agent.config import MCPServerSettings, Settings

logger = get_logger(__name__)

type ServerOrigin = Literal["central", "card", "runtime"]


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
        loaded = config.mcp.servers if config is not None and config.mcp is not None else {}
        self.registry = {name: settings.model_copy(deep=True) for name, settings in loaded.items()}
        self._origins: dict[str, ServerOrigin] = dict.fromkeys(self.registry, "central")
        self._runtime_owners: dict[str, set[str]] = {}
        self._attachment_owners: dict[str, set[str]] = {}

    def register_central(self, server_name: str, config: MCPServerSettings) -> None:
        self._register(server_name, config, "central")

    def register_card(self, server_name: str, config: MCPServerSettings) -> None:
        self._register(server_name, config, "card")

    def register_runtime(
        self,
        server_name: str,
        config: MCPServerSettings,
        *,
        owner: str = "process",
    ) -> None:
        origin = self._origins.get(server_name)
        if origin in {"central", "card"}:
            raise ValueError(
                f"Runtime MCP server '{server_name}' collides with {origin} configuration"
            )
        if origin == "runtime":
            existing = self.registry[server_name]
            if existing != config:
                raise ValueError(
                    f"Runtime MCP server '{server_name}' is already registered with different settings"
                )
            self._runtime_owners.setdefault(server_name, set()).add(owner)
            return
        self._register(server_name, config, "runtime")
        self._runtime_owners[server_name] = {owner}

    def register_runtime_batch(
        self,
        servers: dict[str, MCPServerSettings],
        *,
        owner: str = "process",
    ) -> None:
        for server_name, config in servers.items():
            origin = self._origins.get(server_name)
            if origin in {"central", "card"}:
                raise ValueError(
                    f"Runtime MCP server '{server_name}' collides with {origin} configuration"
                )
            if origin == "runtime" and self.registry[server_name] != config:
                raise ValueError(
                    f"Runtime MCP server '{server_name}' is already registered "
                    "with different settings"
                )
        for server_name, config in servers.items():
            self.register_runtime(server_name, config, owner=owner)

    def replace_card_servers(self, servers: dict[str, MCPServerSettings]) -> None:
        conflicts = [
            (name, origin)
            for name in servers
            if (origin := self._origins.get(name)) in {"central", "runtime"}
        ]
        if conflicts:
            name, origin = conflicts[0]
            raise ValueError(f"Card MCP server '{name}' collides with {origin} configuration")
        for name in [name for name, origin in self._origins.items() if origin == "card"]:
            self.registry.pop(name, None)
            self._origins.pop(name, None)
            self.clear_server_capabilities(name)
        for name, config in servers.items():
            self.register_card(name, config)

    def remove_runtime(self, server_name: str, *, owner: str = "process") -> bool:
        if self._origins.get(server_name) != "runtime":
            return False
        owners = self._runtime_owners.get(server_name, set())
        owners.discard(owner)
        if owners:
            return False
        self.registry.pop(server_name, None)
        self._origins.pop(server_name, None)
        self._runtime_owners.pop(server_name, None)
        self.clear_server_capabilities(server_name)
        return True

    def get_runtime_owners(self, server_name: str) -> frozenset[str]:
        return frozenset(self._runtime_owners.get(server_name, set()))

    def register_attachment(self, server_name: str, *, owner: str) -> None:
        self._attachment_owners.setdefault(server_name, set()).add(owner)

    def release_attachment(self, server_name: str, *, owner: str) -> bool:
        owners = self._attachment_owners.get(server_name)
        if owners is None:
            return True
        owners.discard(owner)
        if owners:
            return False
        self._attachment_owners.pop(server_name, None)
        return True

    def get_attachment_owners(self, server_name: str) -> frozenset[str]:
        return frozenset(self._attachment_owners.get(server_name, set()))

    def get_server_origin(self, server_name: str) -> ServerOrigin | None:
        return self._origins.get(server_name)

    def _register(
        self,
        server_name: str,
        config: MCPServerSettings,
        origin: ServerOrigin,
    ) -> None:
        self.registry[server_name] = config.model_copy(deep=True)
        self._origins[server_name] = origin
        if origin != "runtime":
            self._runtime_owners.pop(server_name, None)
        self.clear_server_capabilities(server_name)

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

    def set_server_capabilities(self, server_name: str, capabilities: "ServerCapabilities") -> None:
        self._capabilities[server_name] = capabilities

    def clear_server_capabilities(self, server_name: str) -> None:
        self._capabilities.pop(server_name, None)
