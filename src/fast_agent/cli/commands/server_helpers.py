"""Helper functions for server configuration and naming."""

import re
from typing import Any

from fast_agent.utils.transports import uses_mcp_remote_transport

SCRIPT_EXTENSIONS = (".py", ".js", ".ts")


def generate_server_name(identifier: str) -> str:
    """Generate a clean server name from various identifiers.

    Args:
        identifier: Package name, file path, or other identifier

    Returns:
        Clean server name with only alphanumeric and underscore characters

    Examples:
        >>> generate_server_name("@modelcontextprotocol/server-filesystem")
        'server_filesystem'
        >>> generate_server_name("./src/my-server.py")
        'src_my_server'
        >>> generate_server_name("my-mcp-server")
        'my_mcp_server'
    """

    # Remove leading ./ if present
    identifier = identifier.removeprefix("./")

    # Handle npm package names with org prefix (only if no file extension)
    has_file_ext = identifier.endswith(SCRIPT_EXTENSIONS)
    if "/" in identifier and not has_file_ext:
        # This is likely an npm package, take the part after the last slash
        identifier = identifier.split("/")[-1]

    # Remove file extension for common script files
    for ext in SCRIPT_EXTENSIONS:
        if identifier.endswith(ext):
            identifier = identifier[: -len(ext)]
            break

    # Replace special characters with underscores
    # Remove @ prefix if present
    identifier = identifier.lstrip("@")

    server_name = re.sub(r"\W+", "_", identifier)
    server_name = re.sub(r"_+", "_", server_name)
    return server_name.strip("_")


async def add_servers_to_config(fast_app: Any, servers: dict[str, dict[str, Any]]) -> None:
    """Add server configurations to the FastAgent app config.

    This function handles the repetitive initialization and configuration
    of MCP servers, ensuring the app is initialized and the config
    structure exists before adding servers.

    Args:
        fast_app: The FastAgent instance
        servers: Dictionary of server configurations
    """
    if not servers:
        return

    from fast_agent.config import MCPServerSettings, MCPSettings

    # Initialize the app to ensure context is ready
    await fast_app.app.initialize()

    context = fast_app.app.context
    config = context.config

    # Initialize mcp settings if needed
    if vars(config).get("mcp") is None:
        config.mcp = MCPSettings()

    # Initialize servers dictionary if needed
    if config.mcp.servers is None:
        config.mcp.servers = {}

    # Add each server to the config (and keep the runtime registry in sync)
    for server_name, server_config in servers.items():
        # Build server settings based on transport type
        server_settings: dict[str, Any] = {"transport": server_config["transport"]}

        # Add transport-specific settings
        if server_config["transport"] == "stdio":
            server_settings["command"] = server_config["command"]
            server_settings["args"] = server_config["args"]
        elif uses_mcp_remote_transport(server_config["transport"]):
            server_settings["url"] = server_config["url"]
            if "headers" in server_config:
                server_settings["headers"] = server_config["headers"]
            if "auth" in server_config:
                server_settings["auth"] = server_config["auth"]

        mcp_server = MCPServerSettings(**server_settings)
        # Update config model
        config.mcp.servers[server_name] = mcp_server
        # Ensure ServerRegistry sees dynamic additions even when no config file exists
        server_registry = vars(context).get("server_registry")
        if server_registry is not None:
            server_registry.registry[server_name] = mcp_server
