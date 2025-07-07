"""Helper functions for server configuration and naming."""

from typing import Any, Dict


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
    if identifier.startswith("./"):
        identifier = identifier[2:]

    # Handle npm package names with org prefix (only if no file extension)
    has_file_ext = any(identifier.endswith(ext) for ext in [".py", ".js", ".ts"])
    if "/" in identifier and not has_file_ext:
        # This is likely an npm package, take the part after the last slash
        identifier = identifier.split("/")[-1]

    # Remove file extension for common script files
    for ext in [".py", ".js", ".ts"]:
        if identifier.endswith(ext):
            identifier = identifier[: -len(ext)]
            break

    # Replace special characters with underscores
    # Remove @ prefix if present
    identifier = identifier.lstrip("@")

    # Replace non-alphanumeric characters with underscores
    server_name = ""
    for char in identifier:
        if char.isalnum():
            server_name += char
        else:
            server_name += "_"

    # Clean up multiple underscores
    while "__" in server_name:
        server_name = server_name.replace("__", "_")

    # Remove leading/trailing underscores
    server_name = server_name.strip("_")

    return server_name


async def add_servers_to_config(fast_app: Any, servers: Dict[str, Dict[str, Any]]) -> None:
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

    from mcp_agent.config import MCPServerSettings, MCPSettings

    # Initialize the app to ensure context is ready
    await fast_app.app.initialize()

    # Initialize mcp settings if needed
    if not hasattr(fast_app.app.context.config, "mcp"):
        fast_app.app.context.config.mcp = MCPSettings()

    # Initialize servers dictionary if needed
    if (
        not hasattr(fast_app.app.context.config.mcp, "servers")
        or fast_app.app.context.config.mcp.servers is None
    ):
        fast_app.app.context.config.mcp.servers = {}

    # Add each server to the config
    for server_name, server_config in servers.items():
        # Build server settings based on transport type
        server_settings = {"transport": server_config["transport"]}

        # Add transport-specific settings
        if server_config["transport"] == "stdio":
            server_settings["command"] = server_config["command"]
            server_settings["args"] = server_config["args"]
        elif server_config["transport"] in ["http", "sse"]:
            server_settings["url"] = server_config["url"]
            if "headers" in server_config:
                server_settings["headers"] = server_config["headers"]

        fast_app.app.context.config.mcp.servers[server_name] = MCPServerSettings(**server_settings)
