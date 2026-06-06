"""Helpers for formatting ACP tool call titles."""

BUILTIN_SERVER_PREFIX = "acp_"


def build_tool_title(
    tool_name: str,
    server_name: str | None = None,
) -> str:
    """Build a user-facing tool title for ACP notifications.

    Args:
        tool_name: Tool name to display.
        server_name: Optional MCP server name.

    Returns:
        A formatted title string for ACP tool calls.
    """
    if server_name and not server_name.startswith(BUILTIN_SERVER_PREFIX):
        title = f"{server_name}/{tool_name}"
    else:
        title = tool_name

    return title.replace("\r", " ").replace("\n", " ")
