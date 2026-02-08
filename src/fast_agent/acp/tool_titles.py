"""Helpers for formatting ACP tool call titles."""

from typing import Any

BUILTIN_SERVER_PREFIX = "acp_"
ARGUMENT_TRUNCATION_LIMIT = 70


def _is_builtin_server(server_name: str | None) -> bool:
    return bool(server_name) and server_name.startswith(BUILTIN_SERVER_PREFIX)


def build_tool_title(
    tool_name: str,
    server_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    *,
    include_args: bool = True,
    arg_truncation_limit: int = ARGUMENT_TRUNCATION_LIMIT,
) -> str:
    """Build a user-facing tool title for ACP notifications.

    Args:
        tool_name: Tool name to display.
        server_name: Optional MCP server name.
        arguments: Optional tool arguments to summarize.
        include_args: Whether to include an argument summary in the title.
        arg_truncation_limit: Maximum length of the argument summary.

    Returns:
        A formatted title string for ACP tool calls.
    """
    if server_name and not _is_builtin_server(server_name):
        title = f"{server_name}/{tool_name}"
    else:
        title = tool_name

    if include_args and arguments:
        arg_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items()))
        if len(arg_str) > arg_truncation_limit:
            cutoff = max(arg_truncation_limit - 3, 0)
            arg_str = f"{arg_str[:cutoff]}..." if cutoff else "..."
        title = f"{title}({arg_str})"

    return title.replace("\r", " ").replace("\n", " ")
