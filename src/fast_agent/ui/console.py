"""
Centralized console configuration for MCP Agent.

This module provides shared console instances for consistent output handling:
- console: Main console for general output
- error_console: Error console for application errors (writes to stderr)
- server_console: Special console for MCP server output
"""

from rich.console import Console

# Main console for general output
# Note: For ACP/stdio modes, all output must go to stderr to avoid polluting JSON-RPC
console = Console(
    stderr=True,  # Always use stderr to avoid stdout pollution in stdio/ACP modes
    color_system="auto",
)

# Error console for application errors
error_console = Console(
    stderr=True,
    style="bold red",
)

# Special console for MCP server output
# This could have custom styling to distinguish server messages
server_console = Console(
    # Not stderr since we want to maintain output ordering with other messages
    style="dim blue",  # Or whatever style makes server output distinct
)
