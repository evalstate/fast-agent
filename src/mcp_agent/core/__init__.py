"""
Core components and utilities for MCP Agent (compat layer).

This module provides compatibility imports that now live under the
`fast_agent` package after migration.
"""

from fast_agent.mcp.mcp_content import (
    Assistant,
    MCPFile,
    MCPImage,
    MCPPrompt,
    MCPText,
    User,
    create_message,
)

__all__ = [
    # MCP content creation functions
    "MCPText",
    "MCPImage",
    "MCPFile",
    "MCPPrompt",
    "User",
    "Assistant",
    "create_message",
]
