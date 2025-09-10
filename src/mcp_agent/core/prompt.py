"""Compatibility shim for Prompt helper during migration.

The actual implementation has moved to fast_agent.mcp.prompt.
"""

from fast_agent.mcp.prompt import Prompt  # re-export

__all__ = ["Prompt"]
