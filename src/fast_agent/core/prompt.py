"""
Compatibility shim: expose Prompt under fast_agent.core.prompt during migration.

Canonical location: fast_agent.mcp.prompt.Prompt
"""

from fast_agent.mcp.prompt import Prompt

__all__ = ["Prompt"]
