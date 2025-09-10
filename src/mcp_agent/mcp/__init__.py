"""MCP (Model Context Protocol) package.

This package's __init__ intentionally avoids re-exporting symbols to prevent
import-time circular dependencies. Import the specific submodules or use the
Fast Agent namespaces instead:

- Helpers: `fast_agent.mcp.helpers`
- Message type: `fast_agent.mcp.PromptMessageExtended`
- Interfaces: `fast_agent.interfaces` (generic) and `fast_agent.mcp.interfaces` (MCP-specific)
"""

__all__: list[str] = []
