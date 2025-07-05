"""Helper functions for type-safe server config access."""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mcp_agent.config import MCPServerSettings


def get_server_config(ctx: Any) -> Optional["MCPServerSettings"]:
    """Extract server config from context if available.
    
    Type guard helper that safely accesses server_config with proper type checking.
    """
    # Import here to avoid circular import
    from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
    
    # Check if ctx has a session attribute (RequestContext case)
    if hasattr(ctx, "session"):
        if isinstance(ctx.session, MCPAgentClientSession) and hasattr(ctx.session, 'server_config'):
            return ctx.session.server_config
    # Also check if ctx itself is MCPAgentClientSession (direct call case)
    elif isinstance(ctx, MCPAgentClientSession) and hasattr(ctx, 'server_config'):
        return ctx.server_config
    
    return None