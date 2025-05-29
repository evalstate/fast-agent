"""Helper functions for type-safe server config access."""

from typing import TYPE_CHECKING, Optional

from mcp import ClientSession

if TYPE_CHECKING:
    from mcp_agent.config import MCPServerSettings


def get_server_config(ctx: ClientSession) -> Optional["MCPServerSettings"]:
    """Extract server config from context if available.
    
    Type guard helper that safely accesses server_config with proper type checking.
    """
    # Import here to avoid circular import
    from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
    
    if (hasattr(ctx, "session") and 
        isinstance(ctx.session, MCPAgentClientSession) and
        ctx.session.server_config):
        return ctx.session.server_config
    return None