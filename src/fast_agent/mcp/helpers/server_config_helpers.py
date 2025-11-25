"""Helper functions for type-safe server config access."""

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings


def get_server_config(ctx: Any) -> Union["MCPServerSettings", None]:
    """Extract server config from context if available.

    Type guard helper that safely accesses server_config with proper type checking.
    """
    # Import here to avoid circular import
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    # Check if ctx has a session attribute (RequestContext case)
    if hasattr(ctx, "session"):
        if isinstance(ctx.session, MCPAgentClientSession):
            return ctx.session.server_config
    # Also check if ctx itself is MCPAgentClientSession (direct call case)
    elif isinstance(ctx, MCPAgentClientSession):
        return ctx.server_config

    return None
