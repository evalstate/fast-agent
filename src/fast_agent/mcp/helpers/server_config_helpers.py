"""Helper functions for type-safe server config access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


@runtime_checkable
class _SessionRequestContext(Protocol):
    session: object


@overload
def get_server_config(ctx: MCPAgentClientSession) -> MCPServerSettings | None: ...


@overload
def get_server_config(ctx: _SessionRequestContext) -> MCPServerSettings | None: ...


@overload
def get_server_config(ctx: object) -> MCPServerSettings | None: ...


def get_server_config(ctx: object) -> MCPServerSettings | None:
    """Extract server config from context if available.

    Supports either an MCPAgentClientSession or an MCP request context carrying one.
    """
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    if isinstance(ctx, MCPAgentClientSession):
        return ctx.server_config

    if isinstance(ctx, _SessionRequestContext) and isinstance(
        ctx.session, MCPAgentClientSession
    ):
        return ctx.session.server_config

    return None
