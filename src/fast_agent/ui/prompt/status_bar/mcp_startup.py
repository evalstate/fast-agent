"""Compact live MCP startup status, replacing the idle version label."""

from collections.abc import Sequence

from fast_agent.mcp.startup import ServerStartupStatus


def render_mcp_startup_status(snapshot: Sequence[ServerStartupStatus]) -> str | None:
    states = [status.state for status in snapshot]
    if "error" in states:
        return "<style fg='ansired'>MCP ERR · /mcp</style>"
    if "auth" in states:
        return "<style fg='ansiyellow'>MCP AUTH · /mcp auth</style>"
    if "pending" in states:
        return f"MCP {states.count('ready')}/{len(states)}"
    return None
