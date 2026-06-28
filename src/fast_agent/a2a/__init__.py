"""A2A support for fast-agent."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.a2a.server import AgentA2AServer as AgentA2AServer


def __getattr__(name: str):
    if name == "AgentA2AServer":
        from fast_agent.a2a.server import AgentA2AServer

        return AgentA2AServer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["AgentA2AServer"]
