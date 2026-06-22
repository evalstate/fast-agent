"""Session-manager context helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.session.session_manager import SessionManager


@runtime_checkable
class SessionContextCapable(Protocol):
    context: Any


def attach_session_manager(instance: AgentInstance, manager: SessionManager) -> None:
    """Attach the active session manager to every agent context in an instance."""
    for agent in instance.agents.values():
        if isinstance(agent, SessionContextCapable) and agent.context is not None:
            context = agent.context
            context.session_manager = manager
