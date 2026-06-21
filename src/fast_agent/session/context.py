"""Session-manager context helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.session.session_manager import SessionManager


def attach_session_manager(instance: AgentInstance, manager: SessionManager) -> None:
    """Attach the active session manager to every agent context in an instance."""
    for agent in instance.agents.values():
        context = agent.context
        if context is not None:
            context.session_manager = manager
