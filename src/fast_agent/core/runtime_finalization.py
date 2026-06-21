"""Runtime state restoration helpers for finalized agent instances."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import SessionHydrationResult
    from fast_agent.session.session_manager import ResumeSessionAgentsResult


@dataclass(frozen=True, slots=True)
class SessionRestoreRequest:
    session_id: str | None
    fallback_agent_name: str | None


async def restore_requested_session(
    agents: Mapping[str, AgentProtocol],
    request: SessionRestoreRequest,
) -> ResumeSessionAgentsResult | None:
    """Hydrate agents from an explicit resume request."""
    from fast_agent.context import get_current_context

    manager = get_current_context().session_manager
    if manager is None:
        raise RuntimeError("Session restore requested without an active session manager.")
    return await manager.resume_session_agents_async(
        agents,
        request.session_id,
        fallback_agent_name=request.fallback_agent_name,
    )


async def hydrate_current_session_for_refresh(
    agents: Mapping[str, AgentProtocol],
    *,
    fallback_agent_name: str | None,
) -> SessionHydrationResult | None:
    """Refresh rebuilt agents from the current persisted session, when one exists."""
    from fast_agent.context import get_current_context
    from fast_agent.session import SessionHydrationPolicy, SessionHydrator

    manager = get_current_context().session_manager
    if manager is None:
        return None
    current_session = manager.current_session
    if current_session is None:
        return None

    loaded_session = manager.load_session(current_session.info.name) or current_session
    hydration = SessionHydrator().hydrate_session(
        session=loaded_session,
        agents=agents,
        fallback_agent_name=fallback_agent_name,
        policy=SessionHydrationPolicy.for_refresh(),
    )
    return await hydration if inspect.isawaitable(hydration) else hydration


def session_restore_warnings(result: ResumeSessionAgentsResult | None) -> list[str]:
    if result is None:
        return []
    warnings = [warning.message for warning in result.warnings]
    warnings.extend(result.usage_notices)
    return warnings


def hydration_warnings(result: SessionHydrationResult | None) -> list[str]:
    if result is None:
        return []
    warnings = [warning.message for warning in result.warnings]
    warnings.extend(result.usage_notices)
    return warnings


def validate_final_provider_state(agents: Mapping[str, AgentProtocol]) -> None:
    from fast_agent.core.validation import validate_provider_keys_post_creation

    validate_provider_keys_post_creation(dict(agents))
