"""Persistence boundary for Harness sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from fast_agent.session.context import attach_session_manager

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.session_manager import Session, SessionManager


class HarnessSessionPersistence(Protocol):
    """Persistence store for Harness conversation history."""

    async def create_or_load(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> object | None: ...

    async def save(
        self,
        handle: object,
        agent: AgentProtocol,
        agent_registry: Mapping[str, AgentProtocol],
    ) -> None: ...

    async def delete(self, session_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class FileHarnessSessionPersistence:
    """File-backed Harness persistence using ``SessionManager``."""

    home: str | Path | None

    async def create_or_load(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> object | None:
        from fast_agent.session import SessionHydrator
        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            home_override=self.home,
        )
        persisted_session = manager.create_session_with_id(
            session_id,
            metadata={"harness_session_id": session_id},
            metadata_id_key="harness_session_id",
        )
        attach_session_manager(instance, manager)
        fallback_agent_name = instance.app.resolve_target_agent_name(default_agent_name)
        hydration = await SessionHydrator().hydrate_session(
            session=persisted_session,
            agents=instance.agents,
            fallback_agent_name=fallback_agent_name,
        )
        from fast_agent.session.session_manager import ResumeSessionAgentsResult

        instance.app.set_session_restore_result(
            ResumeSessionAgentsResult(
                session=hydration.session,
                loaded=hydration.loaded_agents,
                missing_agents=hydration.skipped_agents,
                usage_notices=hydration.usage_notices,
                warnings=hydration.warnings,
                active_agent=hydration.active_agent,
            )
        )
        return hydration.session

    async def save(
        self,
        handle: object,
        agent: AgentProtocol,
        agent_registry: Mapping[str, AgentProtocol],
    ) -> None:
        session = cast("Session", handle)
        await session.save_history(agent, agent_registry=agent_registry)

    async def delete(self, session_id: str) -> None:
        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            home_override=self.home,
        )
        manager.delete_session(session_id)


@dataclass(frozen=True, slots=True)
class CallbackHarnessSessionPersistence:
    """Adapt legacy Harness persistence callbacks to the persistence Protocol."""

    create_persisted_session: Callable[
        [str, AgentInstance, str | None],
        Awaitable[tuple[SessionManager, Session] | None],
    ]
    delete_persisted_session: Callable[[str], Awaitable[None]] | None = None

    async def create_or_load(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> object | None:
        persisted = await self.create_persisted_session(
            session_id,
            instance,
            default_agent_name,
        )
        if persisted is None:
            return None
        manager, session = persisted
        attach_session_manager(instance, manager)
        return session

    async def save(
        self,
        handle: object,
        agent: AgentProtocol,
        agent_registry: Mapping[str, AgentProtocol],
    ) -> None:
        session = cast("Session", handle)
        await session.save_history(agent, agent_registry=agent_registry)

    async def delete(self, session_id: str) -> None:
        if self.delete_persisted_session is not None:
            await self.delete_persisted_session(session_id)


__all__ = [
    "CallbackHarnessSessionPersistence",
    "FileHarnessSessionPersistence",
    "HarnessSessionPersistence",
]
