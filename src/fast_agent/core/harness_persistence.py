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
    from fast_agent.session.terminal_pointers import HcParentSessionTarget


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

    async def delete_owned(self, session_id: str, handle: object | None) -> bool: ...


@dataclass(frozen=True, slots=True)
class FileHarnessSessionPersistence:
    """File-backed Harness persistence using ``SessionManager``."""

    home: str | Path | None
    hc_parent: HcParentSessionTarget | None = None

    async def create_or_load(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> object | None:
        from fast_agent.core.logging.logger import get_logger
        from fast_agent.session import SessionHydrator
        from fast_agent.session.session_manager import SessionManager
        from fast_agent.session.snapshot import SessionChildLinkSnapshot
        from fast_agent.session.subagent_runs import subagent_alias_slug

        manager: SessionManager | None = None
        persisted_session: Session | None = None
        created_hc_child = False
        if self.hc_parent is not None:
            parent_manager = SessionManager(
                home_override=self.hc_parent.home_dir,
                surface="harness",
            )
            child_link = SessionChildLinkSnapshot(
                parent_session_id=self.hc_parent.parent_session_id,
                parent_agent_name=self.hc_parent.parent_agent_name,
                tool_name="hc",
                hc_terminal_id=self.hc_parent.hc_terminal_id,
                hc_name=self.hc_parent.hc_name,
            )
            try:
                persisted_session = parent_manager.create_hc_child_session(
                    self.hc_parent.parent_session_dir,
                    self.hc_parent.parent_session_id,
                    child_link,
                    session_id=session_id,
                    alias_slug=subagent_alias_slug(label=self.hc_parent.hc_name, task="hc"),
                    label=self.hc_parent.hc_name,
                    expected_parent_terminal_id=self.hc_parent.parent_terminal_id,
                    expected_core_id=self.hc_parent.core_id,
                )
                manager = parent_manager
                created_hc_child = True
            except Exception as exc:
                parent_manager.close()
                get_logger(__name__).debug(
                    "Falling back to top-level session after hc child creation failure",
                    data={"parent_session_id": self.hc_parent.parent_session_id, "error": str(exc)},
                )

        if manager is None or persisted_session is None:
            manager = SessionManager(
                home_override=self.home,
                surface="harness",
            )
            persisted_session = manager.create_session_with_id(
                session_id,
                metadata={"harness_session_id": session_id},
                metadata_id_key="harness_session_id",
            )
        try:
            attach_session_manager(instance, manager)
            if created_hc_child:
                return persisted_session
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
        except BaseException:
            persisted_session.delete_if_empty()
            manager.close()
            raise

    async def save(
        self,
        handle: object,
        agent: AgentProtocol,
        agent_registry: Mapping[str, AgentProtocol],
    ) -> None:
        session = cast("Session", handle)
        if not session.directory.is_dir():
            return
        await session.save_history(agent, agent_registry=agent_registry)

    async def delete(self, session_id: str) -> None:
        from fast_agent.session.session_manager import SessionManager

        manager = SessionManager(
            home_override=self.home,
        )
        manager.delete_session(session_id)

    async def delete_owned(self, session_id: str, handle: object | None) -> bool:
        from fast_agent.session.session_manager import SessionManager

        session = cast("Session", handle)
        manager = session.manager
        if manager is None:
            fresh_manager = SessionManager(home_override=self.home)
            return fresh_manager.delete_session(session_id)
        return manager.delete_owned_session(session)


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
        if not session.directory.is_dir():
            return
        await session.save_history(agent, agent_registry=agent_registry)

    async def delete(self, session_id: str) -> None:
        if self.delete_persisted_session is not None:
            await self.delete_persisted_session(session_id)

    async def delete_owned(self, session_id: str, handle: object | None) -> bool:
        del handle
        await self.delete(session_id)
        return True


__all__ = [
    "CallbackHarnessSessionPersistence",
    "FileHarnessSessionPersistence",
    "HarnessSessionPersistence",
]
