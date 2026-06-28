"""Explicit session command runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fast_agent.commands.session_summaries import SessionListSummary
    from fast_agent.config import Settings
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import ResumeSessionAgentsResult, Session, SessionInfo, SessionManager
    from fast_agent.session.identity import SessionStoreScope


@dataclass(slots=True)
class SessionManagerCommandRuntime:
    """Session command runtime backed by one explicit SessionManager store."""

    explicit_manager: SessionManager | None = None
    session_cwd: Path | None = None
    session_store_scope: SessionStoreScope = "workspace"
    session_store_cwd: Path | None = None
    settings: Settings | None = None
    _resolved_manager: SessionManager | None = field(default=None, init=False)

    def resolve_manager(self) -> SessionManager:
        if self.explicit_manager is not None:
            return self.explicit_manager
        if self._resolved_manager is not None:
            return self._resolved_manager
        if self.settings is not None:
            from fast_agent.session import SessionManager

            self._resolved_manager = SessionManager(
                cwd=self._manager_cwd(),
                environment_override=self.settings.environment_dir,
            )
            return self._resolved_manager

        from fast_agent.session import get_session_manager

        return get_session_manager(cwd=self._manager_cwd())

    def current_session_id(self) -> str | None:
        current_session = self.resolve_manager().current_session
        return current_session.info.name if current_session is not None else None

    def active_session_id(self, *, fallback_session_id: str | None = None) -> str | None:
        manager = self.resolve_manager()
        current_session = manager.current_session
        current_session_id = current_session.info.name if current_session is not None else None
        if fallback_session_id is None or current_session_id == fallback_session_id:
            return current_session_id

        session = manager.get_session(fallback_session_id)
        return session.info.name if session is not None else None

    def build_list_summary(self, *, show_help: bool = False) -> SessionListSummary:
        from fast_agent.commands.session_summaries import build_session_list_summary

        return build_session_list_summary(
            manager=self.resolve_manager(),
            show_help=show_help,
        )

    def create_session(
        self,
        *,
        session_name: str | None,
        session_id: str | None = None,
        replace_existing: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> Session:
        manager = self.resolve_manager()
        if session_id is None:
            return manager.create_session(session_name)

        if replace_existing:
            manager.delete_session(session_id)
        return manager.create_session_with_id(session_id, metadata=metadata)

    def list_sessions(self) -> list[SessionInfo]:
        return self.resolve_manager().list_sessions()

    def delete_session(self, session_id: str) -> bool:
        return self.resolve_manager().delete_session(session_id)

    def resolve_session_name(self, name: str | None) -> str | None:
        return self.resolve_manager().resolve_session_name(name)

    def get_session(self, session_id: str) -> Session | None:
        return self.resolve_manager().get_session(session_id)

    async def resume_agents(
        self,
        agents: Mapping[str, AgentProtocol],
        session_id: str | None,
        *,
        fallback_agent_name: str | None,
    ) -> ResumeSessionAgentsResult | None:
        return await self.resolve_manager().resume_session_agents_async(
            agents,
            session_id,
            fallback_agent_name=fallback_agent_name,
        )

    def title_session(self, title: str, *, session_id: str | None = None) -> Session | None:
        manager = self.resolve_manager()
        session = manager.current_session
        if session_id:
            if session is None or session.info.name != session_id:
                session = manager.create_session_with_id(session_id)
        elif session is None:
            session = manager.create_session()
        if session is not None:
            session.set_title(title)
        return session

    def fork_current_session(self, *, title: str | None = None) -> Session | None:
        return self.resolve_manager().fork_current_session(title=title)

    def _manager_cwd(self) -> Path | None:
        if self.session_store_scope == "app":
            return None
        if self.session_store_cwd is not None:
            return self.session_store_cwd
        return self.session_cwd


__all__ = ["SessionManagerCommandRuntime"]
