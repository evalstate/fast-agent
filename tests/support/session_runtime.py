from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fast_agent.commands.session_summaries import SessionListSummary

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class StubSession:
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    history_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.info = SimpleNamespace(
            name=self.name,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            history_files=self.history_files,
            metadata=self.metadata,
        )

    def set_title(self, title: str) -> None:
        self.info.metadata["title"] = title


@dataclass
class StubSessionRuntime:
    manager: Any | None = None
    current_session: StubSession | None = None
    sessions: dict[str, StubSession] = field(default_factory=dict)
    created: list[str | None] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    resumed: Any | None = None
    list_summary: SessionListSummary | None = None
    forked_title: str | None = None

    def resolve_manager(self) -> Any:
        if self.manager is None:
            raise AssertionError("resolve_manager was not expected")
        return self.manager

    def current_session_id(self) -> str | None:
        return self.current_session.info.name if self.current_session is not None else None

    def active_session_id(self, *, fallback_session_id: str | None = None) -> str | None:
        return self.current_session_id() or fallback_session_id

    def build_list_summary(self, *, show_help: bool = False) -> SessionListSummary:
        if self.list_summary is not None:
            return self.list_summary
        del show_help
        return SessionListSummary(
            entries=[],
            usage="",
            entry_summaries=[],
        )

    def create_session(
        self,
        *,
        session_name: str | None,
        session_id: str | None = None,
        replace_existing: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> Any:
        if session_id is not None and replace_existing:
            self.sessions.pop(session_id, None)
        name = session_id or session_name or f"session-{len(self.created) + 1}"
        session = StubSession(name=name, metadata=dict(metadata or {}))
        self.created.append(session_name)
        self.sessions[name] = session
        self.current_session = session
        return session

    def list_sessions(self) -> list[Any]:
        return list(self.sessions.values())

    def delete_session(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        return self.sessions.pop(session_id, None) is not None

    def resolve_session_name(self, name: str | None) -> str | None:
        return name or self.current_session_id()

    def get_session(self, session_id: str) -> Any | None:
        return self.sessions.get(session_id)

    async def resume_agents(
        self,
        agents: "Mapping[str, Any]",
        session_id: str | None,
        *,
        fallback_agent_name: str | None,
    ) -> Any:
        del agents, session_id, fallback_agent_name
        return self.resumed

    def title_session(self, title: str, *, session_id: str | None = None) -> Any | None:
        session = self.current_session
        if session_id is not None:
            session = self.sessions.get(session_id) or self.create_session(
                session_name=None,
                session_id=session_id,
            )
        elif session is None:
            session = self.create_session(session_name=None)
        session.set_title(title)
        return session

    def fork_current_session(self, *, title: str | None = None) -> Any | None:
        self.forked_title = title
        return self.create_session(session_name=title)
