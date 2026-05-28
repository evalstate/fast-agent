from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.session.identity import SessionSaveContext, resolve_session_for_save

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fast_agent.session.session_manager import SessionManager


class _Session:
    def __init__(self, session_id: str, metadata: dict[str, object] | None = None) -> None:
        self.info = SimpleNamespace(name=session_id, metadata=dict(metadata or {}))
        self.save_calls = 0

    def _save_metadata(self) -> None:
        self.save_calls += 1


class _Manager:
    def __init__(self, generated_session_id: str = "generated") -> None:
        self.current_session: _Session | None = None
        self.generated_session_id = generated_session_id
        self.known_sessions: dict[str, _Session] = {}
        self.created_with_id: list[tuple[str, dict[str, object] | None]] = []
        self.created: list[dict[str, object] | None] = []

    def get_session(self, name: str) -> _Session | None:
        return self.known_sessions.get(name)

    def set_current_session(self, session: _Session) -> None:
        self.current_session = session

    def create_session_with_id(
        self,
        session_id: str,
        metadata: dict[str, object] | None = None,
    ) -> _Session:
        seeded_metadata = dict(metadata or {})
        seeded_metadata.setdefault("acp_session_id", session_id)
        session = _Session(session_id, seeded_metadata)
        self.current_session = session
        self.created_with_id.append((session_id, metadata))
        return session

    def create_session(self, name: str | None = None, metadata: dict | None = None) -> _Session:
        del name
        session = _Session(self.generated_session_id, dict(metadata or {}))
        self.current_session = session
        self.created.append(metadata)
        return session


def test_resolve_session_for_save_uses_app_manager_for_app_scope(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manager = _Manager()
    manager_calls: list[Path | None] = []

    def get_manager(cwd: Path | None) -> _Manager:
        manager_calls.append(cwd)
        return manager

    identity = resolve_session_for_save(
        current_session=None,
        get_manager=cast("Callable[[Path | None], SessionManager]", get_manager),
        context=SessionSaveContext(
            acp_session_id="acp-1",
            session_cwd=workspace,
            session_store_scope="app",
            session_store_cwd=None,
        ),
        seed_metadata={"agent_name": "main", "model": "passthrough"},
    )

    assert manager_calls == [None]
    assert identity.created is True
    assert identity.session.info.name == "acp-1"
    assert identity.session.info.metadata == {
        "agent_name": "main",
        "model": "passthrough",
        "cwd": str(workspace),
        "acp_session_id": "acp-1",
    }


def test_resolve_session_for_save_prefers_store_cwd_for_workspace_scope(tmp_path: Path) -> None:
    session_cwd = tmp_path / "workspace"
    store_cwd = tmp_path / "store"
    manager = _Manager()
    manager_calls: list[Path | None] = []

    def get_manager(cwd: Path | None) -> _Manager:
        manager_calls.append(cwd)
        return manager

    resolve_session_for_save(
        current_session=None,
        get_manager=cast("Callable[[Path | None], SessionManager]", get_manager),
        context=SessionSaveContext(
            acp_session_id=None,
            session_cwd=session_cwd,
            session_store_scope="workspace",
            session_store_cwd=store_cwd,
        ),
        seed_metadata={"agent_name": "main"},
    )

    assert manager_calls == [store_cwd]


def test_resolve_session_for_save_loads_existing_acp_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manager = _Manager()
    existing = _Session("acp-1", {"agent_name": "old"})
    manager.current_session = _Session("other")
    manager.known_sessions["acp-1"] = existing

    identity = resolve_session_for_save(
        current_session=None,
        get_manager=cast("Callable[[Path | None], SessionManager]", lambda cwd: manager),
        context=SessionSaveContext(
            acp_session_id="acp-1",
            session_cwd=workspace,
            session_store_scope="workspace",
            session_store_cwd=None,
        ),
        seed_metadata={"agent_name": "main"},
    )

    assert identity.created is False
    assert identity.session is existing
    assert manager.current_session is existing
    assert existing.info.metadata["cwd"] == str(workspace)
    assert existing.info.metadata["acp_session_id"] == "acp-1"
    assert existing.save_calls == 1
