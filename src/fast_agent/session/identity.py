"""Session save-path identity resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from fast_agent.session.session_manager import Session, SessionManager

type SessionStoreScope = Literal["workspace", "app"]


@dataclass(slots=True)
class SessionSaveContext:
    """Runtime inputs that determine where and how a save should land."""

    acp_session_id: str | None
    session_cwd: Path | None
    session_store_scope: SessionStoreScope
    session_store_cwd: Path | None


@dataclass(slots=True)
class SessionSaveIdentity:
    """Resolved session target for a save operation."""

    manager: SessionManager
    session: Session
    created: bool
    acp_session_id: str | None
    session_cwd: Path | None
    session_store_scope: SessionStoreScope
    session_store_cwd: Path | None


def resolve_session_for_save(
    *,
    current_session: Session | None,
    get_manager: Callable[[Path | None], SessionManager],
    context: SessionSaveContext,
    seed_metadata: Mapping[str, object] | None,
) -> SessionSaveIdentity:
    """Resolve the authoritative session target for a save-path operation."""
    manager = get_manager(_manager_cwd_for_save(context))
    session = current_session if current_session is not None else manager.current_session
    created = False
    create_metadata = _seed_save_metadata(seed_metadata, context)

    acp_session_id = context.acp_session_id
    if acp_session_id:
        if session is None or session.info.name != acp_session_id:
            existing_session = manager.get_session(acp_session_id)
            if existing_session is not None:
                manager.set_current_session(existing_session)
                session = existing_session
            else:
                manager.create_session_with_id(acp_session_id, metadata=create_metadata or None)
                session = manager.current_session
                created = True
    elif session is None:
        manager.create_session(metadata=create_metadata or None)
        session = manager.current_session
        created = True

    if session is None:
        raise RuntimeError("Session resolution failed to produce a current session")

    _stamp_identity_metadata(session, context)
    return SessionSaveIdentity(
        manager=manager,
        session=session,
        created=created,
        acp_session_id=context.acp_session_id,
        session_cwd=context.session_cwd,
        session_store_scope=context.session_store_scope,
        session_store_cwd=context.session_store_cwd,
    )


def _manager_cwd_for_save(context: SessionSaveContext) -> Path | None:
    if context.session_store_scope == "app":
        return None
    if context.session_store_cwd is not None:
        return context.session_store_cwd
    return context.session_cwd


def _seed_save_metadata(
    seed_metadata: Mapping[str, object] | None,
    context: SessionSaveContext,
) -> dict[str, object]:
    metadata = dict(seed_metadata or {})
    if context.session_cwd is not None:
        metadata["cwd"] = str(context.session_cwd)
    return metadata


def _stamp_identity_metadata(session: Session, context: SessionSaveContext) -> None:
    changed = False
    metadata = session.info.metadata

    acp_session_id = context.acp_session_id
    if acp_session_id is not None and metadata.get("acp_session_id") != acp_session_id:
        metadata["acp_session_id"] = acp_session_id
        changed = True

    session_cwd = context.session_cwd
    if session_cwd is not None:
        session_cwd_value = str(session_cwd)
        if metadata.get("cwd") != session_cwd_value:
            metadata["cwd"] = session_cwd_value
            changed = True

    if changed:
        session._save_metadata()
