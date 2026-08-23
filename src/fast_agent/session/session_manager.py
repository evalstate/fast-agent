"""
Session management for fast-agent.

Provides automatic saving and loading of conversation sessions in the fast-agent home.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import pathlib
import re
import secrets
import shutil
import string
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.constants import DEFAULT_HOME_DIR
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.paths import resolve_home_paths
from fast_agent.session.locking import (
    SessionBusyError,
    SessionLockStore,
    SessionOwner,
    SessionOwnerLease,
)
from fast_agent.session.snapshot import (
    SessionChildLinkSnapshot,
    SessionExecutionStatus,
    SessionSnapshot,
    capture_session_snapshot,
    clone_session_snapshot_for_fork,
    load_session_snapshot,
    session_info_from_snapshot,
    snapshot_from_session_info,
)
from fast_agent.session.subagent_runs import (
    SUBAGENT_ALIAS_KEY,
    SUBAGENT_LABEL_KEY,
    SUBAGENT_ORDINAL_KEY,
    SUBAGENT_TASK_PREVIEW_KEY,
    format_subagent_alias,
)
from fast_agent.session.trajectory import TRAJECTORIES_DIR
from fast_agent.utils.async_utils import run_coroutine
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.hydrator import SessionHydrationResult, SessionHydrationWarning
    from fast_agent.session.identity import SessionSaveIdentity
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)

SESSION_ID_LENGTH = 6
SESSION_ID_ALPHABET = string.ascii_letters + string.digits
SESSION_TIMESTAMP_FORMAT = "%y%m%d%H%M"
SESSION_ID_PATTERN = re.compile(
    rf"^(?:[A-Za-z0-9]{{{SESSION_ID_LENGTH}}}|\d{{10}}-[A-Za-z0-9]{{{SESSION_ID_LENGTH}}})$"
)
HISTORY_PREFIX = "history_"
HISTORY_SUFFIX = ".json"
HISTORY_PREVIOUS_SUFFIX = "_previous.json"
LAST_USER_PROMPT_LIMIT = 200


def session_metadata_title(metadata: Mapping[str, object] | None) -> str | None:
    """Extract the first displayable title or preview from session metadata."""
    if metadata is None:
        return None
    for key in ("title", "label", "first_user_preview"):
        value = metadata.get(key)
        if isinstance(value, str) and (normalized := " ".join(value.split())):
            return normalized
    return None


def _normalized_home_override(cwd: pathlib.Path) -> str | None:
    """Return the active fast-agent home override as an absolute path string when set."""
    from fast_agent.home import resolve_fast_agent_home

    home = resolve_fast_agent_home(cwd=cwd)
    if home is None or home.source == "default":
        return None
    return str(home.path)


def _session_home_override(
    *,
    cwd: pathlib.Path,
    explicit_cwd: bool,
    home_override: str | pathlib.Path | None,
    respect_env_override: bool,
) -> str | pathlib.Path | None:
    if home_override is not None or not respect_env_override:
        return home_override

    from fast_agent.config import get_settings

    settings = get_settings()
    if settings.home is not None:
        return settings.home

    home_override = _normalized_home_override(cwd)
    if home_override is not None:
        return home_override

    if explicit_cwd and settings.home is None and settings._fast_agent_home_source == "default":
        return DEFAULT_HOME_DIR

    return None


def display_session_name(name: str) -> str:
    """Return a display-friendly session name without timestamp prefixes."""
    if SESSION_ID_PATTERN.match(name) and "-" in name:
        return name.split("-", 1)[1]
    return name


def is_session_pinned(info: "SessionInfo") -> bool:
    """Return True if the session is marked as pinned."""
    value = info.metadata.get("pinned") if isinstance(info.metadata, dict) else None
    return value is True


def _sanitize_component(name: str, limit: int = 100) -> str:
    """Sanitize a name for filesystem safety."""
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    name = "".join(c for c in name if c.isalnum() or c in "_-.")
    return name[:limit] or "agent"


def _extract_history_agent(filename: str) -> str:
    """Extract agent name from a history filename."""
    if filename.startswith(HISTORY_PREFIX) and filename.endswith(HISTORY_SUFFIX):
        agent_name = filename[len(HISTORY_PREFIX) : -len(HISTORY_SUFFIX)]
        agent_name = agent_name.removesuffix("_previous")
        return agent_name or "agent"
    return pathlib.Path(filename).stem


def _normalize_explicit_session_id(session_id: str | None) -> str | None:
    """Return a stripped, path-safe explicit session id or None."""
    requested_id = strip_to_none(session_id)
    if requested_id is None:
        return None
    if pathlib.Path(requested_id).name != requested_id:
        return None
    return requested_id


def _first_user_preview(messages: list["PromptMessageExtended"], limit: int = 240) -> str | None:
    for message in messages:
        if message.role != "user":
            continue
        if message.is_template or message.tool_results:
            continue
        text = message.all_text() or message.first_text() or ""
        text = " ".join(text.split())
        if not text:
            continue
        return text[:limit]
    return None


def normalize_user_prompt(value: object, limit: int = LAST_USER_PROMPT_LIMIT) -> str | None:
    if isinstance(value, str):
        text = value
    else:
        content = getattr(value, "content", None)
        blocks = content if isinstance(content, list) else [content] if content is not None else []
        text = "\n".join(part for block in blocks if (part := get_text(block)) is not None)
    normalized = " ".join(text.split())
    return normalized[:limit] if normalized else None


def _last_user_prompt(
    messages: list["PromptMessageExtended"],
    limit: int = LAST_USER_PROMPT_LIMIT,
) -> str | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        if message.is_template or message.tool_results:
            continue
        if prompt := normalize_user_prompt(message, limit):
            return prompt
    return None


def get_session_history_window() -> int:
    """Return the configured session history window size."""
    try:
        from fast_agent.config import get_settings

        settings = get_settings()
        value = settings.session_history_window
        if isinstance(value, bool):
            return 20
        if isinstance(value, int):
            return value
        return int(value)
    except Exception:
        return 20


def apply_session_window(
    sessions: "Sequence[SessionInfo]",
    limit: int | None = None,
) -> list["SessionInfo"]:
    """Apply the session list window while preserving pinned overflow entries.

    The primary list remains the newest ``limit`` sessions by ``last_activity``. Any
    pinned sessions that would otherwise fall outside the window are appended at the
    bottom so they remain visible/selectable.
    """
    session_list = list(sessions)
    if not session_list:
        return []

    if limit is None:
        limit = get_session_history_window()

    if limit <= 0:
        return session_list

    visible = list(session_list[:limit])
    visible_names = {session.name for session in visible}
    overflow_pinned = [
        session
        for session in session_list[limit:]
        if is_session_pinned(session) and session.name not in visible_names
    ]
    return visible + overflow_pinned


def summarize_session_histories(session: "Session") -> dict[str, int]:
    """Summarize available histories for a session by agent name."""
    history_files = list(session.info.history_files)
    if not history_files:
        history_files = [path.name for path in session.directory.glob("history_*.json")]

    summary: dict[str, int] = {}
    for filename in history_files:
        if filename.endswith(HISTORY_PREVIOUS_SUFFIX):
            continue
        path = session.directory / filename
        if not path.exists():
            continue

        agent_name = _extract_history_agent(filename)
        try:
            from fast_agent.mcp.prompt_serialization import load_messages

            summary[agent_name] = len(load_messages(str(path)))
        except Exception as exc:
            logger.warning(
                "Failed to summarize session history",
                data={"session": session.info.name, "file": filename, "error": str(exc)},
            )
    return summary


@dataclass
class SessionInfo:
    """Metadata about a session."""

    name: str
    created_at: datetime
    last_activity: datetime
    history_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> SessionInfo:
        """Create SessionInfo from dictionary."""
        snapshot = load_session_snapshot(data)
        return session_info_from_snapshot(snapshot)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "history_files": self.history_files,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ResumeSessionAgentsResult:
    """Structured result for SessionManager.resume_session_agents."""

    session: Session
    loaded: dict[str, pathlib.Path]
    missing_agents: list[str]
    usage_notices: list[str] = field(default_factory=list)
    warnings: list[SessionHydrationWarning] = field(default_factory=list)
    active_agent: str | None = None


@dataclass(frozen=True, slots=True)
class SessionDeleteResult:
    """Detailed persisted session deletion outcome."""

    session_id: str
    status: Literal["deleted", "not_found", "invalid", "busy", "error"]
    owner: SessionOwner | None = None

    @property
    def deleted(self) -> bool:
        return self.status == "deleted"


@dataclass(frozen=True, slots=True)
class SessionPruneResult:
    """Detailed persisted session pruning outcome."""

    removed: int
    busy: tuple[SessionDeleteResult, ...] = ()


class Session:
    """Represents a single conversation session."""

    def __init__(
        self,
        info: SessionInfo,
        directory: pathlib.Path,
        *,
        manager: SessionManager | None = None,
    ) -> None:
        """Initialize session."""
        self.info = info
        self.directory = directory
        self._manager = manager
        # History file writes leave the event loop; serialize them per session
        # so temp-file/rotation steps from overlapping saves cannot interleave.
        self._history_save_lock = asyncio.Lock()

    @property
    def manager(self) -> SessionManager | None:
        """Return the manager that owns this session, when available."""
        return self._manager

    async def save_history(
        self,
        agent: AgentProtocol,
        filename: str | None = None,
        *,
        agent_registry: Mapping[str, AgentProtocol] | None = None,
        identity: "SessionSaveIdentity | None" = None,
        resolved_prompts: Mapping[str, str] | None = None,
        checkpoint: bool = False,
    ) -> str:
        """Save agent history to this session.

        ``checkpoint`` marks frequent mid-turn saves: history is written as
        compact JSON and previously captured git state is reused instead of
        re-queried. Turn-boundary saves keep the full-fidelity behaviour.
        """
        async with self._history_save_lock:
            with self._mutation_lock():
                return await self._save_history_transaction(
                    agent,
                    filename,
                    agent_registry=agent_registry,
                    identity=identity,
                    resolved_prompts=resolved_prompts,
                    checkpoint=checkpoint,
                )

    async def _save_history_transaction(
        self,
        agent: AgentProtocol,
        filename: str | None,
        *,
        agent_registry: Mapping[str, AgentProtocol] | None,
        identity: "SessionSaveIdentity | None",
        resolved_prompts: Mapping[str, str] | None,
        checkpoint: bool,
    ) -> str:
        """Write one complete committed checkpoint while exclusively locked."""
        from fast_agent.history.history_exporter import HistoryExporter

        self.info.last_activity = datetime.now()

        rotating = filename is None
        current_filename: str | None = None
        previous_filename: str | None = None

        # Generate filename if not provided
        if filename is None:
            agent_name = getattr(agent, "name", None)
            agent_label = _sanitize_component(agent_name or "agent")
            current_filename = f"history_{agent_label}.json"
            previous_filename = f"history_{agent_label}_previous.json"
            result = await self._save_rotating_history(
                agent,
                current_filename=current_filename,
                previous_filename=previous_filename,
                compact=checkpoint,
            )
            filename = current_filename
        else:
            filepath = self.directory / filename
            result = await HistoryExporter.save(agent, str(filepath), compact=checkpoint)

        # Update session info
        if rotating and current_filename:
            history_files = [
                name
                for name in self.info.history_files
                if name not in {current_filename, previous_filename}
            ]
            if previous_filename:
                previous_path = self.directory / previous_filename
                if previous_path.exists():
                    history_files.append(previous_filename)
            history_files.append(current_filename)
            self.info.history_files = history_files
        elif filename not in self.info.history_files:
            self.info.history_files.append(filename)

        agent_name = getattr(agent, "name", None)
        if agent_name:
            history_map = self.info.metadata.get("last_history_by_agent")
            if not isinstance(history_map, dict):
                history_map = {}
            history_map[agent_name] = filename
            self.info.metadata["last_history_by_agent"] = history_map

        if "first_user_preview" not in self.info.metadata:
            preview = _first_user_preview(agent.message_history)
            if preview:
                self.info.metadata["first_user_preview"] = preview
        last_user_prompt = _last_user_prompt(agent.message_history)
        if last_user_prompt:
            self.info.metadata["last_user_prompt"] = last_user_prompt

        snapshot = capture_session_snapshot(
            session=self,
            active_agent=agent,
            agent_registry=agent_registry,
            identity=identity or self._default_save_identity(),
            resolved_prompts=resolved_prompts,
            refresh_git=not checkpoint,
        )
        self._write_snapshot(snapshot)
        return result

    async def _save_rotating_history(
        self,
        agent: AgentProtocol,
        *,
        current_filename: str,
        previous_filename: str,
        compact: bool = False,
    ) -> str:
        """Save history using a current/previous rotation scheme."""
        from fast_agent.history.history_exporter import HistoryExporter

        current_path = self.directory / current_filename
        previous_path = self.directory / previous_filename
        temp_path: pathlib.Path | None = None

        try:
            suffix = current_path.suffix or ".json"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=self.directory,
                prefix=f".{current_filename}.tmp.",
                suffix=suffix,
            ) as handle:
                temp_path = pathlib.Path(handle.name)

            await HistoryExporter.save(agent, str(temp_path), compact=compact)

            if current_path.exists():
                current_path.replace(previous_path)
            temp_path.replace(current_path)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    logger.warning(
                        "Failed to clean up history temp file",
                        data={"path": str(temp_path)},
                    )

        return str(current_path)

    def _save_metadata(self) -> None:
        """Save session metadata without replacing persisted runtime state."""
        with self._mutation_lock():
            self._save_metadata_locked()

    def _save_metadata_locked(self) -> None:
        updated = snapshot_from_session_info(self.info)
        snapshot = self._load_snapshot_or_compatibility()
        snapshot.created_at = updated.created_at
        snapshot.last_activity = updated.last_activity
        snapshot.metadata = updated.metadata
        _merge_compatibility_continuation(snapshot, updated)
        self._write_snapshot(snapshot)

    def _save_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Save a typed snapshot payload."""
        with self._mutation_lock():
            self._write_snapshot(snapshot)

    def _write_snapshot(self, snapshot: SessionSnapshot) -> None:
        metadata_file = self.directory / "session.json"
        payload = snapshot.model_dump(mode="json")
        self._atomic_write_json(metadata_file, payload)

    def _default_save_identity(self) -> "SessionSaveIdentity":
        """Build a compatibility save identity when a caller does not supply one."""
        from fast_agent.session.identity import SessionSaveIdentity

        metadata = self.info.metadata if isinstance(self.info.metadata, dict) else {}
        raw_session_cwd = metadata.get("cwd")
        session_cwd = None
        if isinstance(raw_session_cwd, str) and raw_session_cwd:
            session_cwd = pathlib.Path(raw_session_cwd).expanduser().resolve()

        acp_session_id = metadata.get("acp_session_id")
        manager = self._manager
        if manager is None:
            raise RuntimeError(
                "Session save requires an owning SessionManager. Load or create sessions through "
                "SessionManager, or pass an explicit SessionSaveIdentity."
            )

        return SessionSaveIdentity(
            manager=manager,
            session=self,
            created=False,
            acp_session_id=acp_session_id if isinstance(acp_session_id, str) else None,
            session_cwd=session_cwd,
            session_store_scope="workspace",
            session_store_cwd=manager.workspace_dir,
        )

    def set_pinned(self, pinned: bool) -> None:
        """Pin or unpin the session to prevent auto-pruning."""
        with self._mutation_lock():
            if pinned:
                self.info.metadata["pinned"] = True
            else:
                self.info.metadata.pop("pinned", None)
            self._save_metadata_locked()

    def has_persisted_content(self) -> bool:
        """Return True when this session has saved conversation history."""
        if self.info.history_files:
            return True
        history_map = self.info.metadata.get("last_history_by_agent")
        if isinstance(history_map, dict) and history_map:
            return True
        trajectory_dir = self.directory / TRAJECTORIES_DIR
        if trajectory_dir.is_dir() and any(trajectory_dir.iterdir()):
            return True
        children_dir = self.directory / "children"
        if children_dir.is_dir() and any(children_dir.iterdir()):
            return True
        return any(self.directory.glob(f"{HISTORY_PREFIX}*{HISTORY_SUFFIX}"))

    def set_execution_status(self, status: SessionExecutionStatus) -> None:
        """Record the terminal lifecycle state for an execution session."""
        with self._mutation_lock():
            snapshot = self._load_snapshot_or_compatibility()
            now = datetime.now()
            snapshot.last_activity = now
            snapshot.execution.status = status
            snapshot.execution.completed_at = now if status != "running" else None
            self.info.last_activity = now
            self._write_snapshot(snapshot)

    def _load_snapshot_or_compatibility(self) -> SessionSnapshot:
        metadata_file = self.directory / "session.json"
        try:
            with metadata_file.open(encoding="utf-8") as handle:
                return load_session_snapshot(json.load(handle))
        except Exception:
            return snapshot_from_session_info(self.info)

    def load_snapshot(self) -> SessionSnapshot:
        """Load the typed continuation snapshot for this session."""
        with self._checkpoint_lock():
            return self._load_snapshot_or_compatibility()

    def is_user_visible(self) -> bool:
        """Return whether this session should appear in interactive session surfaces."""
        if self.has_persisted_content() or is_session_pinned(self.info):
            return True
        return session_metadata_title(self.info.metadata) is not None

    def delete_if_empty(self) -> bool:
        """Delete this session when it only contains startup metadata."""
        if self.is_user_visible():
            return False
        manager = self._manager
        if manager is None:
            return False
        return manager.delete_owned_session(self)

    def _atomic_write_json(self, path: pathlib.Path, payload: dict[str, Any]) -> None:
        temp_path: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = pathlib.Path(handle.name)
            temp_path.replace(path)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    logger.warning(
                        "Failed to clean up session metadata temp file",
                        data={"path": str(temp_path)},
                    )

    def _checkpoint_lock(self):
        manager = self._manager
        if manager is None:
            raise RuntimeError("Persisted session operations require an owning SessionManager.")
        return manager._locks.checkpoint(self.info.name)

    def _mutation_lock(self):
        manager = self._manager
        if manager is None:
            raise RuntimeError("Persisted session operations require an owning SessionManager.")
        manager._acquire_owner(self.info.name)
        return manager._locks.checkpoint(self.info.name)

    def delete(self) -> None:
        """Delete this session."""
        manager = self._manager
        if manager is None:
            raise RuntimeError("Persisted session deletion requires an owning SessionManager.")
        manager.delete_owned_session(self)

    def set_title(self, title: str) -> None:
        """Set a user-friendly title for this session."""
        with self._mutation_lock():
            self.info.metadata["title"] = title
            self.info.last_activity = datetime.now()
            self._save_metadata_locked()

    def set_last_user_prompt(self, prompt: object) -> str | None:
        """Persist the normalized latest external user prompt."""
        normalized = normalize_user_prompt(prompt)
        if normalized is None:
            return None
        with self._mutation_lock():
            self.info.metadata["last_user_prompt"] = normalized
            self.info.last_activity = datetime.now()
            self._save_metadata_locked()
        return normalized

    def latest_history_path(self, agent_name: str | None = None) -> pathlib.Path | None:
        """Return the most recent history file for this session, if any."""
        if agent_name:
            history_map = self.info.metadata.get("last_history_by_agent")
            if isinstance(history_map, dict):
                filename = history_map.get(agent_name)
                if filename:
                    path = self.directory / filename
                    if path.exists():
                        return path

        for filename in reversed(self.info.history_files):
            path = self.directory / filename
            if path.exists():
                return path
        candidates = sorted(
            self.directory.glob("history_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None


def _merge_compatibility_continuation(
    snapshot: SessionSnapshot,
    updated: SessionSnapshot,
) -> None:
    continuation = snapshot.continuation
    updated_continuation = updated.continuation

    if updated_continuation.active_agent is not None:
        continuation.active_agent = updated_continuation.active_agent
    if updated_continuation.cwd is not None:
        continuation.cwd = updated_continuation.cwd
    if updated_continuation.git is not None:
        continuation.git = updated_continuation.git
    if updated_continuation.lineage.forked_from is not None:
        continuation.lineage.forked_from = updated_continuation.lineage.forked_from
    if updated_continuation.lineage.acp_session_id is not None:
        continuation.lineage.acp_session_id = updated_continuation.lineage.acp_session_id

    for agent_name, updated_agent in updated_continuation.agents.items():
        agent = continuation.agents.get(agent_name)
        if agent is None:
            continuation.agents[agent_name] = updated_agent
        elif updated_agent.history_file is not None:
            agent.history_file = updated_agent.history_file


class SessionManager:
    """Manages conversation sessions stored in the fast-agent home."""

    def __init__(
        self,
        *,
        cwd: pathlib.Path | None = None,
        home_override: str | pathlib.Path | None = None,
        respect_env_override: bool = True,
        surface: str | None = None,
        retain_multiple: bool = False,
    ) -> None:
        """Initialize session manager."""
        explicit_cwd = cwd is not None
        base = (cwd or pathlib.Path.cwd()).resolve()
        home_override = _session_home_override(
            cwd=base,
            explicit_cwd=explicit_cwd,
            home_override=home_override,
            respect_env_override=respect_env_override,
        )
        home_paths = resolve_home_paths(cwd=base, override=home_override)
        self.workspace_dir = base
        self.base_dir = home_paths.sessions
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks = SessionLockStore(self.base_dir)
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._surface = surface
        self._retain_multiple = retain_multiple
        self._owner_leases: dict[str, SessionOwnerLease] = {}
        self._current_session: Session | None = None

    @property
    def current_session(self) -> Session | None:
        """Get the currently active session."""
        return self._current_session

    def create_session(self, name: str | None = None, metadata: dict | None = None) -> Session:
        """Create a new session."""
        session_metadata = dict(metadata or {})
        session_id = None
        if name and SESSION_ID_PATTERN.match(name):
            session_id = name
        elif name:
            session_metadata.setdefault("title", name)

        if session_id is None:
            session_id = self._generate_session_id()

        while True:
            session_dir = self.base_dir / session_id
            if session_dir.exists():
                session_id = self._generate_session_id()
                continue
            self._acquire_owner(session_id)
            try:
                session_dir.mkdir()
            except FileExistsError:
                self.release_session(session_id)
                session_id = self._generate_session_id()
                continue
            break

        try:
            now = datetime.now()
            info = SessionInfo(
                name=session_id,
                created_at=now,
                last_activity=now,
                history_files=[],
                metadata=session_metadata,
            )
            session = Session(info, session_dir, manager=self)
            session._save_metadata()
        except BaseException:
            shutil.rmtree(session_dir, ignore_errors=True)
            self.release_session(session_id)
            raise
        self._activate(session)
        self._prune_sessions()
        logger.info(f"Created new session: {session_id}")
        return session

    def create_session_with_id(
        self,
        session_id: str,
        metadata: dict | None = None,
        *,
        metadata_id_key: str | None = "acp_session_id",
    ) -> Session:
        """Create or load a session using the provided id."""
        requested_id = _normalize_explicit_session_id(session_id)
        session_metadata = dict(metadata or {})
        if requested_id and metadata_id_key is not None:
            session_metadata.setdefault(metadata_id_key, requested_id)

        if requested_id is None:
            logger.warning(
                "Invalid session id provided; falling back to generated id",
                data={"session_id": session_id},
            )
            return self.create_session(metadata=session_metadata)

        session_dir = self.base_dir / requested_id
        if session_dir.exists():
            session = self.load_session(requested_id)
            if session:
                if (
                    metadata_id_key is not None
                    and session.info.metadata.get(metadata_id_key) != requested_id
                ):
                    session.info.metadata[metadata_id_key] = requested_id
                    session._save_metadata()
                return session

        self._acquire_owner(requested_id)
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            self.release_session(requested_id)
            loaded = self.load_session(requested_id)
            if loaded is None:
                raise RuntimeError(
                    f"Session '{requested_id}' appeared without valid metadata."
                ) from None
            return loaded

        try:
            now = datetime.now()
            info = SessionInfo(
                name=requested_id,
                created_at=now,
                last_activity=now,
                history_files=[],
                metadata=session_metadata,
            )
            session = Session(info, session_dir, manager=self)
            session._save_metadata()
        except BaseException:
            shutil.rmtree(session_dir, ignore_errors=True)
            self.release_session(requested_id)
            raise
        self._activate(session)
        self._prune_sessions()
        logger.info(f"Created new session: {requested_id}")
        return session

    def create_child_session(
        self,
        parent: Session,
        child_link: SessionChildLinkSnapshot,
        *,
        alias_slug: str | None = None,
        label: str | None = None,
        task_preview: str | None = None,
    ) -> Session:
        """Create a detached, non-resumable child below an existing session."""
        if parent.manager is not self:
            raise ValueError("Parent session is not owned by this SessionManager")
        if child_link.parent_session_id != parent.info.name:
            raise ValueError("Child link parent_session_id does not match parent session")

        children_dir = parent.directory / "children"
        children_dir.mkdir(parents=True, exist_ok=True)
        lock = self._locks.auxiliary(parent.info.name, "subagents", timeout=5)
        with lock:
            ordinal = self._next_subagent_ordinal(children_dir) if alias_slug is not None else None
            while True:
                child_id = self._generate_session_id()
                child_dir = children_dir / child_id
                try:
                    child_dir.mkdir()
                except FileExistsError:
                    continue
                break

            now = datetime.now()
            metadata: dict[str, Any] = {}
            if ordinal is not None and alias_slug is not None:
                metadata[SUBAGENT_ALIAS_KEY] = format_subagent_alias(ordinal, alias_slug)
                metadata[SUBAGENT_ORDINAL_KEY] = ordinal
                if label is not None:
                    metadata[SUBAGENT_LABEL_KEY] = label
                if task_preview is not None:
                    metadata[SUBAGENT_TASK_PREVIEW_KEY] = task_preview
            child = Session(
                SessionInfo(
                    name=child_id,
                    created_at=now,
                    last_activity=now,
                    history_files=[],
                    metadata=metadata,
                ),
                child_dir,
                manager=self,
            )
            snapshot = snapshot_from_session_info(child.info)
            snapshot.execution.resumable = False
            snapshot.execution.child_link = child_link
            snapshot.execution.status = "running"
            snapshot.execution.started_at = now
            try:
                child._save_snapshot(snapshot)
            except Exception:
                shutil.rmtree(child_dir, ignore_errors=True)
                self.release_session(child_id)
                raise
        logger.info(
            "Created child session",
            data={
                "parent_session": parent.info.name,
                "child_session": child_id,
                "subagent_alias": metadata.get(SUBAGENT_ALIAS_KEY),
            },
        )
        return child

    @staticmethod
    def _next_subagent_ordinal(children_dir: pathlib.Path) -> int:
        ordinals: list[int] = []
        child_count = 0
        for child_dir in children_dir.iterdir():
            metadata_file = child_dir / "session.json"
            if not child_dir.is_dir() or not metadata_file.exists():
                continue
            child_count += 1
            try:
                with metadata_file.open(encoding="utf-8") as handle:
                    snapshot = load_session_snapshot(json.load(handle))
            except Exception:
                continue
            ordinal = snapshot.metadata.extras.get(SUBAGENT_ORDINAL_KEY)
            if isinstance(ordinal, int) and ordinal > 0:
                ordinals.append(ordinal)
        return max(max(ordinals, default=0), child_count) + 1

    def list_child_sessions(self, parent: Session) -> list[Session]:
        """Return child sessions without changing the active root session."""
        if parent.manager is not self:
            raise ValueError("Parent session is not owned by this SessionManager")

        children_dir = parent.directory / "children"
        if not children_dir.is_dir():
            return []

        children: list[Session] = []
        for child_dir in children_dir.iterdir():
            metadata_file = child_dir / "session.json"
            if not child_dir.is_dir() or not metadata_file.exists():
                continue
            try:
                with metadata_file.open(encoding="utf-8") as handle:
                    info = session_info_from_snapshot(load_session_snapshot(json.load(handle)))
            except Exception as exc:
                logger.warning(
                    "Failed to load child session metadata",
                    data={
                        "parent_session": parent.info.name,
                        "path": str(metadata_file),
                        "error": str(exc),
                    },
                )
                continue
            children.append(Session(info, child_dir, manager=self))

        children.sort(key=lambda child: child.info.last_activity, reverse=True)
        return children

    def list_sessions(self, *, include_empty: bool = True) -> list[SessionInfo]:
        """List all available sessions."""
        sessions = []
        if not self.base_dir.exists():
            return sessions

        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue

            metadata_file = session_dir / "session.json"
            if metadata_file.exists():
                try:
                    with metadata_file.open(encoding="utf-8") as f:
                        snapshot = load_session_snapshot(json.load(f))
                        if snapshot.execution.child_link is not None:
                            continue
                        info = session_info_from_snapshot(snapshot)
                        if not include_empty:
                            session = Session(info, session_dir, manager=self)
                            if not session.is_user_visible():
                                continue
                        sessions.append(info)
                except Exception as e:
                    logger.warning(f"Failed to load session metadata from {metadata_file}: {e}")

        sessions.sort(key=lambda info: info.last_activity, reverse=True)
        return sessions

    def prune_empty_sessions(self) -> int:
        """Remove abandoned unpinned sessions that contain only startup metadata."""
        return self.prune_empty_sessions_result().removed

    def prune_empty_sessions_result(self) -> SessionPruneResult:
        """Remove disposable sessions and report owner-busy skips."""
        current_name = self._current_session.info.name if self._current_session else None
        removed = 0
        busy: list[SessionDeleteResult] = []
        for info in self.list_sessions():
            if info.name == current_name:
                continue
            session = self.get_session(info.name)
            if session is None or session.is_user_visible():
                continue
            result = self.delete_session_result(info.name)
            if result.deleted:
                removed += 1
            elif result.status == "busy":
                busy.append(result)
        return SessionPruneResult(removed=removed, busy=tuple(busy))

    def load_session(self, name: str) -> Session | None:
        """Acquire ownership, then load an existing session without mutating it."""
        session_id = _normalize_explicit_session_id(name)
        if session_id is None:
            logger.warning(
                "Invalid session id provided for loading",
                data={"session_id": name},
            )
            return None
        session_dir = self.base_dir / session_id
        metadata_file = session_dir / "session.json"

        if not session_dir.is_dir() or not metadata_file.exists():
            return None

        acquired_here = session_id not in self._owner_leases
        self._acquire_owner(session_id)
        try:
            with metadata_file.open(encoding="utf-8") as f:
                data = json.load(f)
                snapshot = load_session_snapshot(data)

            info = session_info_from_snapshot(snapshot)
            session = Session(info, session_dir, manager=self)
            self._activate(session)
            logger.info(f"Loaded session: {session_id}")
            return session
        except Exception as e:
            if acquired_here:
                self.release_session(session_id)
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, name: str) -> bool:
        """Delete an unowned session, preserving the compatibility boolean API."""
        return self.delete_session_result(name).deleted

    def delete_session_result(self, name: str) -> SessionDeleteResult:
        """Delete a session unless any manager or process owns it."""
        session_id = _normalize_explicit_session_id(name)
        if session_id is None:
            logger.warning(
                "Invalid session id provided for deletion",
                data={"session_id": name},
            )
            return SessionDeleteResult(name, "invalid")
        session_dir = self.base_dir / session_id

        if not session_dir.is_dir():
            return SessionDeleteResult(session_id, "not_found")

        if lease := self._owner_leases.get(session_id):
            return SessionDeleteResult(session_id, "busy", lease.owner)

        try:
            lease = self._locks.acquire_owner(
                session_id,
                started_at=self._started_at,
                surface=self._surface or "maintenance",
                token=secrets.token_urlsafe(24),
            )
        except SessionBusyError as exc:
            return SessionDeleteResult(session_id, "busy", exc.owner)

        try:
            with self._locks.checkpoint(session_id):
                shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
            return SessionDeleteResult(session_id, "deleted")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return SessionDeleteResult(session_id, "error")
        finally:
            lease.release()

    def delete_owned_session(self, session: Session) -> bool:
        """Delete a session owned by this manager as an explicit lifecycle action."""
        session_id = session.info.name
        if session_id not in self._owner_leases:
            return self.delete_session(session_id)
        try:
            with self._locks.checkpoint(session_id):
                shutil.rmtree(session.directory)
        except FileNotFoundError:
            self.release_session(session_id)
            return False
        logger.info(f"Deleted session: {session_id}")
        self.release_session(session_id)
        return True

    async def save_current_session(
        self,
        agent: AgentProtocol,
        filename: str | None = None,
        *,
        agent_registry: Mapping[str, AgentProtocol] | None = None,
        identity: "SessionSaveIdentity | None" = None,
        resolved_prompts: Mapping[str, str] | None = None,
        checkpoint: bool = False,
    ) -> str | None:
        """Save history to the current session."""
        if identity is not None:
            self._current_session = identity.session

        if self._current_session and not self._current_session.directory.exists():
            missing_session_id = self._current_session.info.name
            logger.warning(
                "Current session directory is missing; creating a replacement session",
                data={"session": missing_session_id},
            )
            self.release_session(missing_session_id)

        if not self._current_session:
            # Auto-create a session if none exists
            agent_name = agent.name
            metadata: dict[str, Any] = {}
            if agent_name:
                metadata["agent_name"] = agent_name
            model_name = agent.config.model
            if model_name:
                metadata["model"] = model_name
            self.create_session(metadata=metadata or None)
            logger.warning(
                "save_current_session created a fallback session; "
                "the session hook should have created one earlier",
                data={"agent_name": agent_name},
            )

        assert self._current_session is not None
        return await self._current_session.save_history(
            agent,
            filename,
            agent_registry=agent_registry,
            identity=identity,
            resolved_prompts=resolved_prompts,
            checkpoint=checkpoint,
        )

    def load_latest_session(self, *, require_content: bool = False) -> Session | None:
        """Load the most recently used session."""
        sessions = self.list_sessions()
        for info in sessions:
            if not require_content:
                return self.load_session(info.name)
            session = self.get_session(info.name)
            if session is not None and session.has_persisted_content():
                return self.load_session(info.name)
        return None

    async def _hydrate_session_agents_async(
        self,
        agents: Mapping[str, AgentProtocol],
        name: str | None = None,
        fallback_agent_name: str | None = None,
    ) -> SessionHydrationResult | None:
        from fast_agent.session.hydrator import SessionHydrator

        session_name = self._resolve_session_name(name)
        session = (
            self.load_latest_session(require_content=True)
            if session_name is None
            else self.load_session(session_name)
        )
        if session is None:
            return None

        hydration = SessionHydrator().hydrate_session(
            session=session,
            agents=agents,
            fallback_agent_name=fallback_agent_name,
        )
        if inspect.isawaitable(hydration):
            return await hydration
        return hydration

    def resume_session(
        self, agent: AgentProtocol, name: str | None = None
    ) -> tuple[Session, pathlib.Path | None, list[str]] | None:
        """Resume a session through the hydrator compatibility path."""
        result = self.resume_session_agents(
            {agent.name: agent},
            name,
            fallback_agent_name=agent.name,
        )
        if result is None:
            return None

        notices = list(result.usage_notices)
        notices.extend(warning.message for warning in result.warnings)
        return result.session, result.loaded.get(agent.name), notices

    async def resume_session_agents_async(
        self,
        agents: Mapping[str, AgentProtocol],
        name: str | None = None,
        fallback_agent_name: str | None = None,
    ) -> ResumeSessionAgentsResult | None:
        """Resume a session and adapt hydrator output for local callers."""
        displaced_session = self._current_session
        hydration = await self._hydrate_session_agents_async(
            agents,
            name,
            fallback_agent_name=fallback_agent_name,
        )
        if hydration is None:
            return None
        if displaced_session is not None and displaced_session is not hydration.session:
            displaced_session.delete_if_empty()

        missing_agents = list(hydration.skipped_agents)
        if missing_agents:
            logger.warning(
                "Session metadata references missing agents",
                data={"session": hydration.session.info.name, "agents": missing_agents},
            )

        return ResumeSessionAgentsResult(
            session=hydration.session,
            loaded=dict(hydration.loaded_agents),
            missing_agents=missing_agents,
            usage_notices=list(hydration.usage_notices),
            warnings=list(hydration.warnings),
            active_agent=hydration.active_agent,
        )

    def resume_session_agents(
        self,
        agents: Mapping[str, AgentProtocol],
        name: str | None = None,
        fallback_agent_name: str | None = None,
    ) -> ResumeSessionAgentsResult | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run_coroutine(
                self.resume_session_agents_async(
                    agents,
                    name,
                    fallback_agent_name=fallback_agent_name,
                )
            )
        raise RuntimeError(
            "SessionManager.resume_session_agents() cannot be used from an async context; "
            "use resume_session_agents_async() instead."
        )

    def fork_current_session(self, title: str | None = None) -> Session | None:
        """Fork the current or latest committed checkpoint."""
        return self.fork_session(title=title)

    def fork_session(
        self,
        source_name: str | None = None,
        *,
        title: str | None = None,
    ) -> Session | None:
        """Fork a named or current session without acquiring source ownership."""
        if source_name is not None:
            resolved_name = self._resolve_session_name(source_name)
            source = self.get_session(resolved_name) if resolved_name is not None else None
        elif self._current_session is not None:
            source = self._current_session
        else:
            sessions = self.list_sessions()
            source = self.get_session(sessions[0].name) if sessions else None
        if not source:
            return None

        with self._locks.checkpoint(source.info.name):
            source_snapshot = self._load_authoritative_snapshot(source)
            new_session = self._publish_fork(
                source=source,
                source_snapshot=source_snapshot,
                title=title,
            )
        self._activate(new_session)
        return new_session

    def _publish_fork(
        self,
        *,
        source: Session,
        source_snapshot: SessionSnapshot,
        title: str | None,
    ) -> Session:
        session_id = self.generate_session_id()
        self._acquire_owner(session_id)
        lease = self._owner_leases[session_id]
        session_dir = self.base_dir / session_id
        staging_root = self.base_dir / ".staging"
        staging_root.mkdir(exist_ok=True)
        staging_dir = staging_root / f"{session_id}-{lease.owner.token}"
        try:
            staging_dir.mkdir()
            created_at = datetime.now()
            copied_history_files = self._copy_fork_history_files(
                source=source,
                snapshot=source_snapshot,
                dest_dir=staging_dir,
            )
            forked_snapshot = clone_session_snapshot_for_fork(
                source_snapshot,
                new_session_id=session_id,
                copied_history_files=copied_history_files,
                cloned_at=created_at,
                title=title,
            )
            session = Session(
                session_info_from_snapshot(forked_snapshot),
                staging_dir,
                manager=self,
            )
            session._write_snapshot(forked_snapshot)
            staging_dir.replace(session_dir)
            session.directory = session_dir
            return session
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            self.release_session(session_id)
            raise

    def get_session(self, name: str) -> Session | None:
        """Get a session without making it current."""
        session_id = _normalize_explicit_session_id(name)
        if session_id is None:
            logger.warning(
                "Invalid session id provided for lookup",
                data={"session_id": name},
            )
            return None
        session_dir = self.base_dir / session_id
        metadata_file = session_dir / "session.json"

        if not session_dir.is_dir() or not metadata_file.exists():
            return None

        try:
            with metadata_file.open(encoding="utf-8") as f:
                data = json.load(f)
                info = SessionInfo.from_dict(data)
            return Session(info, session_dir, manager=self)
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def _prune_sessions(self, max_sessions: int | None = None) -> None:
        """Remove older sessions beyond the rolling window."""
        if max_sessions is None:
            max_sessions = get_session_history_window()
        if max_sessions <= 0:
            return
        sessions = self.list_sessions()
        if len(sessions) <= max_sessions:
            return
        current_name = self._current_session.info.name if self._current_session else None
        for session_info in sessions[max_sessions:]:
            if current_name and session_info.name == current_name:
                continue
            if is_session_pinned(session_info):
                continue
            self.delete_session(session_info.name)

    def _resolve_session_name(self, name: str | None) -> str | None:
        """Resolve a session name or ordinal index into a session id."""
        session_name = strip_to_none(name)
        if not session_name:
            return None
        if session_name.isdigit():
            ordinal = int(session_name)
            if ordinal > 0:
                sessions = apply_session_window(self.list_sessions())
                if ordinal <= len(sessions):
                    return sessions[ordinal - 1].name
        sessions = self.list_sessions()
        if any(session.name == session_name for session in sessions):
            return session_name
        matches = [
            session.name
            for session in sessions
            if session.name.endswith(f"-{session_name}") and SESSION_ID_PATTERN.match(session.name)
        ]
        if len(matches) == 1:
            return matches[0]
        for session in sessions:
            metadata = session.metadata
            if isinstance(metadata, dict) and metadata.get("acp_session_id") == session_name:
                return session.name
        return session_name

    def resolve_session_name(self, name: str | None) -> str | None:
        """Public wrapper to resolve a session identifier or ordinal index."""
        return self._resolve_session_name(name)

    def generate_session_id(self) -> str:
        """Generate a unique session identifier without creating a session."""
        session_id = self._generate_session_id()
        session_dir = self.base_dir / session_id
        while session_dir.exists():
            session_id = self._generate_session_id()
            session_dir = self.base_dir / session_id
        return session_id

    def _generate_session_id(self) -> str:
        """Generate a secure session identifier."""
        timestamp = datetime.now().strftime(SESSION_TIMESTAMP_FORMAT)
        random_suffix = "".join(
            secrets.choice(SESSION_ID_ALPHABET) for _ in range(SESSION_ID_LENGTH)
        )
        return f"{timestamp}-{random_suffix}"

    def _copy_history_file(self, src_path: pathlib.Path, dest_dir: pathlib.Path) -> str:
        dest_name = src_path.name
        dest_path = dest_dir / dest_name
        if dest_path.exists():
            stem = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while dest_path.exists():
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = dest_dir / dest_name
                counter += 1
        shutil.copy2(src_path, dest_path)
        return dest_name

    def _load_authoritative_snapshot(self, session: Session) -> SessionSnapshot:
        metadata_file = session.directory / "session.json"
        try:
            with metadata_file.open(encoding="utf-8") as handle:
                return load_session_snapshot(json.load(handle))
        except Exception as exc:
            logger.warning(
                "Failed to load typed session snapshot; using compatibility projection",
                data={"session": session.info.name, "error": str(exc)},
            )
            return snapshot_from_session_info(session.info)

    def _copy_fork_history_files(
        self,
        *,
        source: Session,
        snapshot: SessionSnapshot,
        dest_dir: pathlib.Path,
    ) -> dict[str, str]:
        copied_history_files: dict[str, str] = {}
        for agent_snapshot in snapshot.continuation.agents.values():
            history_file = agent_snapshot.history_file
            if history_file is None or history_file in copied_history_files:
                continue

            src_path = source.directory / history_file
            if not src_path.exists():
                logger.warning(
                    "Session history file missing during fork",
                    data={"session": source.info.name, "file": history_file},
                )
                continue

            copied_history_files[history_file] = self._copy_history_file(src_path, dest_dir)
        return copied_history_files

    def _acquire_owner(self, session_id: str) -> SessionOwnerLease:
        existing = self._owner_leases.get(session_id)
        if existing is not None:
            return existing
        lease = self._locks.acquire_owner(
            session_id,
            started_at=self._started_at,
            surface=self._surface,
            token=secrets.token_urlsafe(24),
        )
        self._owner_leases[session_id] = lease
        return lease

    def _activate(self, session: Session) -> None:
        current = self._current_session
        if (
            not self._retain_multiple
            and current is not None
            and current.info.name != session.info.name
        ):
            self.release_session(current.info.name)
        self._current_session = session

    def owns_session(self, session_id: str) -> bool:
        """Return whether this manager retains the session owner lease."""
        return session_id in self._owner_leases

    def release_session(self, session_id: str) -> None:
        """Release one retained session owner lease."""
        lease = self._owner_leases.pop(session_id, None)
        if lease is not None:
            try:
                lease.release()
            except Exception as exc:
                logger.warning(
                    "Failed to release session owner lease",
                    data={"session": session_id, "error": str(exc)},
                )
        if self._current_session is not None and self._current_session.info.name == session_id:
            self._current_session = None

    def close(self) -> None:
        """Release all retained owner leases."""
        for session_id in list(self._owner_leases):
            self.release_session(session_id)

    def set_current_session(self, session: Session) -> None:
        """Set the current session."""
        self._activate(session)


_session_manager: SessionManager | None = None


def reset_session_manager() -> None:
    """Reset the global session manager (forces reinitialization)."""
    global _session_manager
    if _session_manager is not None:
        _session_manager.close()
    _session_manager = None


def set_session_manager(manager: SessionManager) -> None:
    """Set the process-level session manager for legacy consumers."""
    global _session_manager
    if _session_manager is not None and _session_manager is not manager:
        _session_manager.close()
    _session_manager = manager


def get_session_manager(
    *,
    cwd: pathlib.Path | None = None,
    home_override: str | pathlib.Path | None = None,
    respect_env_override: bool = True,
) -> SessionManager:
    """Return the registered process-level session manager.

    Session managers are created by explicit runtime/session boundaries. This
    accessor exists only for legacy paths that receive that established manager
    through process context.
    """
    explicit_cwd = cwd is not None
    resolved_cwd = cwd.resolve() if cwd is not None else pathlib.Path.cwd().resolve()
    home_override = _session_home_override(
        cwd=resolved_cwd,
        explicit_cwd=explicit_cwd,
        home_override=home_override,
        respect_env_override=respect_env_override,
    )
    expected_paths = resolve_home_paths(cwd=resolved_cwd, override=home_override)
    if _session_manager is None:
        raise RuntimeError(
            "No active session manager has been registered. Create a SessionManager at the "
            "runtime/session boundary and pass it through Context or CommandContext."
        )
    if _session_manager.base_dir != expected_paths.sessions:
        raise RuntimeError(
            "Active session manager does not match the requested session store. Pass the "
            "correct SessionManager explicitly instead of resolving a new one."
        )
    if explicit_cwd and _session_manager.workspace_dir != resolved_cwd:
        raise RuntimeError(
            "Active session manager workspace does not match the requested cwd. Pass the "
            "correct SessionManager explicitly instead of switching the global manager."
        )
    return _session_manager


def get_active_session_manager() -> SessionManager:
    """Return the process-level session manager without resolving a store."""
    if _session_manager is None:
        raise RuntimeError("No active session manager has been registered")
    return _session_manager
