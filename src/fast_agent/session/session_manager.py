"""
Session management for fast-agent.

Provides automatic saving and loading of conversation sessions in the fast-agent environment.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import secrets
import shutil
import socket
import string
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from fast_agent.core.logging.logger import get_logger
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)

SESSION_ID_LENGTH = 6
SESSION_ID_ALPHABET = string.ascii_letters + string.digits
SESSION_TIMESTAMP_FORMAT = "%y%m%d%H%M"
SESSION_ID_PATTERN = re.compile(
    rf"^(?:[A-Za-z0-9]{{{SESSION_ID_LENGTH}}}|\d{{10}}-[A-Za-z0-9]{{{SESSION_ID_LENGTH}}})$"
)
SESSION_LOCK_FILENAME = ".session.lock"
SESSION_LOCK_STALE_SECONDS = 300
HISTORY_PREFIX = "history_"
HISTORY_SUFFIX = ".json"
HISTORY_PREVIOUS_SUFFIX = "_previous.json"


def _normalized_environment_override(cwd: pathlib.Path) -> str | None:
    """Return ENVIRONMENT_DIR as an absolute path string when set."""
    override = os.getenv("ENVIRONMENT_DIR")
    if not override:
        return None

    path = pathlib.Path(override).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()
    else:
        path = path.resolve()

    normalized = str(path)
    if normalized != override:
        os.environ["ENVIRONMENT_DIR"] = normalized
    return normalized


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
        if agent_name.endswith("_previous"):
            agent_name = agent_name[: -len("_previous")]
        return agent_name or "agent"
    return pathlib.Path(filename).stem


def _first_user_preview(
    messages: list["PromptMessageExtended"], limit: int = 240
) -> str | None:
    for message in messages:
        if message.role != "user":
            continue
        if getattr(message, "is_template", False):
            continue
        text = message.all_text() or message.first_text() or ""
        text = " ".join(text.split())
        if not text:
            return None
        return text[:limit]
    return None


def get_session_history_window() -> int:
    """Return the configured session history window size."""
    try:
        from fast_agent.config import get_settings

        settings = get_settings()
        value = getattr(settings, "session_history_window", 20)
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
        return cls(
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            history_files=data.get("history_files", []),
            metadata=data.get("metadata", {}),
        )

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


class Session:
    """Represents a single conversation session."""

    def __init__(self, info: SessionInfo, directory: pathlib.Path) -> None:
        """Initialize session."""
        self.info = info
        self.directory = directory
        self._dirty = False


    async def save_history(self, agent: AgentProtocol, filename: str | None = None) -> str:
        """Save agent history to this session."""
        from fast_agent.history.history_exporter import HistoryExporter

        self.info.last_activity = datetime.now()
        self._dirty = True

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
            )
            filename = current_filename
        else:
            filepath = self.directory / filename
            result = await HistoryExporter.save(agent, str(filepath))

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
        else:
            if filename not in self.info.history_files:
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

        self._save_metadata()
        return result

    async def _save_rotating_history(
        self,
        agent: AgentProtocol,
        *,
        current_filename: str,
        previous_filename: str,
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

            await HistoryExporter.save(agent, str(temp_path))

            if current_path.exists():
                os.replace(current_path, previous_path)
            os.replace(temp_path, current_path)
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
        """Save session metadata."""
        metadata_file = self.directory / "session.json"
        with self._metadata_lock():
            self._atomic_write_json(metadata_file, self.info.to_dict())
        self._dirty = False

    def set_pinned(self, pinned: bool) -> None:
        """Pin or unpin the session to prevent auto-pruning."""
        if pinned:
            self.info.metadata["pinned"] = True
        else:
            self.info.metadata.pop("pinned", None)
        self._save_metadata()

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
            os.replace(temp_path, path)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    logger.warning(
                        "Failed to clean up session metadata temp file",
                        data={"path": str(temp_path)},
                    )

    @contextlib.contextmanager
    def _metadata_lock(self):
        lock_path = self.directory / SESSION_LOCK_FILENAME
        acquired = False
        existing_info: dict[str, Any] | None = None
        lock_payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            acquired = self._try_acquire_lock(lock_path, lock_payload)
        except Exception as exc:
            logger.warning(
                "Failed to acquire session metadata lock",
                data={"session": self.info.name, "error": str(exc)},
            )

        if not acquired:
            existing_info = self._read_lock_info(lock_path)
            logger.warning(
                "Session metadata lock already held; proceeding without exclusive lock",
                data={
                    "session": self.info.name,
                    "lock_path": str(lock_path),
                    "locked_by": existing_info,
                },
            )

        try:
            yield
        finally:
            if acquired:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "Failed to release session metadata lock",
                        data={"session": self.info.name, "error": str(exc)},
                    )

    def _try_acquire_lock(self, lock_path: pathlib.Path, payload: dict[str, Any]) -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return self._try_replace_stale_lock(lock_path, payload)
        except FileNotFoundError:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def _try_replace_stale_lock(self, lock_path: pathlib.Path, payload: dict[str, Any]) -> bool:
        try:
            mtime = lock_path.stat().st_mtime
        except FileNotFoundError:
            return self._try_acquire_lock(lock_path, payload)
        except Exception:
            return False

        if (time.time() - mtime) < SESSION_LOCK_STALE_SECONDS:
            return False

        try:
            lock_path.unlink()
        except Exception:
            return False
        return self._try_acquire_lock(lock_path, payload)

    def _read_lock_info(self, lock_path: pathlib.Path) -> dict[str, Any] | None:
        try:
            with open(lock_path, encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else None
        except Exception:
            return None

    def get_history_file(self, filename: str) -> pathlib.Path:
        """Get path to a history file."""
        return self.directory / filename

    def delete(self) -> None:
        """Delete this session."""
        if self.directory.exists():
            shutil.rmtree(self.directory)

    def set_title(self, title: str) -> None:
        """Set a user-friendly title for this session."""
        self.info.metadata["title"] = title
        self.info.last_activity = datetime.now()
        self._save_metadata()

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


class SessionManager:
    """Manages conversation sessions stored in the fast-agent environment."""

    def __init__(self, *, cwd: pathlib.Path | None = None) -> None:
        """Initialize session manager."""
        base = (cwd or pathlib.Path.cwd()).resolve()
        env_override = _normalized_environment_override(base)
        env_paths = resolve_environment_paths(cwd=base, override=env_override)
        self.base_dir = env_paths.sessions
        self.base_dir.mkdir(parents=True, exist_ok=True)
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

        # Create session directory
        session_dir = self.base_dir / session_id
        while session_dir.exists():
            session_id = self._generate_session_id()
            session_dir = self.base_dir / session_id

        session_dir.mkdir(parents=True)

        # Create session info
        now = datetime.now()
        info = SessionInfo(
            name=session_id,
            created_at=now,
            last_activity=now,
            history_files=[],
            metadata=session_metadata,
        )

        session = Session(info, session_dir)
        session._save_metadata()
        self._current_session = session
        self._prune_sessions()
        logger.info(f"Created new session: {session_id}")
        return session

    def create_session_with_id(self, session_id: str, metadata: dict | None = None) -> Session:
        """Create or load a session using the provided id."""
        requested_id = (session_id or "").strip()
        session_metadata = dict(metadata or {})
        if requested_id:
            session_metadata.setdefault("acp_session_id", requested_id)

        if not requested_id or pathlib.Path(requested_id).name != requested_id:
            logger.warning(
                "Invalid session id provided; falling back to generated id",
                data={"session_id": session_id},
            )
            return self.create_session(metadata=session_metadata)

        session_dir = self.base_dir / requested_id
        if session_dir.exists():
            session = self.load_session(requested_id)
            if session:
                if session.info.metadata.get("acp_session_id") != requested_id:
                    session.info.metadata["acp_session_id"] = requested_id
                    session._save_metadata()
                return session

        session_dir.mkdir(parents=True, exist_ok=False)
        now = datetime.now()
        info = SessionInfo(
            name=requested_id,
            created_at=now,
            last_activity=now,
            history_files=[],
            metadata=session_metadata,
        )
        session = Session(info, session_dir)
        session._save_metadata()
        self._current_session = session
        self._prune_sessions()
        logger.info(f"Created new session: {requested_id}")
        return session

    def list_sessions(self) -> list[SessionInfo]:
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
                    with open(metadata_file) as f:
                        data = json.load(f)
                        info = SessionInfo.from_dict(data)
                        sessions.append(info)
                except Exception as e:
                    logger.warning(f"Failed to load session metadata from {metadata_file}: {e}")

        sessions.sort(key=lambda info: info.last_activity, reverse=True)
        return sessions

    def load_session(self, name: str) -> Session | None:
        """Load an existing session."""
        session_dir = self.base_dir / name
        metadata_file = session_dir / "session.json"

        if not session_dir.is_dir() or not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                data = json.load(f)
                info = SessionInfo.from_dict(data)

            session = Session(info, session_dir)
            session.info.last_activity = datetime.now()
            session._save_metadata()
            self._current_session = session
            logger.info(f"Loaded session: {name}")
            return session
        except Exception as e:
            logger.error(f"Failed to load session {name}: {e}")
            return None

    def delete_session(self, name: str) -> bool:
        """Delete a session."""
        session_dir = self.base_dir / name

        if not session_dir.is_dir():
            return False

        try:
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {name}")
            if self._current_session and self._current_session.info.name == name:
                self._current_session = None
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {name}: {e}")
            return False

    async def save_current_session(
        self, agent: AgentProtocol, filename: str | None = None
    ) -> str | None:
        """Save history to the current session."""
        if self._current_session and not self._current_session.directory.exists():
            logger.warning(
                "Current session directory is missing; creating a replacement session",
                data={"session": self._current_session.info.name},
            )
            self._current_session = None

        if not self._current_session:
            # Auto-create a session if none exists
            agent_name = getattr(agent, "name", None)
            metadata: dict[str, Any] = {}
            if agent_name:
                metadata["agent_name"] = agent_name
            agent_config = getattr(agent, "config", None)
            model_name = getattr(agent_config, "model", None) if agent_config else None
            if model_name:
                metadata["model"] = model_name
            self.create_session(metadata=metadata or None)
            logger.warning(
                "save_current_session created a fallback session; "
                "the session hook should have created one earlier",
                data={"agent_name": agent_name},
            )

        assert self._current_session is not None
        return await self._current_session.save_history(agent, filename)

    def load_latest_session(self) -> Session | None:
        """Load the most recently used session."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        return self.load_session(sessions[0].name)

    def resume_session(
        self, agent: AgentProtocol, name: str | None = None
    ) -> tuple[Session, pathlib.Path | None, list[str]] | None:
        """Resume a session and load its latest history into the agent."""
        session_name = self._resolve_session_name(name)
        session = self.load_latest_session() if session_name is None else self.load_session(session_name)
        if not session:
            return None

        history_path = session.latest_history_path(getattr(agent, "name", None))
        notices: list[str] = []
        if history_path and history_path.exists():
            from fast_agent.mcp.prompts.prompt_load import load_history_into_agent

            notice = load_history_into_agent(agent, history_path)
            if notice:
                notices.append(notice)

        return session, history_path, notices

    def resume_session_agents(
        self,
        agents: Mapping[str, AgentProtocol],
        name: str | None = None,
        default_agent_name: str | None = None,
    ) -> ResumeSessionAgentsResult | None:
        """Resume a session and load histories for all known agents."""
        session_name = self._resolve_session_name(name)
        session = self.load_latest_session() if session_name is None else self.load_session(session_name)
        if not session:
            return None

        history_map = session.info.metadata.get("last_history_by_agent")
        loaded: dict[str, pathlib.Path] = {}
        missing_agents: list[str] = []
        notices: list[str] = []

        if isinstance(history_map, dict) and history_map:
            for agent_name, filename in history_map.items():
                if agent_name not in agents:
                    missing_agents.append(agent_name)
                    continue
                if not filename:
                    continue
                history_path = session.directory / filename
                if history_path.exists():
                    from fast_agent.mcp.prompts.prompt_load import load_history_into_agent

                    try:
                        notice = load_history_into_agent(agents[agent_name], history_path)
                        if notice:
                            notices.append(notice)
                        loaded[agent_name] = history_path
                    except Exception as exc:
                        logger.warning(
                            "Failed to load session history file",
                            data={
                                "session": session.info.name,
                                "agent": agent_name,
                                "file": str(history_path),
                                "error": str(exc),
                            },
                        )
                else:
                    logger.warning(
                        "Session history file missing",
                        data={"session": session.info.name, "agent": agent_name, "file": filename},
                    )
        else:
            fallback_agent = None
            if default_agent_name and default_agent_name in agents:
                fallback_agent = agents[default_agent_name]
            elif agents:
                fallback_agent = next(iter(agents.values()))
            if fallback_agent:
                history_path = session.latest_history_path(getattr(fallback_agent, "name", None))
                if history_path and history_path.exists():
                    from fast_agent.mcp.prompts.prompt_load import load_history_into_agent

                    try:
                        notice = load_history_into_agent(fallback_agent, history_path)
                        if notice:
                            notices.append(notice)
                        loaded[fallback_agent.name] = history_path
                    except Exception as exc:
                        logger.warning(
                            "Failed to load fallback session history file",
                            data={
                                "session": session.info.name,
                                "agent": fallback_agent.name,
                                "file": str(history_path),
                                "error": str(exc),
                            },
                        )

        if missing_agents:
            logger.warning(
                "Session metadata references missing agents",
                data={"session": session.info.name, "agents": missing_agents},
            )

        return ResumeSessionAgentsResult(
            session=session,
            loaded=loaded,
            missing_agents=missing_agents,
            usage_notices=notices,
        )

    def fork_current_session(self, title: str | None = None) -> Session | None:
        """Fork the current or latest session into a new session."""
        source = self._current_session or self.load_latest_session()
        if not source:
            return None

        source_metadata = source.info.metadata or {}
        new_metadata: dict[str, Any] = {"forked_from": source.info.name}
        if title:
            new_metadata["title"] = title
        elif isinstance(source_metadata.get("title"), str):
            new_metadata["title"] = source_metadata["title"]
        if isinstance(source_metadata.get("first_user_preview"), str):
            new_metadata["first_user_preview"] = source_metadata["first_user_preview"]

        new_session = self.create_session(metadata=new_metadata)
        history_map = source_metadata.get("last_history_by_agent")
        new_history_map: dict[str, str] = {}

        if isinstance(history_map, dict) and history_map:
            for agent_name, filename in history_map.items():
                if not filename:
                    continue
                src_path = source.directory / filename
                if not src_path.exists():
                    logger.warning(
                        "Session history file missing",
                        data={"session": source.info.name, "agent": agent_name, "file": filename},
                    )
                    continue
                dest_name = self._copy_history_file(src_path, new_session.directory)
                new_session.info.history_files.append(dest_name)
                new_history_map[agent_name] = dest_name
        else:
            for filename in source.info.history_files:
                src_path = source.directory / filename
                if not src_path.exists():
                    continue
                dest_name = self._copy_history_file(src_path, new_session.directory)
                new_session.info.history_files.append(dest_name)

        if new_history_map:
            new_session.info.metadata["last_history_by_agent"] = new_history_map

        new_session._save_metadata()
        return new_session

    def get_session(self, name: str) -> Session | None:
        """Get a session without making it current."""
        session_dir = self.base_dir / name
        metadata_file = session_dir / "session.json"

        if not session_dir.is_dir() or not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                data = json.load(f)
                info = SessionInfo.from_dict(data)
            return Session(info, session_dir)
        except Exception as e:
            logger.error(f"Failed to get session {name}: {e}")
            return None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize session name for filesystem safety."""
        return _sanitize_component(name)

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
        session_name = None if name in (None, "") else name
        if session_name is None:
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
            if session.name.endswith(f"-{session_name}")
            and SESSION_ID_PATTERN.match(session.name)
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
        random_suffix = "".join(secrets.choice(SESSION_ID_ALPHABET) for _ in range(SESSION_ID_LENGTH))
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

    def set_current_session(self, session: Session) -> None:
        """Set the current session."""
        self._current_session = session


_session_manager: SessionManager | None = None


def reset_session_manager() -> None:
    """Reset the global session manager (forces reinitialization)."""
    global _session_manager
    _session_manager = None


def get_session_manager(*, cwd: pathlib.Path | None = None) -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    resolved_cwd = cwd.resolve() if cwd is not None else pathlib.Path.cwd().resolve()
    env_override = _normalized_environment_override(resolved_cwd)
    expected_paths = resolve_environment_paths(cwd=resolved_cwd, override=env_override)
    if _session_manager is None:
        _session_manager = SessionManager(cwd=cwd)
        return _session_manager
    if _session_manager.base_dir != expected_paths.sessions:
        _session_manager = SessionManager(cwd=cwd)
    return _session_manager
