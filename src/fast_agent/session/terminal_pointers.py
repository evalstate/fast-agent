"""Per-terminal session pointers for nesting ``hc`` child sessions.

Each ``fast-agent`` process bound to a local ``herdr-core`` terminal publishes a
small JSON pointer under the user's runtime directory. Child ``fast-agent``
processes spawned via ``hc run`` resolve their parent terminal via passive
``hc get`` and look up this pointer, then validate liveness against the parent
session's owner lease and persisted terminal attachment.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class HcTerminalPointer:
    """Published location of the active session bound to an ``hc`` terminal."""

    core_id: str
    terminal_id: str
    home_dir: Path
    session_id: str
    session_dir: Path
    agent_name: str | None = None


@dataclass(frozen=True, slots=True)
class HcParentSessionTarget:
    """Validated parent session target for nesting an ``hc`` child worker."""

    home_dir: Path
    parent_session_id: str
    parent_session_dir: Path
    parent_agent_name: str
    hc_terminal_id: str
    hc_name: str | None = None
    parent_terminal_id: str | None = None
    core_id: str | None = None


def _is_safe_segment(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and bool(_SAFE_SEGMENT_RE.match(value))


def _ensure_private_dir(path: Path) -> Path | None:
    """Ensure ``path`` exists and, on POSIX, is owned by the current uid."""
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        logger.debug(
            "Failed to create hc terminal pointer dir", data={"path": str(path), "error": str(exc)}
        )
        return None

    if os.name == "posix":
        try:
            st = path.stat()
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
                logger.warning(
                    "Refusing hc terminal pointer dir not owned by current user",
                    data={"path": str(path)},
                )
                return None
        except OSError:
            return None
    return path


def _pointer_root() -> Path | None:
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_runtime:
        base = _ensure_private_dir(Path(xdg_runtime) / "fast-agent")
    else:
        uid_suffix = str(os.getuid()) if os.name == "posix" else "user"
        base = _ensure_private_dir(Path(tempfile.gettempdir()) / f"fast-agent-{uid_suffix}")
    if base is None:
        return None
    return _ensure_private_dir(base / "hc-terminals")


def terminal_pointer_path(core_id: str, terminal_id: str) -> Path | None:
    """Return the pointer file path for ``(core_id, terminal_id)``, or None if unsafe."""
    if not _is_safe_segment(core_id) or not _is_safe_segment(terminal_id):
        return None
    root = _pointer_root()
    if root is None:
        return None
    core_dir = _ensure_private_dir(root / core_id)
    if core_dir is None:
        return None
    return core_dir / f"{terminal_id}.json"


def write_terminal_pointer(pointer: HcTerminalPointer) -> Path | None:
    """Atomically publish the active session pointer for an ``hc`` terminal."""
    target = terminal_pointer_path(pointer.core_id, pointer.terminal_id)
    if target is None:
        return None

    payload = {
        "core_id": pointer.core_id,
        "terminal_id": pointer.terminal_id,
        "home_dir": str(pointer.home_dir.resolve()),
        "session_id": pointer.session_id,
        "session_dir": str(pointer.session_dir.resolve()),
        "agent_name": pointer.agent_name,
    }
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if os.name == "posix":
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
        os.replace(tmp_path, target)
        return target
    except OSError as exc:
        logger.debug(
            "Failed to write hc terminal pointer", data={"path": str(target), "error": str(exc)}
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def read_terminal_pointer(core_id: str, terminal_id: str) -> HcTerminalPointer | None:
    """Read a published terminal pointer, returning None if missing or malformed."""
    target = terminal_pointer_path(core_id, terminal_id)
    if target is None or not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    raw_core = raw.get("core_id")
    raw_terminal = raw.get("terminal_id")
    raw_home = raw.get("home_dir")
    raw_session_id = raw.get("session_id")
    raw_session_dir = raw.get("session_dir")
    raw_agent = raw.get("agent_name")
    if (
        raw_core != core_id
        or raw_terminal != terminal_id
        or not isinstance(raw_home, str)
        or not raw_home
        or not isinstance(raw_session_id, str)
        or not raw_session_id
        or not isinstance(raw_session_dir, str)
        or not raw_session_dir
        or (raw_agent is not None and not isinstance(raw_agent, str))
    ):
        return None

    return HcTerminalPointer(
        core_id=core_id,
        terminal_id=terminal_id,
        home_dir=Path(raw_home),
        session_id=raw_session_id,
        session_dir=Path(raw_session_dir),
        agent_name=raw_agent,
    )


def remove_terminal_pointer(
    core_id: str,
    terminal_id: str,
    *,
    expected_session_id: str | None = None,
) -> None:
    """Best-effort removal of a terminal pointer on clean session close."""
    target = terminal_pointer_path(core_id, terminal_id)
    if target is None or not target.exists():
        return
    if expected_session_id is not None:
        current = read_terminal_pointer(core_id, terminal_id)
        if current is not None and current.session_id != expected_session_id:
            return
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
