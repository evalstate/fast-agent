"""What an idle session inside herdr-core (``hc``) is still attached to.

An idle fast-agent that hc will wake (a child exits or reports) otherwise looks
the same as a finished one. This reads hc's own accounting of this terminal's
children and sorts the unfinished ones into those that will resume the session
and long-lived services. fast-agent's own background processes are counted by
the caller: nothing resumes the session when they end.

All hc output is parsed here. Reading never raises and is a no-op outside hc.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.utils.type_narrowing import is_str_object_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

_COMMAND_TIMEOUT_SECONDS = 2.0
_LIST_COMMAND_TIMEOUT_SECONDS = 4.5
_CACHE_SECONDS = 2.0
_RESTART_EXIT_REASONS = frozenset(
    {"stopped_by_restart", "holder_lost", "server_stop", "host_shutdown"}
)
_CORE_ID_CACHE: dict[tuple[str, str], str] = {}
_HC_MESSAGE = re.compile(
    r"\s*<hc(?:\s+from=\"(?P<sender>[^\"]*)\"[^>]*|\s[^>]*)?>(?P<body>.*)", re.S
)
_HC_BLOCK = re.compile(
    r"<hc(?:\s+from=\"(?P<sender>[^\"]*)\"[^>]*|\s[^>]*)?>(?P<body>.*?)</hc>", re.S
)
_HC_NOTICE = re.compile(
    r"\s*(?P<terminal>\S+)(?:\s+\S+)?\s+(?P<event>exited|is blocked|blocked|hit an error|error)\b"
)


@dataclass(frozen=True, slots=True)
class HcSelfInfo:
    """Passive metadata for the current ``hc`` terminal."""

    terminal_id: str
    parent: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class HcChild:
    address: str
    name: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.address})" if self.name else self.address


@dataclass(frozen=True, slots=True)
class HcEnvironment:
    """Resolved herdr-core terminal context for the current process."""

    terminal_id: str
    socket_path: str
    executable: str


@dataclass(frozen=True, slots=True)
class HcAttachments:
    """Unfinished children of this terminal, by what they mean for the session."""

    resuming: tuple[HcChild, ...] = ()
    services: int = 0


@dataclass(frozen=True, slots=True)
class HcTerminalChildren:
    """Direct local children of a terminal, read passively without acknowledging notes."""

    live: tuple[HcChild, ...] = ()
    stopped_by_restart: tuple[HcChild, ...] = ()


_NONE = HcAttachments()
_current = _NONE
_read_at = 0.0
_waiting_since: float | None = None


def hc_terminal_id() -> str | None:
    """This terminal's hc address, when running inside herdr-core."""
    terminal = os.environ.get("HC_TERMINAL_ID", "").strip()
    return terminal if terminal and os.environ.get("HC_TERMINAL_SOCKET") else None


def hc_environment() -> HcEnvironment | None:
    """Return the active herdr-core environment when running inside an hc terminal."""
    terminal = os.environ.get("HC_TERMINAL_ID", "").strip()
    socket_path = os.environ.get("HC_TERMINAL_SOCKET", "").strip()
    if not terminal or not socket_path:
        return None
    executable = shutil.which("hc")
    if executable is None:
        return None
    return HcEnvironment(terminal_id=terminal, socket_path=socket_path, executable=executable)


def _hc_subprocess_env(hc_env: HcEnvironment, *, strip_caller: bool) -> dict[str, str]:
    # Pin HC_SOCKET to this terminal's core socket so an ambient HC_SOCKET/HC_SESSION
    # never routes recovery or polling to a different core. Keep HC_SESSION intact so
    # named sessions never fall back to the default state directory.
    env = dict(os.environ)
    env["HC_SOCKET"] = hc_env.socket_path
    if strip_caller:
        env.pop("HC_TERMINAL_ID", None)
    return env


def _hc_json(
    hc_env: HcEnvironment,
    *args: str,
    strip_caller: bool = False,
    timeout_seconds: float = _COMMAND_TIMEOUT_SECONDS,
) -> dict[str, object] | None:
    try:
        result = subprocess.run(
            [hc_env.executable, *args],
            check=False,
            env=_hc_subprocess_env(hc_env, strip_caller=strip_caller),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
        payload = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return payload if is_str_object_dict(payload) else None


def read_hc_self_info() -> HcSelfInfo | None:
    """Passively query ``hc get $HC_TERMINAL_ID`` for this terminal's parent and name."""
    env = hc_environment()
    if env is None:
        return None
    detail = _hc_json(env, "get", env.terminal_id, strip_caller=True)
    if not detail:
        return None
    raw_parent = detail.get("parent")
    raw_name = detail.get("name")
    return HcSelfInfo(
        terminal_id=env.terminal_id,
        parent=raw_parent.strip() if isinstance(raw_parent, str) and raw_parent.strip() else None,
        name=raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None,
    )


def read_hc_core_id() -> str | None:
    """Return the current herdr-core instance ID, or None if unavailable."""
    env = hc_environment()
    if env is None:
        return None
    cache_key = (env.executable, env.socket_path)
    cached = _CORE_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    status = _hc_json(env, "server", "status", "--json", strip_caller=True)
    if status is None:
        return None
    core = status.get("core")
    if is_str_object_dict(core):
        core_id = core.get("id")
        if isinstance(core_id, str) and core_id:
            _CORE_ID_CACHE[cache_key] = core_id
            return core_id
    top_id = status.get("id")
    if isinstance(top_id, str) and top_id:
        _CORE_ID_CACHE[cache_key] = top_id
        return top_id
    return None


def read_hc_terminal_children(terminal_id: str | None = None) -> HcTerminalChildren | None:
    """Passively list direct local children of ``terminal_id`` without calling ``saw``.

    Returns None when hc cannot be queried so callers do not confuse a query
    failure with an empty child set.
    """
    env = hc_environment()
    if env is None:
        return None
    parent_id = terminal_id or env.terminal_id
    listed = _hc_json(
        env,
        "ls",
        "--all",
        "--json",
        strip_caller=True,
        timeout_seconds=_LIST_COMMAND_TIMEOUT_SECONDS,
    )
    terminals = listed.get("terminals") if listed is not None else None
    if not isinstance(terminals, list):
        return None

    live: list[HcChild] = []
    stopped_by_restart: list[HcChild] = []
    for entry in terminals:
        if not is_str_object_dict(entry):
            continue
        if entry.get("parent") != parent_id:
            continue
        address = entry.get("address")
        if not isinstance(address, str) or not address or "/" in address:
            continue
        raw_name = entry.get("name")
        child = HcChild(
            address=address, name=raw_name if isinstance(raw_name, str) and raw_name else None
        )
        state = entry.get("state")
        if state == "live":
            live.append(child)
        elif state == "finished" and entry.get("reason") in _RESTART_EXIT_REASONS:
            stopped_by_restart.append(child)
    return HcTerminalChildren(
        live=tuple(live),
        stopped_by_restart=tuple(stopped_by_restart),
    )


def remove_hc_terminals(addresses: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove finished local hc terminals by exact address. Returns (removed, failed)."""
    env = hc_environment()
    if env is None:
        return [], list(addresses)
    removed: list[str] = []
    failed: list[str] = []
    for address in addresses:
        if not address or "/" in address:
            failed.append(address)
            continue
        try:
            result = subprocess.run(
                [env.executable, "rm", "--json", address],
                check=False,
                env=_hc_subprocess_env(env, strip_caller=True),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            failed.append(address)
            continue
        if result.returncode == 0:
            removed.append(address)
        else:
            failed.append(address)
    return removed, failed


def read_hc_attachments() -> HcAttachments:
    """Ask hc which children are unfinished without acknowledging parent notices.

    Passing the explicit parent terminal with caller identity stripped lets hc
    compute real busy/ended state (including pending launch tasks and remote
    children) while preventing ``core.saw()`` from acknowledging unseen notes.
    Blocking; call off the event loop.
    """
    env = hc_environment()
    if env is None:
        return _NONE
    waited = _hc_json(
        env, "wait", env.terminal_id, "--children", "all", "--timeout", "0", strip_caller=True
    )
    children = waited.get("children") if waited is not None else None
    if not isinstance(children, list):
        return _NONE
    resuming: list[HcChild] = []
    services = 0
    for child in children:
        if not is_str_object_dict(child) or child.get("ended") is not False:
            continue
        address = child.get("terminal")
        if not isinstance(address, str):
            continue
        detail = _hc_json(env, "get", address, strip_caller=True) or {}
        labels = detail.get("labels")
        name = detail.get("name")
        if is_str_object_dict(labels) and labels.get("role") == "service":
            services += 1
        elif (
            detail.get("notify") is True  # hc sends a note when it exits
            or detail.get("reported") is True  # an agent: it reports, or hc does for it
            or detail.get("first_message") is not None
        ):
            resuming.append(HcChild(address, name if isinstance(name, str) else None))
    return HcAttachments(tuple(resuming), services)


async def refresh_hc_attachments(*, reuse_recent: bool = False) -> HcAttachments:
    """Update the shared snapshot without blocking the event loop.

    ``reuse_recent`` accepts a snapshot read moments ago, e.g. at the turn end
    that preceded this prompt.
    """
    global _read_at

    if hc_terminal_id() is None:
        _update_current(_NONE)
    elif not reuse_recent or time.monotonic() - _read_at >= _CACHE_SECONDS:
        _update_current(await asyncio.to_thread(read_hc_attachments))
        _read_at = time.monotonic()
    return _current


def _update_current(attachments: HcAttachments) -> None:
    """Replace the snapshot; the wait clock runs while any child will resume us."""
    global _current, _waiting_since

    _current = attachments
    if not attachments.resuming:
        _waiting_since = None
    elif _waiting_since is None:
        _waiting_since = time.monotonic()


def current_hc_attachments() -> HcAttachments:
    """The last snapshot read, for synchronous renderers such as the toolbar."""
    return _current


def hc_waiting_since() -> float | None:
    """``time.monotonic()`` when this session started waiting on resuming children."""
    return _waiting_since


def format_hc_elapsed(seconds: float) -> str | None:
    """A coarse wait duration, or None for the first minute."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return None
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _count(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


def format_turn_end_lines(attachments: HcAttachments, background_processes: int) -> list[str]:
    """Scrollback lines saying what will, and will not, bring this session back."""
    lines: list[str] = []
    resuming = attachments.resuming
    services = (
        f" · {_count(attachments.services, 'service')} running" if attachments.services else ""
    )
    if resuming:
        names = ", ".join(child.label for child in resuming)
        pronoun = "it reports or exits" if len(resuming) == 1 else "they report or exit"
        lines.append(f"⧗ hc · waiting on {names} · resumes when {pronoun}{services}")
    elif services:
        lines.append(f"hc{services}")
    if background_processes > 0:
        pronoun = "it ends" if background_processes == 1 else "they end"
        lines.append(
            f"{_count(background_processes, 'background process', 'background processes')}"
            f" still running; nothing will resume this session when {pronoun}"
        )
    return lines


def format_status_summary(attachments: HcAttachments, background_processes: int) -> str | None:
    """One short phrase for the terminal status message and the toolbar."""
    parts: list[str] = []
    if attachments.resuming:
        names = ", ".join(child.name or child.address for child in attachments.resuming)
        parts.append(f"waiting on {len(attachments.resuming)}: {names}")
    if attachments.services:
        parts.append(_count(attachments.services, "service"))
    if background_processes > 0:
        parts.append(f"{background_processes} bg (no resume)")
    return ", ".join(parts) or None


def is_hc_message(text: str) -> bool:
    """Whether submitted text is an hc peer message or an hc notice."""
    return _HC_MESSAGE.match(text) is not None


def _parse_hc_blocks(text: str) -> list[re.Match[str]]:
    """Parse one or more newline-joined ``<hc>...</hc>`` blocks."""
    stripped = text.strip()
    if not stripped:
        return []
    blocks: list[re.Match[str]] = []
    pos = 0
    for match in _HC_BLOCK.finditer(stripped):
        if stripped[pos : match.start()].strip():
            return []
        blocks.append(match)
        pos = match.end()
    if blocks and not stripped[pos:].strip():
        return blocks
    return []


def _normalize_notice_event(raw: str) -> str:
    if raw == "is blocked":
        return "blocked"
    if raw == "hit an error":
        return "error"
    return raw


def unwrap_hc_message(text: str) -> str:
    """The body of an hc message without its envelope, for display only."""
    blocks = _parse_hc_blocks(text)
    if blocks:
        return "\n\n".join(block.group("body").strip() for block in blocks)
    match = _HC_MESSAGE.match(text)
    if match is None:
        return text
    return match.group("body").rstrip().removesuffix("</hc>").strip()


def hc_message_heading(texts: list[str]) -> str | None:
    """A panel heading for input that hc delivered, or None if a person typed it."""
    all_blocks: list[re.Match[str]] = []
    for text in texts:
        blocks = _parse_hc_blocks(text)
        if blocks:
            all_blocks.extend(blocks)
            continue
        match = _HC_MESSAGE.match(text)
        if match is None:
            return None
        all_blocks.append(match)
    if not all_blocks:
        return None
    if len(all_blocks) > 1:
        return f"hc · {len(all_blocks)} messages"
    match = all_blocks[0]
    if sender := match.group("sender"):
        return f"hc · {sender}"
    notice = _HC_NOTICE.match(match.group("body"))
    if notice:
        event = _normalize_notice_event(notice.group("event"))
        return f"hc · {notice.group('terminal')} {event}"
    return "hc"
