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

from fast_agent.utils.type_narrowing import is_str_object_dict

_COMMAND_TIMEOUT_SECONDS = 2.0
_CACHE_SECONDS = 2.0
_HC_MESSAGE = re.compile(
    r"\s*<hc(?:\s+from=\"(?P<sender>[^\"]*)\"[^>]*|\s[^>]*)?>(?P<body>.*)", re.S
)
_HC_NOTICE = re.compile(r"\s*(?P<terminal>\S+)(?:\s+\S+)?\s+(?P<event>exited|blocked|error)\b")


@dataclass(frozen=True, slots=True)
class HcChild:
    address: str
    name: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.address})" if self.name else self.address


@dataclass(frozen=True, slots=True)
class HcAttachments:
    """Unfinished children of this terminal, by what they mean for the session."""

    resuming: tuple[HcChild, ...] = ()
    services: int = 0


_NONE = HcAttachments()
_current = _NONE
_read_at = 0.0
_waiting_since: float | None = None


def hc_terminal_id() -> str | None:
    """This terminal's hc address, when running inside herdr-core."""
    terminal = os.environ.get("HC_TERMINAL_ID", "").strip()
    return terminal if terminal and os.environ.get("HC_TERMINAL_SOCKET") else None


def _hc_json(executable: str, *args: str) -> dict[str, object] | None:
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
        payload = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return payload if is_str_object_dict(payload) else None


def read_hc_attachments() -> HcAttachments:
    """Ask hc which children are unfinished. Blocking; call off the event loop."""
    if hc_terminal_id() is None or (executable := shutil.which("hc")) is None:
        return _NONE
    # hc's own accounting: an agent with a delivered task counts as unfinished
    # until its turn advances, and a plain program until it exits.
    waited = _hc_json(executable, "wait", "--children", "all", "--timeout", "0")
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
        detail = _hc_json(executable, "get", address) or {}
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


def unwrap_hc_message(text: str) -> str:
    """The body of an hc message without its envelope, for display only."""
    match = _HC_MESSAGE.match(text)
    if match is None:
        return text
    return match.group("body").rstrip().removesuffix("</hc>").strip()


def hc_message_heading(texts: list[str]) -> str | None:
    """A panel heading for input that hc delivered, or None if a person typed it."""
    matches = [_HC_MESSAGE.match(text) for text in texts]
    if not matches or any(match is None for match in matches):
        return None
    if len(texts) > 1:
        return f"hc · {len(texts)} messages"
    match = matches[0]
    assert match is not None
    if sender := match.group("sender"):
        return f"hc · {sender}"
    notice = _HC_NOTICE.match(match.group("body"))
    return f"hc · {notice.group('terminal')} {notice.group('event')}" if notice else "hc"
