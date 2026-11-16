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
    from collections.abc import Awaitable, Callable

_COMMAND_TIMEOUT_SECONDS = 2.0
_EVENT_SETTLE_SECONDS = 0.3
_EVENT_RETRY_SECONDS = 5.0
_CACHE_SECONDS = 2.0
_HC_MESSAGE = re.compile(
    r"\s*<hc(?:\s+from=\"(?P<sender>[^\"]*)\"[^>]*|\s[^>]*)?>(?P<body>.*)", re.S
)
SESSION_LABEL = "fast-agent.session"
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
    # Live terminals labelled with this session but parented elsewhere, e.g.
    # ``--outlive-parent`` children of an earlier run: hc wakes their old parent.
    detached: tuple[HcChild, ...] = ()
    # Every unfinished child of this terminal, for the session record.
    addresses: tuple[str, ...] = ()


_NONE = HcAttachments()
_current = _NONE
_read_at = 0.0
_watcher: asyncio.Task[None] | None = None


def hc_terminal_id() -> str | None:
    """This terminal's hc address, when running inside herdr-core."""
    terminal = os.environ.get("HC_TERMINAL_ID", "").strip()
    return terminal if terminal and os.environ.get("HC_TERMINAL_SOCKET") else None


def _hc_run(executable: str, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [executable, *args],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _hc_json(executable: str, *args: str) -> dict[str, object] | None:
    result = _hc_run(executable, *args)
    try:
        payload = json.loads(result.stdout) if result is not None else None
    except ValueError:
        return None
    return payload if is_str_object_dict(payload) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _detached_children(executable: str, terminal: str, session_id: str) -> tuple[HcChild, ...]:
    listed = _hc_json(executable, "ls", "--json", "--label", f"{SESSION_LABEL}={session_id}")
    terminals = listed.get("terminals") if listed is not None else None
    if not isinstance(terminals, list):
        return ()
    return tuple(
        HcChild(address, _optional_str(entry.get("name")))
        for entry in terminals
        if is_str_object_dict(entry)
        and isinstance(address := entry.get("address"), str)
        and entry.get("parent") != terminal
        and entry.get("state") == "live"
    )


def read_hc_attachments(session_id: str | None = None) -> HcAttachments:
    """Ask hc which children are unfinished. Blocking; call off the event loop.

    With a ``session_id``, unfinished children are labelled with it so a resumed
    session can find them, and live terminals from an earlier run are listed.
    """
    if (terminal := hc_terminal_id()) is None or (executable := shutil.which("hc")) is None:
        return _NONE
    # hc's own accounting: an agent with a delivered task counts as unfinished
    # until its turn advances, and a plain program until it exits.
    waited = _hc_json(executable, "wait", "--children", "all", "--timeout", "0")
    children = waited.get("children") if waited is not None else None
    if not isinstance(children, list):
        return _NONE
    resuming: list[HcChild] = []
    addresses: list[str] = []
    services = 0
    for child in children:
        if not is_str_object_dict(child) or child.get("ended") is not False:
            continue
        address = child.get("terminal")
        if not isinstance(address, str):
            continue
        addresses.append(address)
        detail = _hc_json(executable, "get", address) or {}
        labels = detail.get("labels")
        labels = labels if is_str_object_dict(labels) else {}
        name = detail.get("name")
        if session_id is not None and labels.get(SESSION_LABEL) != session_id:
            _hc_run(executable, "label", address, f"{SESSION_LABEL}={session_id}")
        if labels.get("role") == "service":
            services += 1
        elif (
            detail.get("notify") is True  # hc sends a note when it exits
            or detail.get("reported") is True  # an agent: it reports, or hc does for it
            or detail.get("first_message") is not None
        ):
            resuming.append(HcChild(address, _optional_str(name)))
    detached = (
        _detached_children(executable, terminal, session_id) if session_id is not None else ()
    )
    return HcAttachments(tuple(resuming), services, detached, tuple(addresses))


async def refresh_hc_attachments(
    *, reuse_recent: bool = False, session_id: str | None = None
) -> HcAttachments:
    """Update the shared snapshot without blocking the event loop.

    ``reuse_recent`` accepts a snapshot read moments ago, e.g. at the turn end
    that preceded this prompt.
    """
    global _current, _read_at

    if hc_terminal_id() is None:
        _current = _NONE
    elif not reuse_recent or time.monotonic() - _read_at >= _CACHE_SECONDS:
        _current = await asyncio.to_thread(read_hc_attachments, session_id)
        _read_at = time.monotonic()
    return _current


async def _watch_children(
    executable: str, terminal: str, on_change: Callable[[], Awaitable[None]]
) -> None:
    while True:
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "events",
                "--children-of",
                terminal,
                "--json",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return
        stdout = process.stdout
        assert stdout is not None
        try:
            while await stdout.readline():
                # Let a burst of facts (created, labels, foreground...) settle,
                # then re-read once.
                await asyncio.sleep(_EVENT_SETTLE_SECONDS)
                while True:
                    try:
                        if not await asyncio.wait_for(stdout.readline(), 0):
                            break
                    except TimeoutError:
                        break
                await on_change()
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
        await asyncio.sleep(_EVENT_RETRY_SECONDS)  # hc's core went away; follow it back


def watch_hc_children(on_change: Callable[[], Awaitable[None]]) -> None:
    """Call ``on_change`` whenever hc reports a change to one of this terminal's
    children, so a child that exits without a note stops showing as running.

    Starts once per event loop; a no-op outside hc.
    """
    global _watcher

    if _watcher is not None and not _watcher.done():
        return
    if (terminal := hc_terminal_id()) is None or (executable := shutil.which("hc")) is None:
        return
    _watcher = asyncio.create_task(_watch_children(executable, terminal, on_change))


def current_hc_attachments() -> HcAttachments:
    """The last snapshot read, for synchronous renderers such as the toolbar."""
    return _current


def _count(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


def format_turn_end_lines(attachments: HcAttachments, background_processes: int) -> list[str]:
    """Scrollback lines saying what will, and will not, bring this session back."""
    lines: list[str] = []
    resuming = attachments.resuming
    if resuming or attachments.services:
        parts: list[str] = []
        if resuming:
            names = ", ".join(child.label for child in resuming)
            pronoun = "it reports or exits" if len(resuming) == 1 else "they report or exit"
            parts.append(f"{len(resuming)} running - {names} · will resume when {pronoun}")
        if attachments.services:
            parts.append(f"{_count(attachments.services, 'service')} running")
        lines.append("hc: " + " · ".join(parts))
    if detached := attachments.detached:
        names = ", ".join(child.label for child in detached)
        lines.append(
            f"hc: {len(detached)} from an earlier run still running - {names}"
            " · reports go to its old parent; check with hc read"
        )
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
    if attachments.detached:
        names = ", ".join(child.name or child.address for child in attachments.detached)
        parts.append(f"earlier run {len(attachments.detached)}: {names}")
    if background_processes > 0:
        parts.append(f"{background_processes} bg (no resume)")
    return ", ".join(parts) or None


def is_hc_message(text: str) -> bool:
    """Whether submitted text is an hc peer message or an hc notice."""
    return _HC_MESSAGE.match(text) is not None


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
