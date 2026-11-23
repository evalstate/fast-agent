"""Terminal-independent Program Status Protocol (OSC 7501) output.

https://www.superlogical.com/rex/docs/build/program-status

The lifecycle coordinator owns ordering. This sink is synchronous: one small
write per snapshot, with no socket, worker, capability query or Herdr dependency.
"""

from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from typing import Literal

from rich.file_proxy import FileProxy

from fast_agent.ui.terminal_streams import TtyCapableStream, is_tty_stream
from fast_agent.utils.env import is_truthy_env_value

type ProgramState = Literal["idle", "working", "blocked", "done", "error"]
# What a blocked program is waiting for (OSC 7501 ``kind``).
type BlockedKind = Literal["permission", "question", "auth"]


def program_status_enabled() -> bool:
    """Eligible output, not permission to emit outside an interactive session."""
    return _status_stream() is not None


def _status_stream() -> TtyCapableStream | None:
    if (
        is_truthy_env_value(os.environ.get("FAST_AGENT_DISABLE_PROGRAM_STATUS"))
        or os.environ.get("TERM", "").lower() == "dumb"
    ):
        return None
    stream: object = sys.stdout
    # Streaming wraps stdout to render ordinary prints through Rich. Control
    # sequences must bypass that formatter, without bypassing redirection.
    while isinstance(stream, FileProxy):
        stream = stream.rich_proxied_file
    return stream if is_tty_stream(stream) else None


def _clean_text(text: str, limit: int) -> str:
    # OSC free text forbids C0/C1 controls, including tabs and newlines.
    text = "".join(" " if ord(c) < 32 or 0x7F <= ord(c) <= 0x9F else c for c in text)
    text = " ".join(text.split())
    # Limits are UTF-8 bytes; never leave a partial code point at the boundary.
    return text.encode("utf-8", errors="replace")[:limit].decode("utf-8", errors="ignore")


def _encoded_text(text: str, limit: int) -> str:
    return base64.b64encode(_clean_text(text, limit).encode("utf-8")).decode("ascii")


@dataclass(frozen=True, slots=True)
class ProgramStatus:
    state: ProgramState
    title: str | None = None
    message: str | None = None
    kind: BlockedKind | None = None

    def encode(self) -> str:
        fields = [f"state={self.state}", "app=fast-agent"]
        if self.state == "blocked" and self.kind:
            fields.append(f"kind={self.kind}")
        if self.title:
            fields.append(f"title={_encoded_text(self.title, 192)}")
        if self.message:
            fields.append(f"msg={_encoded_text(self.message, 2048)}")
        return f"\x1b]7501;{':'.join(fields)}\x1b\\"


class ProgramStatusReporter:
    """Retain presentation across state changes and publish complete snapshots.

    Activated only by interactive prompt events, not metadata updates or ACP
    working scopes. The current stdout must still be a TTY on every write.
    """

    def __init__(self) -> None:
        self._active = False
        self._state: ProgramState = "idle"
        self._kind: BlockedKind | None = None
        self._queued = 0
        self._attached: str | None = None
        self._warm: str | None = None
        self._title: str | None = None
        self._context: str | None = None
        self._usage: str | None = None
        self._plugin_usage: str | None = None
        self._metadata: tuple[str | None, str | None, str | None, str | None, str | None] | None = (
            None
        )
        self._last_sequence: str | None = None
        self._last_window_title: str = ""

    def activate(self) -> None:
        self._active = True

    def refresh(self) -> None:
        """Restore the record after a prompt mark may have cleared it."""
        self._emit(force=True)

    def report_state(
        self, state: ProgramState, *, force: bool = False, kind: BlockedKind | None = None
    ) -> None:
        # Completing one turn is not quiescence while accepted input is still
        # awaiting the next prompt. Retain working through that handoff; the
        # next completion/prompt mark will settle the state when the queue drains.
        self._state = "working" if state == "idle" and self._queued else state
        self._kind = kind if state == "blocked" else None
        self._emit(force=force)

    def clear_state(self) -> None:
        """Withdraw the record without deactivating, e.g. after a startup question."""
        if self._last_sequence is not None and self._write("\x1b]7501;state=clear\x1b\\"):
            self._last_sequence = None
        if self._last_window_title and self._write("\x1b]2;\x1b\\"):
            self._last_window_title = ""

    def report_metadata(
        self,
        *,
        session_id: str | None,
        title: str | None,
        context_usage: str | None,
        token_usage: str | None,
        agent_name: str | None = None,
    ) -> None:
        metadata = (session_id, title, context_usage, token_usage, agent_name)
        if metadata == self._metadata:
            return
        if self._metadata is not None and (
            session_id != self._metadata[0] or agent_name != self._metadata[4]
        ):
            self._plugin_usage = None
        self._metadata = metadata
        self._title = title
        self._context = context_usage
        self._usage = token_usage
        self._emit()

    def report_queued(self, count: int) -> None:
        """Messages held for the next prompt (typed ahead or sent by hc)."""
        self._queued = max(0, count)
        self._emit()

    def report_attached(self, summary: str | None) -> None:
        """What the session is still attached to (children that will wake it)."""
        self._attached = summary
        self._emit()

    def report_warm(self, summary: str | None) -> None:
        """Whether the idle session is keeping its prompt cache warm."""
        self._warm = summary
        self._emit()

    def report_usage(self, usage: str) -> None:
        self._plugin_usage = usage
        self._emit()

    def release(self) -> None:
        if self._last_sequence is not None:
            self._write("\x1b]7501;state=clear\x1b\\")
        if self._last_window_title:
            self._write("\x1b]2;\x1b\\")
        self._active = False
        self._plugin_usage = None
        self._queued = 0
        self._attached = None
        self._warm = None
        self._last_sequence = None
        self._last_window_title = ""

    def _emit(self, *, force: bool = False) -> None:
        if not self._active:
            return
        # A refreshed title/context/token count must not erase plugin pricing.
        # Keep the last completed-turn projection until replaced or the session changes.
        usage = self._plugin_usage or self._usage
        queued = f"{self._queued} queued" if self._queued else None
        message = (
            " · ".join(
                part for part in (queued, self._attached, self._warm, self._context, usage) if part
            )
            or None
        )
        sequence = ProgramStatus(self._state, self._title, message, self._kind).encode()
        if (force or sequence != self._last_sequence) and self._write(sequence):
            self._last_sequence = sequence
        window_title = _clean_text(self._title, 192) if self._title else ""
        if window_title != self._last_window_title and self._write(f"\x1b]2;{window_title}\x1b\\"):
            self._last_window_title = window_title

    def _write(self, sequence: str) -> bool:
        if not self._active:
            return False
        stream = _status_stream()
        if stream is None:
            return False
        try:
            stream.write(sequence)
            stream.flush()
        except (OSError, ValueError):
            # Optional presentation must never interrupt an agent turn.
            return False
        return True
