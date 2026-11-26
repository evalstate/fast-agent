"""Terminal-independent Program Resume Protocol (OSC 7577) output.

https://www.superlogical.com/rex/docs/build/program-resume

Reports the command a terminal can run to resume the current interactive
session after a terminal restart or reboot. Reports are authenticated with a
per-terminal token supplied in ``OSC7577_TOKEN``; the token is removed from
``os.environ`` on capture so child processes cannot spoof or overwrite the
parent's resume record.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.file_proxy import FileProxy

from fast_agent.ui.terminal_streams import TtyCapableStream, is_tty_stream
from fast_agent.utils.env import is_truthy_env_value

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

PROGRAM_RESUME_TOKEN_ENV = "OSC7577_TOKEN"
DISABLE_PROGRAM_RESUME_ENV = "FAST_AGENT_DISABLE_PROGRAM_RESUME"

# Minimum limits every OSC 7577 terminal is required to accept.
MAX_SEQUENCE_BYTES = 2048
MAX_ARGV_ELEMENTS = 64


@dataclass(frozen=True, slots=True)
class ProgramResumeCommand:
    """Command and optional absolute working directory to resume a program."""

    argv: tuple[str, ...]
    cwd: str | None = None


def _resume_stream() -> TtyCapableStream | None:
    if (
        is_truthy_env_value(os.environ.get(DISABLE_PROGRAM_RESUME_ENV))
        or os.environ.get("TERM", "").lower() == "dumb"
    ):
        return None
    stream: object = sys.stdout
    while isinstance(stream, FileProxy):
        stream = stream.rich_proxied_file
    return stream if is_tty_stream(stream) else None


def take_program_resume_token(env: MutableMapping[str, str] | None = None) -> str | None:
    """Remove ``OSC7577_TOKEN`` from ``env`` (default ``os.environ``) and return it."""
    target = os.environ if env is None else env
    token = target.pop(PROGRAM_RESUME_TOKEN_ENV, None)
    return token or None


def _is_reportable_text(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return all(ord(ch) >= 0x20 and not 0x7F <= ord(ch) <= 0x9F for ch in text)


def _is_valid_program(program: str) -> bool:
    if not program or not _is_reportable_text(program):
        return False
    if os.path.isabs(program):
        return True
    return "/" not in program and "\\" not in program and not program.startswith("-")


def _is_reportable_command(command: ProgramResumeCommand) -> bool:
    if not (1 <= len(command.argv) <= MAX_ARGV_ELEMENTS):
        return False
    if not _is_valid_program(command.argv[0]):
        return False
    if not all(_is_reportable_text(arg) for arg in command.argv):
        return False
    if command.cwd is not None and (
        not _is_reportable_text(command.cwd) or not os.path.isabs(command.cwd)
    ):
        return False
    return True


def _encode_b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _format_report(pairs: Sequence[str], *, token: str, pid: int, now: int) -> str:
    body = ":".join((f"pid={pid}", f"time={now}", *pairs))
    # Plain SHA-256 with the token last (not HMAC), per OSC 7577 spec.
    mac = hashlib.sha256(f"{body}:{token}".encode("utf-8")).hexdigest()
    return f"\x1b]7577;{body}:mac={mac}\x1b\\"


def format_program_resume(
    command: ProgramResumeCommand,
    *,
    token: str,
    pid: int,
    now: int,
) -> str | None:
    """Encode an OSC 7577 resume report, or ``None`` if it cannot be carried."""
    if not token or pid <= 0 or now <= 0 or not _is_reportable_command(command):
        return None
    pairs = [f"argv={','.join(_encode_b64(arg) for arg in command.argv)}"]
    if command.cwd is not None:
        pairs.append(f"cwd={_encode_b64(command.cwd)}")
    sequence = _format_report(pairs, token=token, pid=pid, now=now)
    return sequence if len(sequence.encode("utf-8")) <= MAX_SEQUENCE_BYTES else None


def format_program_resume_clear(*, token: str, pid: int, now: int) -> str | None:
    """Encode an OSC 7577 ``clear=1`` report removing the owning record."""
    if not token or pid <= 0 or now <= 0:
        return None
    return _format_report(("clear=1",), token=token, pid=pid, now=now)


class ProgramResumeReporter:
    """Synchronous OSC 7577 writer for a single process."""

    def __init__(
        self,
        token: str,
        *,
        pid: int | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._token = token
        self._pid = os.getpid() if pid is None else pid
        self._clock = (lambda: int(time.time())) if clock is None else clock
        self._reported: ProgramResumeCommand | None = None

    @property
    def reported_command(self) -> ProgramResumeCommand | None:
        return self._reported

    def report(self, command: ProgramResumeCommand | None) -> bool:
        """Report ``command`` if it changed since the last accepted write."""
        if command == self._reported:
            return False
        return self._send(command)

    def report_again(self) -> bool:
        """Re-emit the current command (e.g. on ``SIGCONT`` after suspension)."""
        if self._reported is None:
            return False
        return self._send(self._reported)

    def clear(self) -> bool:
        """Withdraw the current record when the user intentionally exits."""
        if self._reported is None:
            return False
        return self._send(None)

    def _send(self, command: ProgramResumeCommand | None) -> bool:
        now = self._clock()
        if command is not None:
            sequence = format_program_resume(
                command,
                token=self._token,
                pid=self._pid,
                now=now,
            )
            if sequence is not None:
                if self._write(sequence):
                    self._reported = command
                    return True
                return False
            # Command cannot be carried (e.g. >2048 bytes or control chars):
            # clear any prior record so the terminal does not resume a stale one.
            if self._reported is None:
                return False

        clear_seq = format_program_resume_clear(
            token=self._token,
            pid=self._pid,
            now=now,
        )
        if clear_seq is not None and self._write(clear_seq):
            self._reported = None
            return True
        return False

    def _write(self, sequence: str) -> bool:
        stream = _resume_stream()
        if stream is None:
            return False
        try:
            stream.write(sequence)
            stream.flush()
        except (OSError, ValueError):
            return False
        return True


_captured_token: str | None = None
_reporter: ProgramResumeReporter | None = None
_command_builder: Callable[[str | None], ProgramResumeCommand | None] | None = None
_sigcont_loop: asyncio.AbstractEventLoop | None = None


def _after_fork_child() -> None:
    global _captured_token, _reporter, _command_builder, _sigcont_loop

    _captured_token = None
    _reporter = None
    _command_builder = None
    _sigcont_loop = None


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_after_fork_child)


def capture_program_resume_token() -> None:
    """Pop ``OSC7577_TOKEN`` from ``os.environ`` once and retain it privately.

    Always strips the variable from ``os.environ`` so child processes (shell
    tools, MCP servers, sandboxes) cannot inherit it.
    """
    global _captured_token

    token = take_program_resume_token()
    if token is not None:
        _captured_token = token


def _install_sigcont_handler() -> None:
    global _sigcont_loop

    if sys.platform == "win32" or _sigcont_loop is not None:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGCONT, refresh_program_resume)
        _sigcont_loop = loop
    except (RuntimeError, NotImplementedError, OSError, ValueError):
        pass


def _remove_sigcont_handler() -> None:
    global _sigcont_loop

    loop = _sigcont_loop
    _sigcont_loop = None
    if sys.platform == "win32" or loop is None or loop.is_closed():
        return
    try:
        loop.remove_signal_handler(signal.SIGCONT)
    except (RuntimeError, NotImplementedError, OSError, ValueError):
        pass


def configure_program_resume(
    command_builder: Callable[[str | None], ProgramResumeCommand | None] | None,
) -> None:
    """Attach (or clear) the session-to-command builder for this process."""
    global _reporter, _command_builder

    _command_builder = command_builder
    if command_builder is None or _captured_token is None:
        _remove_sigcont_handler()
        return
    if _reporter is None:
        _reporter = ProgramResumeReporter(_captured_token)
    _install_sigcont_handler()


def program_resume_configured() -> bool:
    """Whether OSC 7577 reporting has both a captured token and a command builder."""
    return _captured_token is not None and _command_builder is not None


def report_program_resume_session(session_id: str | None) -> None:
    """Update the OSC 7577 record for ``session_id`` (or clear when ``None``)."""
    try:
        if _reporter is None or _command_builder is None:
            return
        command = _command_builder(session_id)
        _reporter.report(command)
    except Exception:
        pass


def refresh_program_resume() -> None:
    """Re-emit the active OSC 7577 record (used after ``SIGCONT``)."""
    try:
        if _reporter is None:
            return
        _reporter.report_again()
    except Exception:
        pass


def clear_program_resume() -> None:
    """Send ``clear=1`` when the user intentionally exits the session."""
    try:
        if _reporter is None:
            return
        _reporter.clear()
    except Exception:
        pass


def deactivate_program_resume() -> None:
    """Detach callbacks and remove ``SIGCONT`` without sending ``clear=1``.

    Preserves any reported record when the process ends due to terminal close,
    ``SIGHUP``, ``SIGTERM``, or an unexpected error.
    """
    global _command_builder

    _command_builder = None
    _remove_sigcont_handler()


def _reset_for_tests() -> None:
    global _captured_token, _reporter, _command_builder

    _remove_sigcont_handler()
    _captured_token = None
    _reporter = None
    _command_builder = None
