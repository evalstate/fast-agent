"""Shared file-backed output capture for detached shell processes."""

from __future__ import annotations

import asyncio
import codecs
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import BinaryIO, Protocol

_FINAL_DRAIN_PAUSE_SECONDS = 0.05
_FINAL_DRAIN_GRACE_SECONDS = 0.25
_MAX_PENDING_LINE_CHARACTERS = 65536


class SpoolChunkReader(Protocol):
    async def __call__(self, path: str, offset: int, size: int) -> bytes: ...


class SpoolOutputHandler(Protocol):
    async def __call__(self, text: str) -> None: ...


class SpoolOutputActivityHandler(Protocol):
    async def __call__(self, byte_count: int) -> None: ...


class SpoolExitCheck(Protocol):
    async def __call__(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ShellOutputSpoolPaths:
    directory: str
    stdout: str
    stderr: str


class ShellOutputSpoolTailer:
    """Incrementally decode and emit stdout/stderr spool files until process exit."""

    def __init__(
        self,
        paths: ShellOutputSpoolPaths,
        *,
        read_chunk: SpoolChunkReader,
        on_stdout: SpoolOutputHandler,
        on_stderr: SpoolOutputHandler,
        on_stdout_activity: SpoolOutputActivityHandler | None = None,
        on_stderr_activity: SpoolOutputActivityHandler | None = None,
        chunk_size: int = 1024 * 1024,
        chunks_per_poll: int = 4,
    ) -> None:
        self._paths = paths
        self._read_chunk = read_chunk
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._on_stdout_activity = on_stdout_activity
        self._on_stderr_activity = on_stderr_activity
        self._chunk_size = chunk_size
        self._chunks_per_poll = chunks_per_poll
        self._stdout_offset = 0
        self._stderr_offset = 0
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stdout_pending = ""
        self._stderr_pending = ""
        self._last_output_at: float | None = None

    async def tail_until(
        self,
        process_exited: SpoolExitCheck,
        *,
        poll_interval: float,
        final_grace_seconds: float = _FINAL_DRAIN_GRACE_SECONDS,
    ) -> None:
        while True:
            await self._emit_deltas()
            if await process_exited():
                break
            await asyncio.sleep(poll_interval)

        # Give surviving descendants a bounded append window only when the
        # tracked process was producing output immediately before exit.
        if (
            self._last_output_at is not None
            and time.monotonic() - self._last_output_at <= final_grace_seconds
        ):
            await asyncio.sleep(final_grace_seconds)
        while not await self._emit_deltas():
            # A surviving descendant may still be appending; require a bounded
            # grace period after tracked-process exit before closing the spool.
            await asyncio.sleep(_FINAL_DRAIN_PAUSE_SECONDS)

        stdout_tail = self._stdout_decoder.decode(b"", final=True)
        stderr_tail = self._stderr_decoder.decode(b"", final=True)
        if stdout_tail:
            await self._emit_text(stdout_tail, is_stderr=False)
        if stderr_tail:
            await self._emit_text(stderr_tail, is_stderr=True)
        await self._flush_pending()

    async def _emit_deltas(self) -> bool:
        stdout_result, stderr_result = await asyncio.gather(
            self._read_available(self._paths.stdout, self._stdout_offset),
            self._read_available(self._paths.stderr, self._stderr_offset),
        )
        stdout_payload, stdout_caught_up = stdout_result
        stderr_payload, stderr_caught_up = stderr_result
        self._stdout_offset += len(stdout_payload)
        self._stderr_offset += len(stderr_payload)
        if stdout_payload or stderr_payload:
            self._last_output_at = time.monotonic()
        if stdout_payload and self._on_stdout_activity is not None:
            await self._on_stdout_activity(len(stdout_payload))
        if stderr_payload and self._on_stderr_activity is not None:
            await self._on_stderr_activity(len(stderr_payload))

        stdout = self._stdout_decoder.decode(stdout_payload, final=False)
        stderr = self._stderr_decoder.decode(stderr_payload, final=False)
        await self._emit_text(stdout, is_stderr=False)
        await self._emit_text(stderr, is_stderr=True)
        return stdout_caught_up and stderr_caught_up

    async def _emit_text(self, text: str, *, is_stderr: bool) -> None:
        pending = (self._stderr_pending if is_stderr else self._stdout_pending) + text
        handler = self._on_stderr if is_stderr else self._on_stdout

        while pending:
            line_end = _line_end(pending)
            if line_end is not None:
                line = pending[:line_end]
                pending = pending[line_end:]
                await handler(line)
                continue
            if len(pending) < _MAX_PENDING_LINE_CHARACTERS:
                break
            line = pending[:_MAX_PENDING_LINE_CHARACTERS]
            pending = pending[_MAX_PENDING_LINE_CHARACTERS:]
            await handler(line)

        if is_stderr:
            self._stderr_pending = pending
        else:
            self._stdout_pending = pending

    async def _flush_pending(self) -> None:
        if self._stdout_pending:
            stdout = self._stdout_pending
            self._stdout_pending = ""
            await self._on_stdout(stdout)
        if self._stderr_pending:
            stderr = self._stderr_pending
            self._stderr_pending = ""
            await self._on_stderr(stderr)

    async def _read_available(self, path: str, offset: int) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        current_offset = offset
        for _ in range(self._chunks_per_poll):
            payload = await self._read_chunk(path, current_offset, self._chunk_size)
            chunks.append(payload)
            current_offset += len(payload)
            if len(payload) < self._chunk_size:
                return b"".join(chunks), True
        return b"".join(chunks), False


def _line_end(text: str) -> int | None:
    """Return the end of the next LF-, CRLF-, or CR-terminated line."""
    newline = text.find("\n")
    carriage_return = text.find("\r")
    indexes = [index for index in (newline, carriage_return) if index >= 0]
    if not indexes:
        return None

    delimiter = min(indexes)
    end = delimiter + 1
    if text[delimiter] == "\r" and end < len(text) and text[end] == "\n":
        end += 1
    return end


def _local_spool_root() -> str | None:
    """Return a stable per-user root so leftover spools stay discoverable.

    Falls back to ``None`` (system temp) unless the root is a private directory
    owned by the current user, so an attacker-created shared-temp entry can
    never become the parent of a spool.
    """
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    root = Path(tempfile.gettempdir()) / f"fast-agent-managed{suffix}"
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        details = root.lstat()
        if not stat.S_ISDIR(details.st_mode) or details.st_mode & 0o022:
            return None
        if hasattr(os, "getuid") and details.st_uid != os.getuid():
            return None
    except OSError:
        return None
    return str(root)


def create_local_output_spool() -> ShellOutputSpoolPaths:
    directory = Path(tempfile.mkdtemp(prefix="fast-agent-managed-", dir=_local_spool_root()))
    directory.chmod(0o700)
    stdout = directory / "stdout.log"
    stderr = directory / "stderr.log"
    for path in (stdout, stderr):
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
    return ShellOutputSpoolPaths(
        directory=str(directory),
        stdout=str(stdout),
        stderr=str(stderr),
    )


def open_local_output_spool(
    paths: ShellOutputSpoolPaths,
) -> tuple[BinaryIO, BinaryIO]:
    stdout = os.fdopen(os.open(paths.stdout, os.O_WRONLY | os.O_APPEND), "ab", buffering=0)
    try:
        stderr = os.fdopen(os.open(paths.stderr, os.O_WRONLY | os.O_APPEND), "ab", buffering=0)
    except BaseException:
        stdout.close()
        raise
    return stdout, stderr


async def read_local_output_chunk(path: str, offset: int, size: int) -> bytes:
    def read() -> bytes:
        try:
            with Path(path).open("rb") as stream:
                stream.seek(offset)
                return stream.read(size)
        except FileNotFoundError:
            return b""

    return await asyncio.to_thread(read)


def delete_local_output_spool(paths: ShellOutputSpoolPaths) -> None:
    rmtree(paths.directory, ignore_errors=True)
