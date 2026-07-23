"""Shared file-backed output capture for detached shell processes."""

from __future__ import annotations

import asyncio
import codecs
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import BinaryIO, Protocol

_FINAL_DRAIN_PAUSE_SECONDS = 0.05


class SpoolChunkReader(Protocol):
    async def __call__(self, path: str, offset: int, size: int) -> bytes: ...


class SpoolOutputHandler(Protocol):
    async def __call__(self, text: str) -> None: ...


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
        chunk_size: int = 1024 * 1024,
        chunks_per_poll: int = 4,
    ) -> None:
        self._paths = paths
        self._read_chunk = read_chunk
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._chunk_size = chunk_size
        self._chunks_per_poll = chunks_per_poll
        self._stdout_offset = 0
        self._stderr_offset = 0
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    async def tail_until(
        self,
        process_exited: SpoolExitCheck,
        *,
        poll_interval: float,
    ) -> None:
        while True:
            await self._emit_deltas()
            if await process_exited():
                break
            await asyncio.sleep(poll_interval)

        while not await self._emit_deltas():
            # A surviving descendant may still be appending; yield between
            # catch-up rounds instead of spinning at full read speed.
            await asyncio.sleep(_FINAL_DRAIN_PAUSE_SECONDS)

        stdout_tail = self._stdout_decoder.decode(b"", final=True)
        stderr_tail = self._stderr_decoder.decode(b"", final=True)
        if stdout_tail:
            await self._on_stdout(stdout_tail)
        if stderr_tail:
            await self._on_stderr(stderr_tail)

    async def _emit_deltas(self) -> bool:
        stdout_result, stderr_result = await asyncio.gather(
            self._read_available(self._paths.stdout, self._stdout_offset),
            self._read_available(self._paths.stderr, self._stderr_offset),
        )
        stdout_payload, stdout_caught_up = stdout_result
        stderr_payload, stderr_caught_up = stderr_result
        self._stdout_offset += len(stdout_payload)
        self._stderr_offset += len(stderr_payload)

        stdout = self._stdout_decoder.decode(stdout_payload, final=False)
        stderr = self._stderr_decoder.decode(stderr_payload, final=False)
        if stdout:
            await self._on_stdout(stdout)
        if stderr:
            await self._on_stderr(stderr)
        return stdout_caught_up and stderr_caught_up

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
    stdout = Path(paths.stdout).open("ab", buffering=0)
    try:
        stderr = Path(paths.stderr).open("ab", buffering=0)
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
