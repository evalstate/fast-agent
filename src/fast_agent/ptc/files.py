"""Experimental local filesystem pass-through for PTC.

We may add adapters/security wrapping, or remove this entirely and prompt agents to use tools
instead of Python's file functions. This is not a filesystem sandbox.
"""

from __future__ import annotations

import asyncio
import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Any

from pydantic_monty import NOT_HANDLED, MontyFileHandle
from pydantic_monty.os_access import AbstractOS, OsFunction, StatResult, path_from_arg

if TYPE_CHECKING:
    from collections.abc import Generator

MAX_BINARY_READ_BYTES = 512 * 1024
MAX_TEXT_READ_BYTES = 4 * 1024 * 1024
_MAX_DIRECTORY_ENTRIES = 10_000
_MAX_DIRECTORY_PATH_BYTES = 4 * 1024 * 1024
_WRITE_OPERATIONS = frozenset(
    {
        "Path.write_text",
        "Path.write_bytes",
        "Path.append_text",
        "Path.append_bytes",
        "Path.mkdir",
        "Path.unlink",
        "Path.rmdir",
        "Path.rename",
    }
)


class SandboxFilesystem(AbstractOS):
    """Forward Monty's file callbacks to local Python, without caches or deferred writes."""

    def __init__(self, *, read: bool, write: bool) -> None:
        self._read = read
        self._write = write
        self._lock = asyncio.Lock()
        self._closed = False

    def __call__(
        self,
        *,
        name: OsFunction,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        is_async: bool,
        **_future_kwargs: Any,
    ) -> Any:
        # Do not expose host environment variables, clocks or other OS functionality.
        if name != "open" and not name.startswith("Path."):
            return NOT_HANDLED
        return self._run(name, args, kwargs)

    async def _run(self, name: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        writing = name in _WRITE_OPERATIONS or (name == "open" and args[1][0] != "r")
        if not (self._write if writing else self._read):
            raise PermissionError(
                "Local filesystem "
                + ("writing" if writing else "reading")
                + " is not enabled for this agent"
            )
        async with self._lock:
            if self._closed:
                raise asyncio.CancelledError
            task = asyncio.create_task(asyncio.to_thread(self.dispatch, name, args, kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A thread cannot be cancelled. Settle its operation before ending the run.
                await asyncio.gather(task, return_exceptions=True)
                raise

    async def close(self) -> None:
        """Stop new operations and settle native work even if Monty stops awaiting callbacks."""
        self._closed = True
        async with self._lock:
            pass

    def path_open(self, path: PurePosixPath, mode: str) -> MontyFileHandle:
        handle = MontyFileHandle(str(path), mode)
        # Monty maintains the logical handle; the host keeps no descriptor or file contents.
        with _regular_file(Path(path), mode) as stream:
            if mode[0] == "r":
                limit = MAX_BINARY_READ_BYTES if handle.binary else MAX_TEXT_READ_BYTES
                size = os.fstat(stream.fileno()).st_size
                if size > limit:
                    raise _too_large(path, size, limit)
        return handle

    def path_read_bytes(self, path: PurePosixPath | MontyFileHandle) -> bytes:
        return _read(path_from_arg(path), MAX_BINARY_READ_BYTES)

    def path_read_text(self, path: PurePosixPath | MontyFileHandle) -> str:
        return (
            _read(path_from_arg(path), MAX_TEXT_READ_BYTES)
            .decode("utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

    def path_write_text(self, path: PurePosixPath | MontyFileHandle, data: str) -> int:
        with _regular_file(Path(path_from_arg(path)), "w") as stream:
            return stream.write(data)

    def path_write_bytes(self, path: PurePosixPath | MontyFileHandle, data: bytes) -> int:
        with _regular_file(Path(path_from_arg(path)), "wb") as stream:
            return stream.write(data)

    def path_append_text(self, path: PurePosixPath | MontyFileHandle, data: str) -> int:
        with _regular_file(Path(path_from_arg(path)), "a") as stream:
            return stream.write(data)

    def path_append_bytes(self, path: PurePosixPath | MontyFileHandle, data: bytes) -> int:
        with _regular_file(Path(path_from_arg(path)), "ab") as stream:
            return stream.write(data)

    def path_exists(self, path: PurePosixPath) -> bool:
        return Path(path).exists()

    def path_is_file(self, path: PurePosixPath) -> bool:
        return Path(path).is_file()

    def path_is_dir(self, path: PurePosixPath) -> bool:
        return Path(path).is_dir()

    def path_is_symlink(self, path: PurePosixPath) -> bool:
        return Path(path).is_symlink()

    def path_mkdir(self, path: PurePosixPath, parents: bool, exist_ok: bool) -> None:
        Path(path).mkdir(parents=parents, exist_ok=exist_ok)

    def path_unlink(self, path: PurePosixPath) -> None:
        Path(path).unlink()

    def path_rmdir(self, path: PurePosixPath) -> None:
        Path(path).rmdir()

    def path_rename(self, path: PurePosixPath, target: PurePosixPath) -> None:
        Path(path).rename(target)

    def path_resolve(self, path: PurePosixPath) -> str:
        return str(Path(path).resolve())

    def path_absolute(self, path: PurePosixPath) -> str:
        return str(Path(path).absolute())

    def path_stat(self, path: PurePosixPath) -> StatResult:
        result = Path(path).stat()
        return StatResult(
            result.st_mode,
            result.st_ino,
            result.st_dev,
            result.st_nlink,
            result.st_uid,
            result.st_gid,
            result.st_size,
            result.st_atime,
            result.st_mtime,
            result.st_ctime,
        )

    def path_iterdir(self, path: PurePosixPath) -> list[PurePosixPath]:
        entries: list[PurePosixPath] = []
        size = 0
        with os.scandir(path) as iterator:
            for entry in iterator:
                size += 4 * len(entry.path)
                if len(entries) >= _MAX_DIRECTORY_ENTRIES or size > _MAX_DIRECTORY_PATH_BYTES:
                    raise OSError(
                        errno.EFBIG,
                        "Directory listing too large for sandbox file access",
                        str(path),
                    )
                entries.append(PurePosixPath(entry.path))
        return entries

    # AbstractOS supplies Monty's bounded walk/glob implementation using the methods above.
    def getenv(self, key: str, default: str | None = None) -> str | None:
        raise NotImplementedError

    def get_environ(self) -> dict[str, str]:
        raise NotImplementedError


@contextmanager
def _regular_file(path: Path, mode: str) -> Generator[IO[Any]]:
    flags = os.O_RDONLY if mode[0] == "r" else os.O_WRONLY | os.O_CREAT
    if mode[0] == "w":
        flags |= os.O_TRUNC
    elif mode[0] == "a":
        flags |= os.O_APPEND
    # Avoid blocking forever opening a FIFO. Special files have no bounded file-read contract.
    descriptor = os.open(path, flags | os.O_NONBLOCK, 0o666)
    try:
        kind = os.fstat(descriptor).st_mode
        if stat.S_ISDIR(kind):
            raise IsADirectoryError(errno.EISDIR, "Is a directory", str(path))
        if not stat.S_ISREG(kind):
            raise OSError(errno.EINVAL, "Sandbox file access requires a regular file", str(path))
        stream = os.fdopen(descriptor, mode, encoding=None if "b" in mode else "utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def _read(path: PurePosixPath, limit: int) -> bytes:
    with _regular_file(Path(path), "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit:
            raise _too_large(path, size, limit)
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise _too_large(path, len(data), limit)
    return data


def _too_large(path: PurePosixPath, size: int, limit: int) -> OSError:
    return OSError(
        errno.EFBIG,
        f"File too large to read in the sandbox ({format_size(size)}; the limit is "
        f"{format_size(limit)}, and even read(n) loads the whole file). Process it with a shell "
        "command instead, e.g. head -c, xxd, sha256sum or python3",
        str(path),
    )


def format_size(size: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:.1f}".removesuffix(".0") + f" {unit}"
    return f"{size} bytes"
