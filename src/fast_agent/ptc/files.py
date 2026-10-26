"""Serve sandbox file operations through the agent's file and shell tools.

Monty has no filesystem: ``open()``, ``pathlib`` and the ``os`` file functions suspend as OS calls
that the host answers. This module answers them by calling the agent's own tools, so sandbox code
reaches exactly the files those tools can, with the same permissions, hooks and execution
environment (a remote container, say). Each operation becomes one or more nested tool calls:

- text reads use ``read_text_file`` and whole-file text writes use ``write_text_file``;
- everything else (binary data, appends, metadata, listings, mkdir/unlink/rename) runs a small
  POSIX shell script through the shell tool, with binary data carried as base64.

Writes are buffered and committed before the next other file operation or tool call, and when
the script ends, so a loop of ``f.write()`` calls costs one tool call rather than one per write.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from mcp_types import CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_PTC_VALUE
from fast_agent.ptc.stubs import PtcFunction, build_functions
from fast_agent.tools.filesystem_tool_definitions import (
    READ_TEXT_FILE_TOOL_NAME,
    WRITE_TEXT_FILE_TOOL_NAME,
)
from fast_agent.utils.tool_names import is_shell_command_tool_name

if TYPE_CHECKING:
    from mcp_types import Tool

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[CallToolResult]]
"""Run a nested tool call: (tool name, arguments) -> result."""

# OS calls answered here; others (environment, clocks, entropy) keep Monty's default behaviour.
_FILE_OPERATIONS = frozenset(
    {
        "open",
        "Path.exists",
        "Path.is_file",
        "Path.is_dir",
        "Path.is_symlink",
        "Path.read_text",
        "Path.read_bytes",
        "Path.write_text",
        "Path.write_bytes",
        "Path.append_text",
        "Path.append_bytes",
        "Path.mkdir",
        "Path.unlink",
        "Path.rmdir",
        "Path.iterdir",
        "Path.scan",
        "Path.stat",
        "Path.rename",
        "Path.resolve",
        "Path.absolute",
    }
)
_WRITE_OPERATIONS = frozenset(
    {"Path.write_text", "Path.write_bytes", "Path.append_text", "Path.append_bytes"}
)
# Raw bytes per shell write; base64 inflates them by 4/3, staying well under Linux's 128 KiB
# limit on a single command-line argument.
_SHELL_WRITE_CHUNK = 64 * 1024
# Shell scripts report a known failure as "E:<ERRNO NAME>" on stderr and exit 3.
_FAIL = 'fail() { printf "E:%s" "$1" >&2; exit 3; }; '
_ERRNO_TOKEN = re.compile(r"E:([A-Z]+)")
_ERRNO_NUMBER = re.compile(r"\[Errno (\d+)\]")
_ERRNO_NAMES = {
    "ENOENT": errno.ENOENT,
    "EISDIR": errno.EISDIR,
    "ENOTDIR": errno.ENOTDIR,
    "EEXIST": errno.EEXIST,
    "ENOTEMPTY": errno.ENOTEMPTY,
    "EACCES": errno.EACCES,
}
_ERRNO_PHRASES = (
    ("No such file or directory", errno.ENOENT),
    ("Is a directory", errno.EISDIR),
    ("Not a directory", errno.ENOTDIR),
    ("Directory not empty", errno.ENOTEMPTY),
    ("File exists", errno.EEXIST),
    ("Permission denied", errno.EACCES),
)
# Classifies a path as d(irectory), f(ile), o(ther) or n(othing), then 1 if it is a symlink.
_KIND_SCRIPT = (
    'if [ -d "$p" ]; then k=d; elif [ -f "$p" ]; then k=f; elif [ -e "$p" ]; then k=o; '
    'else k=n; fi; l=0; [ -L "$p" ] && l=1; printf "%s%s\\n" "$k" "$l"'
)
_SCAN_ENTRY_SCRIPT = (
    'for f do t=o; if [ -d "$f" ]; then t=d; elif [ -f "$f" ]; then t=f; fi; l=0; '
    '[ -L "$f" ] && l=1; printf "%s%s %s\\n" "$t" "$l" "$f"; done'
)
_REQUIRE_EXISTS = 'if [ ! -e "$p" ] && [ ! -L "$p" ]; then fail ENOENT; fi; '
_REQUIRE_FILE = _REQUIRE_EXISTS + '[ -d "$p" ] && fail EISDIR; '
_REQUIRE_DIR = _REQUIRE_EXISTS + '[ -d "$p" ] || fail ENOTDIR; '
_UNLINK = _REQUIRE_EXISTS + 'if [ -d "$p" ] && [ ! -L "$p" ]; then fail EISDIR; fi; rm -f -- "$p"'


@dataclass(frozen=True, slots=True)
class SandboxFileTools:
    """The agent tools that can serve sandbox file operations."""

    read: PtcFunction | None = None
    write: PtcFunction | None = None
    shell: PtcFunction | None = None

    @classmethod
    def from_tools(cls, tools: Sequence[Tool]) -> SandboxFileTools:
        candidates = build_functions(
            [
                tool
                for tool in tools
                if tool.name in (READ_TEXT_FILE_TOOL_NAME, WRITE_TEXT_FILE_TOOL_NAME)
                or is_shell_command_tool_name(tool.name)
            ]
        ).values()
        by_name = {function.tool_name: function for function in candidates}
        shell = next((function for function in candidates if _runs_commands(function)), None)
        return cls(
            read=by_name.get(READ_TEXT_FILE_TOOL_NAME),
            write=by_name.get(WRITE_TEXT_FILE_TOOL_NAME),
            shell=shell,
        )

    @property
    def available(self) -> bool:
        return self.read is not None or self.write is not None or self.shell is not None

    def tool_names(self) -> list[str]:
        return [
            function.tool_name
            for function in (self.read, self.write, self.shell)
            if function is not None
        ]


def _runs_commands(function: PtcFunction) -> bool:
    names = {parameter.python_name for parameter in function.parameters}
    required = {parameter.python_name for parameter in function.parameters if parameter.required}
    return "command" in names and required <= {"command", "description"}


@dataclass(slots=True)
class _PendingWrite:
    truncate: bool
    chunks: list[str | bytes] = field(default_factory=list)

    def data(self) -> str | bytes:
        if all(isinstance(chunk, str) for chunk in self.chunks):
            return "".join(str(chunk) for chunk in self.chunks)
        return b"".join(
            chunk.encode() if isinstance(chunk, str) else chunk for chunk in self.chunks
        )


class SandboxFilesystem:
    """Answers one execution's sandbox OS calls with nested tool calls."""

    def __init__(self, tools: SandboxFileTools, call: ToolCaller, *, guidance: str) -> None:
        self._tools = tools
        self._call = call
        self._guidance = guidance
        self._pending: dict[str, _PendingWrite] = {}
        # Content read by open() to check the file, served to the read that follows it.
        self._opened: dict[tuple[str, bool], str | bytes] = {}
        self._flush_lock = asyncio.Lock()

    def handler(
        self,
        *,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        is_async: bool,
        **_future_kwargs: Any,
    ) -> Any:
        """Monty ``os=`` callback; returns a coroutine for file operations."""
        from pydantic_monty import NOT_HANDLED

        if name not in _FILE_OPERATIONS:
            return NOT_HANDLED
        return self._run(name, args, kwargs)

    async def before_tool_call(self) -> None:
        """Commit buffered writes and forget cached reads before a tool can see the files."""
        await self.flush()
        self._opened.clear()

    async def finish(self) -> str | None:
        """Commit buffered writes at the end of a script; return an error message on failure."""
        self._opened.clear()
        try:
            await self.flush()
        except OSError as exc:
            return str(exc)
        return None

    async def flush(self) -> None:
        async with self._flush_lock:
            while self._pending:
                path = next(iter(self._pending))
                pending = self._pending.pop(path)
                try:
                    await self._commit(path, pending)
                except OSError:
                    unwritten = list(self._pending)
                    self._pending.clear()
                    if unwritten:
                        raise OSError(
                            f"writing {path} failed, so later writes to {', '.join(unwritten)} "
                            "were not made"
                        ) from None
                    raise

    async def _run(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        path = str(args[0])
        if name in _WRITE_OPERATIONS:
            self._require_writer(name)
            self._buffer(path, args[1], truncate=name.startswith("Path.write"))
            return len(args[1])
        if name in ("Path.resolve", "Path.absolute"):
            # Monty already made the path absolute and normalised; symlinks are not resolved.
            return path
        await self.flush()
        match name:
            case "open":
                return await self._open(path, str(args[1]))
            case "Path.read_text":
                cached = self._opened.pop((path, False), None)
                return cached if cached is not None else await self._read_text(path)
            case "Path.read_bytes":
                cached = self._opened.pop((path, True), None)
                return cached if cached is not None else await self._read_bytes(path)
            case "Path.exists":
                return (await self._kind(path, name))[0] != "n"
            case "Path.is_file":
                return (await self._kind(path, name))[0] == "f"
            case "Path.is_dir":
                return (await self._kind(path, name))[0] == "d"
            case "Path.is_symlink":
                return (await self._kind(path, name))[1] == "1"
            case "Path.stat":
                return await self._stat(path)
            case "Path.iterdir":
                return await self._iterdir(path)
            case "Path.scan":
                return await self._scan(
                    path,
                    max_depth=kwargs.get("max_depth"),
                    follow_symlinks=bool(kwargs.get("follow_symlinks")),
                )
            case "Path.mkdir":
                return await self._mkdir(
                    path,
                    parents=bool(kwargs.get("parents")),
                    exist_ok=bool(kwargs.get("exist_ok")),
                )
            case "Path.unlink":
                await self._shell(name, path, _UNLINK)
                return None
            case "Path.rmdir":
                await self._shell(name, path, _REQUIRE_DIR + 'rmdir -- "$p"')
                return None
            case "Path.rename":
                await self._rename(path, str(args[1]))
                return None
        raise NotImplementedError(name)

    def _require_writer(self, operation: str) -> None:
        if self._tools.write is None and self._tools.shell is None:
            raise self._unavailable(operation)

    def _unavailable(self, operation: str) -> PermissionError:
        return PermissionError(
            f"{operation}: no tool can serve this from the sandbox. {self._guidance}"
        )

    def _buffer(self, path: str, data: str | bytes, *, truncate: bool) -> None:
        pending = self._pending.get(path)
        if truncate or pending is None:
            pending = self._pending[path] = _PendingWrite(truncate=truncate)
        pending.chunks.append(data)

    async def _open(self, path: str, mode: str) -> Any:
        from pydantic_monty import MontyFileHandle

        handle = MontyFileHandle(path, mode)
        action = handle.mode[0]
        if action == "r":
            # Reading now reports a missing file from open() itself, as CPython does.
            self._opened[(path, handle.binary)] = (
                await self._read_bytes(path) if handle.binary else await self._read_text(path)
            )
        else:
            self._require_writer("open")
            self._buffer(path, b"" if handle.binary else "", truncate=action == "w")
        return handle

    async def _read_text(self, path: str) -> str:
        if self._tools.read is not None:
            result = await self._call_function(self._tools.read, {"path": path})
            if result.is_error:
                raise _os_error(_result_text(result), path)
            return _result_text(result)
        return (await self._read_bytes(path)).decode()

    async def _read_bytes(self, path: str) -> bytes:
        output = await self._shell("read", path, _REQUIRE_FILE + 'base64 < "$p"')
        return base64.b64decode(output)

    async def _commit(self, path: str, pending: _PendingWrite) -> None:
        data = pending.data()
        if pending.truncate and isinstance(data, str) and self._tools.write is not None:
            result = await self._call_function(self._tools.write, {"path": path, "content": data})
            if result.is_error:
                raise _os_error(_result_text(result), path)
            return
        if self._tools.shell is not None:
            await self._shell_write(path, data.encode() if isinstance(data, str) else data, pending)
            return
        if self._tools.write is not None and self._tools.read is not None and isinstance(data, str):
            try:
                existing = await self._read_text(path)
            except FileNotFoundError:
                existing = ""
            result = await self._call_function(
                self._tools.write, {"path": path, "content": existing + data}
            )
            if result.is_error:
                raise _os_error(_result_text(result), path)
            return
        raise self._unavailable(f"writing {path} (binary data and appends need a shell tool)")

    async def _shell_write(self, path: str, data: bytes, pending: _PendingWrite) -> None:
        checks = '[ -d "$p" ] && fail EISDIR; [ -d "$(dirname -- "$p")" ] || fail ENOENT; '
        chunks = [data[i : i + _SHELL_WRITE_CHUNK] for i in range(0, len(data), _SHELL_WRITE_CHUNK)]
        redirect = ">" if pending.truncate else ">>"
        if not chunks:
            await self._shell("write", path, f'{checks}: {redirect} "$p"')
            return
        for chunk in chunks:
            encoded = base64.b64encode(chunk).decode()
            await self._shell(
                "write", path, f'{checks}printf %s {encoded} | base64 -d {redirect} "$p"'
            )
            redirect = ">>"

    async def _kind(self, path: str, operation: str) -> str:
        return (await self._shell(operation, path, _KIND_SCRIPT)).strip()

    async def _stat(self, path: str) -> Any:
        from pydantic_monty.os_access import StatResult

        output = await self._shell(
            "Path.stat",
            path,
            _REQUIRE_EXISTS
            + "if s=$(stat -L -c '%f %i %d %h %u %g %s %X %Y %Z' -- \"$p\" 2>/dev/null); "
            'then printf "%s" "$s"; elif [ -d "$p" ]; then printf "x d"; '
            'else printf "x f %s" "$(wc -c < "$p" | tr -d " ")"; fi',
        )
        fields = output.split()
        if fields[0] == "x":
            if fields[1] == "d":
                return StatResult.dir_stat()
            return StatResult.file_stat(size=int(fields[2]))
        mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime = fields
        return StatResult(
            int(mode, 16),
            int(ino),
            int(dev),
            int(nlink),
            int(uid),
            int(gid),
            int(size),
            float(atime),
            float(mtime),
            float(ctime),
        )

    async def _iterdir(self, path: str) -> list[PurePosixPath]:
        output = await self._shell(
            "Path.iterdir",
            path,
            _REQUIRE_DIR + '[ -r "$p" ] || fail EACCES; find "$p" -mindepth 1 -maxdepth 1',
        )
        return [PurePosixPath(line) for line in output.splitlines() if line]

    async def _scan(self, path: str, *, max_depth: int | None, follow_symlinks: bool) -> list[Any]:
        from pydantic_monty.os_access import ScanEntry

        depth = "" if max_depth is None else f" -maxdepth {int(max_depth)}"
        follow = " -L" if follow_symlinks else ""
        output = await self._shell(
            "Path.scan",
            path,
            _REQUIRE_DIR
            + f'find{follow} "$p"{depth} -exec sh -c {shlex.quote(_SCAN_ENTRY_SCRIPT)} sh {{}} + '
            "2>/dev/null; exit 0",
        )
        prefix = path.rstrip("/") + "/"
        entries: list[Any] = []
        for line in output.splitlines():
            flags, _, entry_path = line.partition(" ")
            if len(flags) != 2 or not entry_path:
                continue
            relative = "" if entry_path == path else entry_path.removeprefix(prefix)
            entries.append(ScanEntry(relative, flags[0] == "d", flags[0] == "f", flags[1] == "1"))
        return entries

    async def _mkdir(self, path: str, *, parents: bool, exist_ok: bool) -> None:
        existing = "exit 0" if exist_ok else "fail EEXIST"
        parent_check = "" if parents else '[ -d "$(dirname -- "$p")" ] || fail ENOENT; '
        await self._shell(
            "Path.mkdir",
            path,
            f'if [ -d "$p" ]; then {existing}; fi; '
            'if [ -e "$p" ] || [ -L "$p" ]; then fail EEXIST; fi; '
            f'{parent_check}mkdir{" -p" if parents else ""} -- "$p"',
        )

    async def _rename(self, source: str, target: str) -> None:
        await self._shell(
            "Path.rename",
            source,
            _REQUIRE_EXISTS + f"t={shlex.quote(target)}; "
            'if [ -d "$t" ] && [ ! -L "$t" ]; then [ -d "$p" ] || fail EISDIR; '
            'rmdir -- "$t" 2>/dev/null || fail ENOTEMPTY; fi; mv -f -- "$p" "$t"',
        )

    async def _shell(self, operation: str, path: str, script: str) -> str:
        """Run ``script`` with ``$p`` set to ``path``; return stdout or raise the OSError."""
        shell = self._tools.shell
        if shell is None:
            raise self._unavailable(operation)
        command = f"p={shlex.quote(path)}; {_FAIL}{script}"
        arguments: dict[str, Any] = {"command": command}
        if any(parameter.python_name == "description" for parameter in shell.parameters):
            arguments["description"] = f"Python {operation} {path}"
        result = await self._call_function(shell, arguments)
        value = (result.meta or {}).get(FAST_AGENT_PTC_VALUE)
        if not isinstance(value, Mapping):
            raise OSError(f"{operation} {path}: {_result_text(result) or 'shell call failed'}")
        stdout = str(value.get("stdout") or "")
        if value.get("exit_code") != 0:
            if value.get("status") not in ("completed", "failed"):
                raise OSError(f"{operation} {path}: shell command {value.get('status')}")
            raise _os_error(str(value.get("stderr") or stdout), path)
        if value.get("truncated"):
            raise OSError(
                f"{operation} {path}: output too large for sandbox file access; "
                "process it with a shell command instead"
            )
        return stdout

    async def _call_function(
        self, function: PtcFunction, arguments: dict[str, Any]
    ) -> CallToolResult:
        return await self._call(function.tool_name, function.bind_arguments(arguments))


def _os_error(message: str, path: str) -> OSError:
    """Rebuild the OSError a tool or shell script reported, keeping its errno subclass."""
    code: int | None = None
    if (token := _ERRNO_TOKEN.search(message)) is not None:
        code = _ERRNO_NAMES.get(token.group(1))
    elif (number := _ERRNO_NUMBER.search(message)) is not None:
        code = int(number.group(1))
    else:
        code = next((value for phrase, value in _ERRNO_PHRASES if phrase in message), None)
    if code is None:
        return OSError(f"{path}: {message.strip() or 'operation failed'}")
    return OSError(code, os.strerror(code), path)


def _result_text(result: CallToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))
