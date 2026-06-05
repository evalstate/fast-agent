"""Structured session environment protocols for sandbox-style runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ShellExecutionResult:
    """Structured result from programmatic shell execution."""

    stdout: str
    stderr: str
    exit_code: int


class ShellExecutor(Protocol):
    """Lower-level shell executor below model-facing tool adapters."""

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult: ...


class SessionEnvironment(ShellExecutor, Protocol):
    """Sandbox-like environment behind shell and filesystem tool adapters."""

    @property
    def cwd(self) -> Path | str: ...

    async def read_text_file(self, path: str | Path) -> str: ...

    async def write_text_file(self, path: str | Path, content: str) -> None: ...

    async def read_bytes(self, path: str | Path) -> bytes: ...

    async def write_bytes(self, path: str | Path, content: bytes) -> None: ...

    async def list_dir(self, path: str | Path) -> Sequence[str]: ...

    async def mkdir(self, path: str | Path, *, parents: bool = False) -> None: ...

    async def remove(
        self,
        path: str | Path,
        *,
        recursive: bool = False,
        force: bool = False,
    ) -> None: ...

    def metadata(self) -> Mapping[str, object]: ...


__all__ = [
    "SessionEnvironment",
    "ShellExecutionResult",
    "ShellExecutor",
]
