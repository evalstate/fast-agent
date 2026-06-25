"""Structured session environment protocols for sandbox-style runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
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


__all__ = [
    "ShellExecutionResult",
    "ShellExecutor",
]
