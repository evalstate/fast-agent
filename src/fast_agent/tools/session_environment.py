"""Structured session environment protocols for shell and sandbox runtimes.

Adapter authoring rule of thumb:

* implement ``ShellEnvironment`` when the provider only runs commands;
* also implement ``SessionFilesystem`` when the provider owns the files the LLM
  should read and edit;
* pass that environment to ``fast.run(environment=...)`` or
  ``fast.harness(environment=...)``. ``McpAgent`` will expose the existing LLM
  shell and filesystem tools against the right backing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RuntimeEnvironmentKind = Literal["local", "docker", "remote"]
SessionFileKind = Literal["file", "directory", "other", "unknown"]


@dataclass(frozen=True, slots=True)
class SessionFileEntry:
    """Directory entry in a session-owned filesystem."""

    path: str
    name: str
    kind: SessionFileKind = "unknown"


@dataclass(frozen=True, slots=True)
class ShellRuntimeInfo:
    """Display and diagnostics metadata for the active shell runtime."""

    name: str
    path: str | None = None
    kind: RuntimeEnvironmentKind = "local"
    provider: str | None = None


@dataclass(frozen=True, slots=True)
class ShellExecutionRequest:
    """One command execution request."""

    command: str
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ShellExecutionResult:
    """Structured result from programmatic shell execution."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass(frozen=True, slots=True)
class ShellExecutionOptions:
    """Effective execution options used by an adapter."""

    timeout_seconds: float | None = None
    warning_interval_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ShellExecution:
    """Full execution outcome used by model-facing shell tooling."""

    result: ShellExecutionResult
    options: ShellExecutionOptions
    timed_out: bool = False
    io_drain_timed_out: bool = False


class ShellExecutionCallbacks(Protocol):
    """Optional observer hooks for streaming shell execution."""

    async def on_stdout(self, text: str) -> None: ...

    async def on_stderr(self, text: str) -> None: ...

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None: ...

    async def on_timeout(self) -> None: ...


class ShellEnvironment(Protocol):
    """Minimal environment contract used by harness and shell tools."""

    async def open(self) -> None:
        """Start or attach to the environment, if needed."""
        ...

    @property
    def cwd(self) -> str:
        """Default working directory in this environment."""
        ...

    def set_cwd(self, cwd: str | None) -> None:
        """Set the default working directory for later command executions."""
        ...

    def runtime_info(self) -> ShellRuntimeInfo:
        """Return best-effort shell/environment metadata for display."""
        ...

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        """Execute one command and return full execution metadata."""
        ...

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult: ...

    async def close(self) -> None:
        """Release owned runtime resources, if any."""
        ...


class ShellExecutor(Protocol):
    """Compatibility protocol for older direct shell executor integrations."""

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult: ...


@runtime_checkable
class SessionFilesystem(Protocol):
    """Filesystem contract for shell-session-backed model tools.

    Implement this in addition to ``ShellEnvironment`` when the shell runtime
    also owns a filesystem that the LLM should naturally read and edit.
    Paths are provider-side strings. Relative paths are resolved against
    ``cwd`` by the adapter, and text operations are UTF-8.
    """

    @property
    def cwd(self) -> str:
        """Default working directory in this environment."""
        ...

    def resolve_path(self, path: str) -> str:
        """Resolve a possibly-relative path inside the session filesystem."""
        ...

    async def read_text(self, path: str) -> str:
        """Read UTF-8 text from the session filesystem."""
        ...

    async def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text to the session filesystem, creating parents as needed."""
        ...

    async def exists(self, path: str) -> bool:
        """Return whether a path exists in the session filesystem."""
        ...

    async def list_dir(self, path: str) -> list[SessionFileEntry]:
        """List direct children of a directory in the session filesystem."""
        ...

    async def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents in the session filesystem."""
        ...

    async def remove(self, path: str) -> None:
        """Remove a file from the session filesystem."""
        ...


class SessionEnvironment(ShellEnvironment, SessionFilesystem, Protocol):
    """Environment that owns both shell execution and model-facing files."""

    pass

__all__ = [
    "RuntimeEnvironmentKind",
    "SessionEnvironment",
    "SessionFileEntry",
    "SessionFileKind",
    "SessionFilesystem",
    "ShellEnvironment",
    "ShellExecutor",
    "ShellExecution",
    "ShellExecutionCallbacks",
    "ShellExecutionOptions",
    "ShellExecutionRequest",
    "ShellExecutionResult",
    "ShellRuntimeInfo",
]
