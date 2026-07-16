"""Structured environment protocols for shell and filesystem runtimes.

Adapter authoring rule of thumb:

* implement ``ShellEnvironment`` when the provider only runs commands;
* also implement ``EnvironmentFilesystem`` when the provider owns the files the LLM
  should read and edit;
* pass that environment to ``fast.run(environment=...)`` or
  ``fast.harness(environment=...)``. ``McpAgent`` will expose the existing LLM
  shell and filesystem tools against the right backing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RuntimeEnvironmentKind = str
EnvironmentFileKind = Literal["file", "directory", "other", "unknown"]


@dataclass(frozen=True, slots=True)
class EnvironmentFileEntry:
    """Directory entry in an environment-owned filesystem."""

    path: str
    name: str
    kind: EnvironmentFileKind = "unknown"


@dataclass(frozen=True, slots=True)
class ShellRuntimeInfo:
    """Display and diagnostics metadata for the active shell runtime.

    ``kind`` is coarse display metadata. Built-in environments use values such
    as ``local``, ``docker``, and ``remote``; custom providers may use another
    stable string and should set ``provider`` to the more specific adapter name.
    """

    name: str
    path: str | None = None
    kind: RuntimeEnvironmentKind = "local"
    provider: str | None = None
    environment_name: str | None = None


@dataclass(frozen=True, slots=True)
class ShellExecutionRequest:
    """One command execution request."""

    command: str
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout: float | None = None
    terminate_after_idle: bool = True
    retain_output: bool = True


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
    """Optional observer hooks for shell execution.

    Environments that support live output should call stdout/stderr callbacks as
    chunks arrive. Environments that cannot stream live output may omit these
    callbacks; the returned ``ShellExecution`` remains the authoritative complete
    result. Callers must tolerate zero, one, or many stdout/stderr callback chunks.
    """

    async def on_started(self, process_id: int | None) -> None: ...

    async def on_stdout(self, text: str) -> None: ...

    async def on_stderr(self, text: str) -> None: ...

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None: ...

    async def on_timeout(self) -> None: ...


@runtime_checkable
class ShellEnvironment(Protocol):
    """Minimal environment contract used by harness and shell tools.

    Environment objects may be shared across agents and harness sessions. Avoid
    mutable per-session defaults; use ``ShellExecutionRequest.cwd`` for
    request-specific working directories.
    """

    async def open(self) -> None:
        """Start or attach to the environment, if needed.

        Implementations should make repeated ``open()`` calls safe, including
        after ``close()`` when the adapter supports restarting. If restarting is
        unsupported, raise a clear runtime error.
        """
        ...

    @property
    def cwd(self) -> str:
        """Default working directory in this environment."""
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
        """Execute one command and return full execution metadata.

        If the coroutine is cancelled, adapters that own a running process or
        remote job must make a best effort to terminate that execution before
        re-raising cancellation.
        """
        ...

    async def close(self) -> None:
        """Release owned runtime resources, if any."""
        ...


@runtime_checkable
class EnvironmentStartupProgress(Protocol):
    """Optional startup progress hook for environment adapters."""

    def set_startup_progress_callback(
        self,
        callback: Callable[[str], None] | None,
    ) -> None: ...


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
class EnvironmentFilesystem(Protocol):
    """Filesystem contract for environment-backed model tools.

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
        """Resolve a possibly-relative path inside the environment filesystem."""
        ...

    async def read_text(self, path: str) -> str:
        """Read UTF-8 text from the environment filesystem.

        Missing paths should raise ``FileNotFoundError`` or the provider's
        closest equivalent.
        """
        ...

    async def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text to the environment filesystem, creating parents as needed."""
        ...

    async def exists(self, path: str) -> bool:
        """Return whether a path exists in the environment filesystem."""
        ...

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        """List direct children of a directory in the environment filesystem.

        Missing paths should raise ``FileNotFoundError`` or the provider's
        closest equivalent. Non-directory paths should raise ``NotADirectoryError``
        or the provider's closest equivalent.
        """
        ...

    async def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents in the environment filesystem."""
        ...

    async def remove(self, path: str) -> None:
        """Remove a file from the environment filesystem."""
        ...


@runtime_checkable
class EnvironmentBinaryFilesystem(Protocol):
    """Binary filesystem operations for environment-to-environment transfer."""

    async def read_bytes(self, path: str) -> bytes:
        """Read raw bytes from the environment filesystem."""
        ...

    async def write_bytes(self, path: str, content: bytes) -> None:
        """Write raw bytes to the environment filesystem, creating parents as needed."""
        ...


@runtime_checkable
class EnvironmentFilesystemWithBytes(
    EnvironmentFilesystem,
    EnvironmentBinaryFilesystem,
    Protocol,
):
    """Filesystem contract that supports text tools and binary-safe transfer."""

    pass


class ShellEnvironmentWithFilesystem(ShellEnvironment, EnvironmentFilesystem, Protocol):
    """Environment that owns both shell execution and model-facing files."""

    pass


async def execute_shell(
    environment: ShellEnvironment,
    command: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> ShellExecutionResult:
    """Execute a command through a ``ShellEnvironment`` and return its result."""

    execution = await environment.execute(
        ShellExecutionRequest(
            command=command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            timeout=timeout,
        )
    )
    return execution.result

__all__ = [
    "EnvironmentFileEntry",
    "EnvironmentFileKind",
    "EnvironmentBinaryFilesystem",
    "EnvironmentFilesystem",
    "EnvironmentFilesystemWithBytes",
    "RuntimeEnvironmentKind",
    "ShellEnvironment",
    "ShellEnvironmentWithFilesystem",
    "ShellExecutor",
    "ShellExecution",
    "ShellExecutionCallbacks",
    "ShellExecutionOptions",
    "ShellExecutionRequest",
    "ShellExecutionResult",
    "ShellRuntimeInfo",
    "execute_shell",
]
