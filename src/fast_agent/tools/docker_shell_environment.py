"""Docker-backed shell environments for harness execution."""

from __future__ import annotations

import asyncio
import posixpath
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Literal

from fast_agent.tools.session_environment import (
    EnvironmentFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_STREAM_READ_CHUNK_SIZE = 4096
_PROCESS_EXIT_POLL_SECONDS = 0.1
_IDLE_POLL_SECONDS = 1.0

DockerMountMode = Literal["ro", "rw"]


@dataclass(frozen=True, slots=True)
class DockerMount:
    """Host path mounted into a managed Docker shell environment."""

    source: str | Path
    target: str
    mode: DockerMountMode = "rw"

    def docker_arg(self) -> str:
        return f"{Path(self.source).resolve()}:{self.target}:{self.mode}"


@dataclass(slots=True)
class _DockerOutputCapture:
    stdout_segments: list[str]
    stderr_segments: list[str]
    last_output_time: float
    timed_out: bool = False
    exit_code: int = 0

    @classmethod
    def create(cls) -> _DockerOutputCapture:
        return cls(stdout_segments=[], stderr_segments=[], last_output_time=time.monotonic())

    @property
    def result(self) -> ShellExecutionResult:
        return ShellExecutionResult(
            stdout="".join(self.stdout_segments),
            stderr="".join(self.stderr_segments),
            exit_code=self.exit_code,
        )


class DockerShellEnvironment:
    """Execute shell commands in an existing Docker container."""

    def __init__(
        self,
        *,
        container: str,
        container_cli: str = "docker",
        shell: str = "bash",
        cwd: str = "/workspace",
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
    ) -> None:
        self._container = container
        self._container_cli = container_cli
        self._shell = shell
        self._cwd = cwd
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return self._cwd

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(
            name=self._shell,
            kind="docker",
            provider=self._container_cli,
        )

    async def execute(
        self,
        request: ShellExecutionRequest,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> ShellExecution:
        options = ShellExecutionOptions(
            timeout_seconds=self._timeout_seconds
            if request.timeout is None
            else request.timeout,
            warning_interval_seconds=self._warning_interval_seconds,
        )
        argv = self._exec_argv(request)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output = _DockerOutputCapture.create()
        stdout_task = asyncio.create_task(
            self._stream_output(
                process.stdout,
                output=output,
                callbacks=callbacks,
                is_stderr=False,
            )
        )
        stderr_task = asyncio.create_task(
            self._stream_output(
                process.stderr,
                output=output,
                callbacks=callbacks,
                is_stderr=True,
            )
        )

        try:
            output.exit_code = await self._wait_for_exit(
                process,
                options=options,
                output=output,
                callbacks=callbacks,
            )
        except asyncio.CancelledError:
            await self._terminate_process(process)
            raise
        drain_timed_out = await self._drain_output_tasks([stdout_task, stderr_task])
        return ShellExecution(
            result=output.result,
            options=options,
            timed_out=output.timed_out,
            io_drain_timed_out=drain_timed_out,
        )

    def _exec_argv(self, request: ShellExecutionRequest) -> list[str]:
        cwd = request.cwd or self._cwd
        env_args = [
            item
            for name, value in (request.env or {}).items()
            for item in ("-e", f"{name}={value}")
        ]
        base = [self._container_cli, "exec", *env_args, "-w", cwd, self._container, self._shell]
        if self._shell in {"pwsh", "powershell"}:
            return [*base, "-NoLogo", "-NoProfile", "-Command", request.command]
        return [*base, "-lc", request.command]

    async def _stream_output(
        self,
        stream: asyncio.StreamReader | None,
        *,
        output: _DockerOutputCapture,
        callbacks: ShellExecutionCallbacks | None,
        is_stderr: bool,
    ) -> None:
        if stream is None:
            return

        while True:
            chunk = await stream.read(_STREAM_READ_CHUNK_SIZE)
            if not chunk:
                return
            text = chunk.decode(errors="replace")
            output.last_output_time = time.monotonic()
            if is_stderr:
                output.stderr_segments.append(text)
                if callbacks is not None:
                    await callbacks.on_stderr(text)
            else:
                output.stdout_segments.append(text)
                if callbacks is not None:
                    await callbacks.on_stdout(text)

    async def _wait_for_exit(
        self,
        process: asyncio.subprocess.Process,
        *,
        options: ShellExecutionOptions,
        output: _DockerOutputCapture,
        callbacks: ShellExecutionCallbacks | None,
    ) -> int:
        last_warning_time = 0.0
        timeout_seconds = options.timeout_seconds
        warning_interval_seconds = options.warning_interval_seconds
        while process.returncode is None:
            await asyncio.sleep(_PROCESS_EXIT_POLL_SECONDS)
            if timeout_seconds is None:
                continue

            elapsed = time.monotonic() - output.last_output_time
            remaining = timeout_seconds - elapsed
            if (
                warning_interval_seconds is not None
                and elapsed - last_warning_time >= warning_interval_seconds
                and remaining > 0
            ):
                if callbacks is not None:
                    await callbacks.on_idle_warning(elapsed, remaining)
                last_warning_time = elapsed
            if elapsed < timeout_seconds:
                continue

            output.timed_out = True
            if callbacks is not None:
                await callbacks.on_timeout()
            await self._terminate_process(process)
            break

        return process.returncode if process.returncode is not None else -1

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            process.kill()

    async def _drain_output_tasks(self, tasks: list[asyncio.Task[None]]) -> bool:
        done, pending = await asyncio.wait(tasks, timeout=2)
        for task in done:
            await task
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return bool(pending)

    async def close(self) -> None:
        return None


class DockerManagedShellEnvironment(DockerShellEnvironment):
    """Own a disposable Docker container for shell execution."""

    def __init__(
        self,
        *,
        image: str,
        container_cli: str = "docker",
        shell: str = "bash",
        cwd: str = "/workspace",
        mounts: Sequence[DockerMount] = (),
        remove: bool = True,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
    ) -> None:
        self._image = image
        self._mounts = tuple(mounts)
        self._remove = remove
        self._owned_container: str | None = None
        super().__init__(
            container="",
            container_cli=container_cli,
            shell=shell,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
        )

    async def open(self) -> None:
        if self._owned_container is not None:
            return
        container = f"fast-agent-{uuid.uuid4().hex[:12]}"
        argv = [
            self._container_cli,
            "run",
            "--name",
            container,
            "-d",
            "-w",
            self.cwd,
            *self._mount_args(),
            self._image,
            "sleep",
            "infinity",
        ]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode(errors="replace") or stdout.decode(errors="replace")
            raise RuntimeError(
                f"Failed to start {self._container_cli} shell environment: {message.strip()}"
            )
        self._owned_container = container
        self._container = container

    def _mount_args(self) -> list[str]:
        return [item for mount in self._mounts for item in ("-v", mount.docker_arg())]

    async def close(self) -> None:
        container = self._owned_container
        if container is None:
            return
        argv = (
            [self._container_cli, "rm", "-f", container]
            if self._remove
            else [self._container_cli, "stop", container]
        )
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        self._owned_container = None
        self._container = ""


class DockerMountedEnvironment(DockerManagedShellEnvironment):
    """Docker shell environment whose file tools target one bind mount."""

    def __init__(
        self,
        *,
        image: str,
        workspace: str | Path,
        target: str = "/workspace",
        container_cli: str = "docker",
        shell: str = "bash",
        remove: bool = True,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
    ) -> None:
        self._host_workspace = Path(workspace).resolve()
        self._target = _normalize_container_path(target)
        super().__init__(
            image=image,
            container_cli=container_cli,
            shell=shell,
            cwd=self._target,
            mounts=(DockerMount(self._host_workspace, self._target, "rw"),),
            remove=remove,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
        )

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return _normalize_container_path(path)
        return _normalize_container_path(posixpath.join(self.cwd, path))

    async def read_text(self, path: str) -> str:
        return self._host_path(path).read_text(encoding="utf-8", errors="replace")

    async def write_text(self, path: str, content: str) -> None:
        host_path = self._host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(content, encoding="utf-8")

    async def exists(self, path: str) -> bool:
        return self._host_path(path).exists()

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        host_dir = self._host_path(path)
        container_dir = self.resolve_path(path)
        entries: list[EnvironmentFileEntry] = []
        for child in sorted(host_dir.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                kind = "directory"
            elif child.is_file():
                kind = "file"
            else:
                kind = "other"
            entries.append(
                EnvironmentFileEntry(
                    path=_normalize_container_path(posixpath.join(container_dir, child.name)),
                    name=child.name,
                    kind=kind,
                )
            )
        return entries

    async def mkdir(self, path: str) -> None:
        self._host_path(path).mkdir(parents=True, exist_ok=True)

    async def remove(self, path: str) -> None:
        host_path = self._host_path(path)
        if host_path.is_dir():
            rmtree(host_path)
            return
        host_path.unlink()

    def _host_path(self, path: str) -> Path:
        container_path = self.resolve_path(path)
        relative = _relative_to_mount(container_path, self._target)
        return self._host_workspace / relative


__all__ = [
    "DockerMountedEnvironment",
    "DockerManagedShellEnvironment",
    "DockerMount",
    "DockerMountMode",
    "DockerShellEnvironment",
]


def _normalize_container_path(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return posixpath.normpath(path)


def _relative_to_mount(path: str, mount_target: str) -> Path:
    normalized_path = _normalize_container_path(path)
    normalized_target = _normalize_container_path(mount_target)
    if normalized_path == normalized_target:
        return Path()
    prefix = f"{normalized_target}/"
    if not normalized_path.startswith(prefix):
        raise ValueError(
            f"Path {path!r} is outside the mounted Docker workspace {normalized_target!r}."
        )
    return Path(normalized_path[len(prefix) :])
