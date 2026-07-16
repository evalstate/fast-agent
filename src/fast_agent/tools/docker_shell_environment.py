"""Docker-backed shell environments for harness execution."""

from __future__ import annotations

import asyncio
import os
import posixpath
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Callable, Literal

from fast_agent.core.exceptions import EnvironmentStartupError
from fast_agent.tools.execution_environment import (
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
_MANAGED_PROCESS_DISCOVERY_TIMEOUT_SECONDS = 5.0
_MANAGED_PROCESS_TERM_GRACE_SECONDS = 2.0
_MANAGED_PROCESS_TERMINATION_TIMEOUT_SECONDS = 5.0
_MANAGED_PROCESS_ROOT = "/tmp/fast-agent-managed"
_DOCKER_FS_MISSING_EXIT_CODE = 43
_DOCKER_FS_NOT_DIRECTORY_EXIT_CODE = 44
_DOCKER_MANAGED_EXEC_SCRIPT = """
set -eu
pid_file="$1"
shell="$2"
command="$3"
command -v setsid >/dev/null 2>&1 || {
    echo "fast-agent managed Docker execution requires setsid inside the container" >&2
    exit 127
}
mkdir -p -- "$(dirname -- "$pid_file")"
setsid "$shell" -lc "$command" &
child=$!
printf '%s\n' "$child" > "$pid_file"
set +e
wait "$child"
status=$?
set -e
exit "$status"
""".strip()
_DOCKER_LIST_DIR_SCRIPT = """
dir="$1"
[ -e "$dir" ] || exit 43
[ -d "$dir" ] || exit 44
find "$dir" -mindepth 1 -maxdepth 1 -printf '%y\\0%p\\0%f\\0'
""".strip()
_DOCKER_READ_FILE_SCRIPT = """
path="$1"
[ -e "$path" ] || exit 43
exec cat -- "$path"
""".strip()
_DOCKER_WRITE_FILE_SCRIPT = """
path="$1"
parent="$2"
mkdir -p -- "$parent" && cat > "$path"
""".strip()

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
    retain_output: bool
    timed_out: bool = False
    exit_code: int = 0

    @classmethod
    def create(cls, *, retain_output: bool = True) -> _DockerOutputCapture:
        return cls(
            stdout_segments=[],
            stderr_segments=[],
            last_output_time=time.monotonic(),
            retain_output=retain_output,
        )

    @property
    def result(self) -> ShellExecutionResult:
        return ShellExecutionResult(
            stdout="".join(self.stdout_segments),
            stderr="".join(self.stderr_segments),
            exit_code=self.exit_code,
        )


@dataclass(frozen=True, slots=True)
class _DockerExecResult:
    exit_code: int
    stdout: bytes
    stderr: str


class DockerShellEnvironment:
    """Execute shell commands in an existing Docker container."""

    def __init__(
        self,
        *,
        container: str,
        container_cli: str = "docker",
        shell: str = "bash",
        cwd: str = "/workspace",
        default_env: dict[str, str] | None = None,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
    ) -> None:
        self._container = container
        self._container_cli = container_cli
        self._shell = shell
        self._cwd = cwd
        self._default_env = dict(default_env or {})
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._startup_progress_callback: Callable[[str], None] | None = None

    async def open(self) -> None:
        self._emit_startup_stage(f"using existing container {self._container}")
        return None

    def set_startup_progress_callback(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        self._startup_progress_callback = callback

    def _emit_startup_stage(self, stage: str) -> None:
        if self._startup_progress_callback is not None:
            self._startup_progress_callback(stage)

    @property
    def cwd(self) -> str:
        return self._cwd

    def resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return _normalize_container_path(path)
        return _normalize_container_path(posixpath.join(self._cwd, path))

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
        if not request.terminate_after_idle and self._shell in {"pwsh", "powershell"}:
            raise RuntimeError(
                "Managed Docker execution is not supported for PowerShell because "
                "container-side process termination is unavailable."
            )
        options = ShellExecutionOptions(
            timeout_seconds=(
                self._timeout_seconds if request.timeout is None else request.timeout
            )
            if request.terminate_after_idle
            else None,
            warning_interval_seconds=self._warning_interval_seconds,
        )
        managed_pid_file = (
            f"{_MANAGED_PROCESS_ROOT}/{uuid.uuid4().hex}.pid"
            if not request.terminate_after_idle
            else None
        )
        argv = (
            self._managed_exec_argv(request, managed_pid_file)
            if managed_pid_file is not None
            else self._exec_argv(request)
        )
        process_env = self._exec_process_env(request)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=(
                    asyncio.subprocess.PIPE
                    if request.terminate_on_cancel
                    else asyncio.subprocess.DEVNULL
                ),
                stderr=(
                    asyncio.subprocess.PIPE
                    if request.terminate_on_cancel
                    else asyncio.subprocess.DEVNULL
                ),
                env=process_env,
            )
        except FileNotFoundError as exc:
            raise EnvironmentStartupError(
                f"Could not start {self._container_cli} shell environment.",
                f"Container CLI not found: {self._container_cli}. "
                f"Install {self._container_cli} or choose an environment that uses an available CLI.",
            ) from exc
        container_process_id: int | None = None
        if managed_pid_file is not None:
            try:
                container_process_id = await self._discover_managed_process_id(
                    managed_pid_file
                )
            except asyncio.CancelledError:
                if request.terminate_on_cancel:
                    await self._cancel_managed_execution(
                        process,
                        managed_pid_file=managed_pid_file,
                        container_process_id=None,
                    )
                else:
                    await self._terminate_process(process)
                raise
            except BaseException:
                await self._terminate_process(process)
                await self._delete_managed_pid_file(managed_pid_file)
                raise
        if callbacks is not None:
            await callbacks.on_started(
                container_process_id if container_process_id is not None else process.pid
            )
        output = _DockerOutputCapture.create(retain_output=request.retain_output)
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
            if managed_pid_file is not None and request.terminate_on_cancel:
                await self._cancel_managed_execution(
                    process,
                    managed_pid_file=managed_pid_file,
                    container_process_id=container_process_id,
                )
            else:
                await self._terminate_process(process)
            raise
        drain_timed_out = await self._drain_output_tasks([stdout_task, stderr_task])
        if managed_pid_file is not None:
            await self._delete_managed_pid_file(managed_pid_file)
        return ShellExecution(
            result=output.result,
            options=options,
            timed_out=output.timed_out,
            io_drain_timed_out=drain_timed_out,
        )

    def _exec_argv(self, request: ShellExecutionRequest) -> list[str]:
        cwd = request.cwd or self._cwd
        effective_env = dict(self._default_env)
        effective_env.update(request.env or {})
        env_args = [item for name in effective_env for item in ("-e", name)]
        base = [self._container_cli, "exec", *env_args, "-w", cwd, self._container, self._shell]
        if self._shell in {"pwsh", "powershell"}:
            return [*base, "-NoLogo", "-NoProfile", "-Command", request.command]
        return [*base, "-lc", request.command]

    def _managed_exec_argv(
        self,
        request: ShellExecutionRequest,
        managed_pid_file: str,
    ) -> list[str]:
        cwd = request.cwd or self._cwd
        effective_env = dict(self._default_env)
        effective_env.update(request.env or {})
        env_args = [item for name in effective_env for item in ("-e", name)]
        return [
            self._container_cli,
            "exec",
            *env_args,
            "-w",
            cwd,
            self._container,
            "sh",
            "-c",
            _DOCKER_MANAGED_EXEC_SCRIPT,
            "fast-agent-managed",
            managed_pid_file,
            self._shell,
            request.command,
        ]

    def _exec_process_env(self, request: ShellExecutionRequest) -> dict[str, str]:
        effective_env = dict(self._default_env)
        effective_env.update(request.env or {})
        process_env = dict(os.environ)
        process_env.update(effective_env)
        return process_env

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
                if output.retain_output:
                    output.stderr_segments.append(text)
                if callbacks is not None:
                    await callbacks.on_stderr(text)
            else:
                if output.retain_output:
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
        try:
            process.terminate()
        except ProcessLookupError:
            await process.wait()
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    async def _discover_managed_process_id(self, managed_pid_file: str) -> int:
        deadline = time.monotonic() + _MANAGED_PROCESS_DISCOVERY_TIMEOUT_SECONDS
        while True:
            result = await self._docker_exec_bytes(["cat", managed_pid_file])
            if result.exit_code == 0:
                try:
                    process_id = int(result.stdout.strip())
                except ValueError:
                    process_id = 0
                if process_id > 0:
                    return process_id
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Docker managed command did not publish its container-side process ID"
                )
            await asyncio.sleep(_PROCESS_EXIT_POLL_SECONDS)

    async def _cancel_managed_execution(
        self,
        process: asyncio.subprocess.Process,
        *,
        managed_pid_file: str,
        container_process_id: int | None,
    ) -> None:
        try:
            if container_process_id is None:
                container_process_id = await asyncio.shield(
                    self._discover_managed_process_id(managed_pid_file)
                )
            await self._kill_container_process_group(container_process_id)
        finally:
            await self._terminate_process(process)
            await self._delete_managed_pid_file(managed_pid_file)

    async def _kill_container_process_group(self, process_id: int) -> None:
        await self._signal_container_process_group(process_id, signal_name="TERM")
        deadline = time.monotonic() + _MANAGED_PROCESS_TERM_GRACE_SECONDS
        while await self._container_process_is_running(process_id):
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(_PROCESS_EXIT_POLL_SECONDS)
        else:
            return

        await self._signal_container_process_group(process_id, signal_name="KILL")
        deadline = time.monotonic() + (
            _MANAGED_PROCESS_TERMINATION_TIMEOUT_SECONDS
            - _MANAGED_PROCESS_TERM_GRACE_SECONDS
        )
        while await self._container_process_is_running(process_id):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Docker container process group {process_id} remained alive after TERM and KILL"
                )
            await asyncio.sleep(_PROCESS_EXIT_POLL_SECONDS)

    async def _signal_container_process_group(
        self,
        process_id: int,
        *,
        signal_name: str,
    ) -> None:
        result = await self._docker_shell_bytes(
            (
                'kill -"$1" -- -"$2" 2>/dev/null '
                '|| kill -"$1" -"$2" 2>/dev/null '
                '|| kill -"$1" "$2" 2>/dev/null || true'
            ),
            [signal_name, str(process_id)],
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Could not send {signal_name} to Docker container process group {process_id}: "
                f"{result.stderr.strip() or 'container command failed'}"
            )

    async def _container_process_is_running(self, process_id: int) -> bool:
        result = await self._docker_shell_bytes(
            'kill -0 -- -"$1" 2>/dev/null || kill -0 -"$1" 2>/dev/null',
            [str(process_id)],
        )
        return result.exit_code == 0

    async def _delete_managed_pid_file(self, managed_pid_file: str) -> None:
        await self._docker_shell_bytes('rm -f -- "$1"', [managed_pid_file])

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

    async def _docker_exec_bytes(
        self,
        args: list[str],
        *,
        stdin: bytes | None = None,
    ) -> _DockerExecResult:
        argv = [self._container_cli, "exec"]
        if stdin is not None:
            argv.append("-i")
        argv.extend([self._container, *args])
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise EnvironmentStartupError(
                f"Could not start {self._container_cli} filesystem operation.",
                f"Container CLI not found: {self._container_cli}. "
                f"Install {self._container_cli} or choose an environment that uses an available CLI.",
            ) from exc
        stdout, stderr = await process.communicate(stdin)
        return _DockerExecResult(
            exit_code=process.returncode if process.returncode is not None else 1,
            stdout=stdout,
            stderr=stderr.decode(errors="replace"),
        )

    async def _docker_shell_bytes(
        self,
        script: str,
        args: list[str],
        *,
        stdin: bytes | None = None,
    ) -> _DockerExecResult:
        return await self._docker_exec_bytes(
            ["sh", "-c", script, "fast-agent-docker-fs", *args],
            stdin=stdin,
        )

    def _raise_filesystem_error(
        self,
        result: _DockerExecResult,
        *,
        path: str,
        operation: str,
    ) -> None:
        message = result.stderr.strip() or f"Docker filesystem {operation} failed for {path}"
        raise RuntimeError(message)

    async def read_bytes(self, path: str) -> bytes:
        resolved = self.resolve_path(path)
        result = await self._docker_shell_bytes(_DOCKER_READ_FILE_SCRIPT, [resolved])
        if result.exit_code == _DOCKER_FS_MISSING_EXIT_CODE:
            raise FileNotFoundError(resolved)
        if result.exit_code != 0:
            self._raise_filesystem_error(result, path=resolved, operation="read")
        return result.stdout

    async def read_text(self, path: str) -> str:
        return (await self.read_bytes(path)).decode("utf-8", errors="replace")

    async def write_bytes(self, path: str, content: bytes) -> None:
        resolved = self.resolve_path(path)
        parent = posixpath.dirname(resolved) or "/"
        result = await self._docker_shell_bytes(
            _DOCKER_WRITE_FILE_SCRIPT,
            [resolved, parent],
            stdin=content,
        )
        if result.exit_code != 0:
            self._raise_filesystem_error(result, path=resolved, operation="write")

    async def write_text(self, path: str, content: str) -> None:
        await self.write_bytes(path, content.encode("utf-8"))

    async def exists(self, path: str) -> bool:
        resolved = self.resolve_path(path)
        result = await self._docker_shell_bytes('test -e "$1"', [resolved])
        return result.exit_code == 0

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        resolved = self.resolve_path(path)
        result = await self._docker_shell_bytes(_DOCKER_LIST_DIR_SCRIPT, [resolved])
        if result.exit_code == _DOCKER_FS_MISSING_EXIT_CODE:
            raise FileNotFoundError(resolved)
        if result.exit_code == _DOCKER_FS_NOT_DIRECTORY_EXIT_CODE:
            raise NotADirectoryError(resolved)
        if result.exit_code != 0:
            self._raise_filesystem_error(result, path=resolved, operation="list")
        return _parse_docker_directory_entries(result.stdout)

    async def mkdir(self, path: str) -> None:
        resolved = self.resolve_path(path)
        result = await self._docker_shell_bytes('mkdir -p -- "$1"', [resolved])
        if result.exit_code != 0:
            self._raise_filesystem_error(result, path=resolved, operation="mkdir")

    async def remove(self, path: str) -> None:
        resolved = self.resolve_path(path)
        result = await self._docker_shell_bytes('rm -f -- "$1"', [resolved])
        if result.exit_code != 0:
            self._raise_filesystem_error(result, path=resolved, operation="remove")

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
        docker_args: Sequence[str] = (),
        default_env: dict[str, str] | None = None,
        remove: bool = True,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
    ) -> None:
        self._image = image
        self._mounts = tuple(mounts)
        self._docker_args = tuple(docker_args)
        self._remove = remove
        self._owned_container: str | None = None
        super().__init__(
            container="",
            container_cli=container_cli,
            shell=shell,
            cwd=cwd,
            default_env=default_env,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
        )

    async def open(self) -> None:
        if self._owned_container is not None:
            self._emit_startup_stage(f"container already running {self._owned_container}")
            return
        container = f"fast-agent-{uuid.uuid4().hex[:12]}"
        self._emit_startup_stage(f"starting {self._container_cli} container {container}")
        argv = [
            self._container_cli,
            "run",
            "--name",
            container,
            "-d",
            "-w",
            self.cwd,
            *self._mount_args(),
            *self._docker_args,
            self._image,
            "sleep",
            "infinity",
        ]
        try:
            self._emit_startup_stage(f"running {' '.join(argv[:3])} ... {self._image}")
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise EnvironmentStartupError(
                f"Could not start {self._container_cli} shell environment.",
                f"Container CLI not found: {self._container_cli}. "
                f"Install {self._container_cli} or choose an environment that uses an available CLI.",
            ) from exc
        self._emit_startup_stage("waiting for container start")
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode(errors="replace") or stdout.decode(errors="replace")
            raise RuntimeError(
                f"Failed to start {self._container_cli} shell environment: {message.strip()}"
            )
        self._owned_container = container
        self._container = container
        self._emit_startup_stage(f"container ready {container}")

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
        docker_args: Sequence[str] = (),
        default_env: dict[str, str] | None = None,
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
            docker_args=docker_args,
            default_env=default_env,
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

    async def read_bytes(self, path: str) -> bytes:
        return self._host_path(path).read_bytes()

    async def write_bytes(self, path: str, content: bytes) -> None:
        host_path = self._host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_bytes(content)

    async def exists(self, path: str) -> bool:
        return self._host_path(path).exists()

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        host_dir = self._host_path(path)
        container_dir = self.resolve_path(path)
        entries: list[EnvironmentFileEntry] = []
        for child in sorted(host_dir.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                kind = "other"
            elif child.is_dir():
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


def _parse_docker_directory_entries(payload: bytes) -> list[EnvironmentFileEntry]:
    if not payload:
        return []
    parts = payload.split(b"\0")
    if parts[-1] == b"":
        parts = parts[:-1]
    if len(parts) % 3 != 0:
        raise RuntimeError("Docker directory listing returned invalid data.")

    entries: list[EnvironmentFileEntry] = []
    for index in range(0, len(parts), 3):
        type_code = parts[index].decode("ascii", errors="replace")
        path = parts[index + 1].decode("utf-8", errors="replace")
        name = parts[index + 2].decode("utf-8", errors="replace")
        kind = "directory" if type_code == "d" else "file" if type_code == "f" else "other"
        entries.append(EnvironmentFileEntry(path=path, name=name, kind=kind))
    return entries


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
