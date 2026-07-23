"""Local structured shell execution below model-facing tool adapters."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any, BinaryIO

from fast_agent.core.logging.logger import Logger
from fast_agent.home import build_child_environment
from fast_agent.tools.execution_environment import (
    EnvironmentFileEntry,
    ShellExecution,
    ShellExecutionCallbacks,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)
from fast_agent.tools.shell_output_spool import (
    ShellOutputSpoolPaths,
    ShellOutputSpoolTailer,
    create_local_output_spool,
    delete_local_output_spool,
    open_local_output_spool,
    read_local_output_chunk,
)
from fast_agent.utils.shell_detection import shell_runtime_info
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.config import Settings

_STREAM_READ_CHUNK_SIZE = 4096
_MAX_PENDING_STREAM_BYTES = 65536
_IO_DRAIN_TIMEOUT_SECONDS = 2.0
_PROCESS_TERMINATION_GRACE_SECONDS = 2.0
_PROCESS_EXIT_POLL_SECONDS = 0.1
_WATCHDOG_POLL_SECONDS = 1.0
_asyncio_sleep = asyncio.sleep


type ShellEnvironmentLogger = logging.Logger | Logger


@dataclass(frozen=True, slots=True)
class _ShellProcessPlan:
    working_dir: Path
    shell_name: str
    shell_path: str | None
    is_windows: bool
    process_kwargs: dict[str, Any]
    output_spool: ShellOutputSpoolPaths | None = None
    output_files: tuple[BinaryIO, BinaryIO] | None = None


@dataclass(slots=True)
class _ShellOutputCapture:
    stdout_segments: list[str] = field(default_factory=list)
    stderr_segments: list[str] = field(default_factory=list)
    last_output_time: float = field(default_factory=time.monotonic)
    timeout_occurred: bool = False
    exit_code: int = 0
    retain_output: bool = True

    @property
    def result(self) -> ShellExecutionResult:
        return ShellExecutionResult(
            stdout="".join(self.stdout_segments),
            stderr="".join(self.stderr_segments),
            exit_code=self.exit_code,
        )


class LocalShellExecutor:
    """Execute local shell commands and return structured stdout/stderr results."""

    def __init__(
        self,
        *,
        logger: ShellEnvironmentLogger,
        timeout_seconds: float = 90,
        warning_interval_seconds: int = 30,
        working_directory: Path | None = None,
        config: Settings | None = None,
        default_env: Mapping[str, str] | None = None,
    ) -> None:
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._working_directory = working_directory
        self._config = config
        self._default_env = dict(default_env or {})

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def warning_interval_seconds(self) -> int:
        return self._warning_interval_seconds

    async def open(self) -> None:
        return None

    @property
    def cwd(self) -> str:
        return str(self.working_directory())

    def working_directory(self) -> Path:
        if self._working_directory is not None:
            return self._working_directory
        return Path.cwd()

    def set_working_directory(self, working_directory: Path | None) -> None:
        self._working_directory = working_directory

    @staticmethod
    def resolve_working_directory(path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (Path.cwd() / path).resolve()

    @classmethod
    def validate_working_directory(cls, configured_path: Path) -> str | None:
        resolved_path = cls.resolve_working_directory(configured_path)

        if not resolved_path.exists():
            return " ".join(
                [
                    f"Shell working directory does not exist: {resolved_path}.",
                    f"Configured cwd: {configured_path}.",
                    "Check the agent card 'cwd' setting or create the directory.",
                ]
            )

        if not resolved_path.is_dir():
            return " ".join(
                [
                    f"Shell working directory is not a directory: {resolved_path}.",
                    f"Configured cwd: {configured_path}.",
                    "Check the agent card 'cwd' setting.",
                ]
            )

        return None

    def runtime_info(self) -> ShellRuntimeInfo:
        info = shell_runtime_info()
        return ShellRuntimeInfo(
            name=info.get("name") or "shell",
            path=info.get("path"),
            kind="local",
        )

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        execution = await self.execute(
            ShellExecutionRequest(
                command=command,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                timeout=timeout,
            )
        )
        return execution.result

    async def execute(
        self,
        request: ShellExecutionRequest | str,
        *,
        callbacks: ShellExecutionCallbacks | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecution:
        if isinstance(request, str):
            request = ShellExecutionRequest(
                command=request,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                timeout=timeout,
            )
        options = ShellExecutionOptions(
            timeout_seconds=(
                self._timeout_seconds if request.timeout is None else request.timeout
            )
            if request.terminate_after_idle
            else None,
            warning_interval_seconds=self._warning_interval_seconds,
        )
        configured_working_dir = (
            self.working_directory() if request.cwd is None else Path(request.cwd)
        )
        working_dir_error = self.validate_working_directory(configured_working_dir)
        if working_dir_error:
            raise ValueError(working_dir_error)

        plan = self._build_process_plan(
            configured_working_dir,
            env=request.env,
            detach=request.detach,
        )
        try:
            try:
                process = await self._start_shell_process(request.command, plan)
            finally:
                # Close before any spool deletion: Windows cannot remove a
                # directory containing open files.
                if plan.output_files is not None:
                    for output_file in plan.output_files:
                        output_file.close()
        except BaseException:
            if plan.output_spool is not None:
                delete_local_output_spool(plan.output_spool)
            raise
        if plan.output_spool is not None:
            request.output_spool_path = plan.output_spool.directory
        if callbacks is not None:
            await callbacks.on_started(process.pid)
        output = _ShellOutputCapture(retain_output=request.retain_output)

        if plan.output_spool is not None:
            async def on_stdout(text: str) -> None:
                await self._record_stream_output(
                    text,
                    output=output,
                    callbacks=callbacks,
                    is_stderr=False,
                )

            async def on_stderr(text: str) -> None:
                await self._record_stream_output(
                    text,
                    output=output,
                    callbacks=callbacks,
                    is_stderr=True,
                )

            async def process_exited() -> bool:
                return process.returncode is not None

            tailer = ShellOutputSpoolTailer(
                plan.output_spool,
                read_chunk=read_local_output_chunk,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
            output_tasks = [
                asyncio.create_task(
                    tailer.tail_until(
                        process_exited,
                        poll_interval=_PROCESS_EXIT_POLL_SECONDS,
                    )
                )
            ]
        else:
            output_tasks = [
                asyncio.create_task(
                    self._stream_process_output(
                        process.stdout,
                        output=output,
                        callbacks=callbacks,
                        is_stderr=False,
                    )
                ),
                asyncio.create_task(
                    self._stream_process_output(
                        process.stderr,
                        output=output,
                        callbacks=callbacks,
                        is_stderr=True,
                    )
                ),
            ]
        watchdog_task = asyncio.create_task(
            self._watch_process_timeout(
                process,
                is_windows=plan.is_windows,
                options=options,
                output=output,
                callbacks=callbacks,
            )
        )

        try:
            output.exit_code = await self._wait_for_process_exit(process)
        except asyncio.CancelledError:
            if request.terminate_on_cancel:
                await self._terminate_cancelled_process(
                    process,
                    is_windows=plan.is_windows,
                )
                try:
                    await self._drain_output_tasks(
                        output_tasks,
                        timeout_seconds=_IO_DRAIN_TIMEOUT_SECONDS,
                    )
                finally:
                    if plan.output_spool is not None:
                        delete_local_output_spool(plan.output_spool)
                        request.output_spool_path = None
            else:
                for task in output_tasks:
                    await self._cancel_task_if_running(task)
            raise
        finally:
            await self._cancel_task_if_running(watchdog_task)
        try:
            drain_timed_out = await self._drain_output_tasks(
                output_tasks,
                timeout_seconds=_IO_DRAIN_TIMEOUT_SECONDS,
            )
        finally:
            if plan.output_spool is not None:
                delete_local_output_spool(plan.output_spool)
                request.output_spool_path = None
        return ShellExecution(
            result=output.result,
            options=options,
            timed_out=output.timeout_occurred,
            io_drain_timed_out=drain_timed_out,
        )

    def _build_process_plan(
        self,
        configured_working_dir: Path,
        *,
        env: Mapping[str, str] | None = None,
        detach: bool = False,
    ) -> _ShellProcessPlan:
        working_dir = self.resolve_working_directory(configured_working_dir)
        runtime_details = self.runtime_info()
        shell_name = strip_casefold(runtime_details.name)
        shell_path = runtime_details.path
        is_windows = platform.system() == "Windows"
        child_env = build_child_environment(
            active_home=getattr(self._config, "_fast_agent_home", None),
            no_home=bool(getattr(self._config, "_fast_agent_no_home", False)),
        )
        if self._default_env:
            child_env.update(self._default_env)
        if env is not None:
            child_env.update(env)
        output_spool = create_local_output_spool() if detach else None
        try:
            output_files = (
                open_local_output_spool(output_spool)
                if output_spool is not None
                else None
            )
        except BaseException:
            if output_spool is not None:
                delete_local_output_spool(output_spool)
            raise
        process_kwargs: dict[str, Any] = {
            "stdout": (
                output_files[0]
                if output_files is not None
                else asyncio.subprocess.PIPE
            ),
            "stderr": (
                output_files[1]
                if output_files is not None
                else asyncio.subprocess.PIPE
            ),
            "cwd": working_dir,
            "env": child_env,
        }
        if is_windows:
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creation_flags:
                process_kwargs["creationflags"] = creation_flags
        else:
            process_kwargs["start_new_session"] = True
        return _ShellProcessPlan(
            working_dir=working_dir,
            shell_name=shell_name,
            shell_path=shell_path,
            is_windows=is_windows,
            process_kwargs=process_kwargs,
            output_spool=output_spool,
            output_files=output_files,
        )

    async def _start_shell_process(
        self,
        command: str,
        plan: _ShellProcessPlan,
    ) -> asyncio.subprocess.Process:
        if plan.is_windows and plan.shell_path and plan.shell_name in {"pwsh", "powershell"}:
            return await asyncio.create_subprocess_exec(
                plan.shell_path,
                "-NoLogo",
                "-NoProfile",
                "-Command",
                command,
                **plan.process_kwargs,
            )

        process_kwargs = dict(plan.process_kwargs)
        if plan.shell_path:
            process_kwargs["executable"] = plan.shell_path
        return await asyncio.create_subprocess_shell(command, **process_kwargs)

    async def _stream_process_output(
        self,
        stream: asyncio.StreamReader | None,
        *,
        output: _ShellOutputCapture,
        callbacks: ShellExecutionCallbacks | None,
        is_stderr: bool,
    ) -> None:
        if stream is None:
            return

        pending = bytearray()
        while True:
            chunk = await stream.read(_STREAM_READ_CHUNK_SIZE)
            if not chunk:
                if pending:
                    await self._record_stream_output(
                        pending.decode(errors="replace"),
                        output=output,
                        callbacks=callbacks,
                        is_stderr=is_stderr,
                    )
                break
            pending.extend(chunk)

            while pending:
                newline_index = pending.find(b"\n")
                if newline_index >= 0:
                    line = bytes(pending[: newline_index + 1])
                    del pending[: newline_index + 1]
                    await self._record_stream_output(
                        line.decode(errors="replace"),
                        output=output,
                        callbacks=callbacks,
                        is_stderr=is_stderr,
                    )
                    continue

                if len(pending) < _MAX_PENDING_STREAM_BYTES:
                    break

                line = bytes(pending[:_MAX_PENDING_STREAM_BYTES])
                del pending[:_MAX_PENDING_STREAM_BYTES]
                await self._record_stream_output(
                    line.decode(errors="replace"),
                    output=output,
                    callbacks=callbacks,
                    is_stderr=is_stderr,
                )

    async def _record_stream_output(
        self,
        text: str,
        *,
        output: _ShellOutputCapture,
        callbacks: ShellExecutionCallbacks | None,
        is_stderr: bool,
    ) -> None:
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
        output.last_output_time = time.monotonic()

    async def _watch_process_timeout(
        self,
        process: asyncio.subprocess.Process,
        *,
        is_windows: bool,
        options: ShellExecutionOptions,
        output: _ShellOutputCapture,
        callbacks: ShellExecutionCallbacks | None,
    ) -> None:
        last_warning_time = 0.0
        self._logger.debug(
            "Watchdog started: "
            f"timeout={options.timeout_seconds}s, "
            f"warning_interval={options.warning_interval_seconds}s"
        )

        while True:
            await asyncio.sleep(_WATCHDOG_POLL_SECONDS)
            if process.returncode is not None:
                self._logger.debug("Watchdog: process exited normally")
                return

            timeout_seconds = options.timeout_seconds
            warning_interval_seconds = options.warning_interval_seconds
            if timeout_seconds is None:
                continue
            elapsed = time.monotonic() - output.last_output_time
            remaining = timeout_seconds - elapsed
            time_since_warning = elapsed - last_warning_time
            if (
                warning_interval_seconds is not None
                and time_since_warning >= warning_interval_seconds
                and remaining > 0
            ):
                self._logger.debug(f"Watchdog: warning at {int(remaining)}s remaining")
                if callbacks is not None:
                    await callbacks.on_idle_warning(elapsed, remaining)
                last_warning_time = elapsed

            if elapsed < timeout_seconds:
                continue

            output.timeout_occurred = True
            self._logger.debug("Watchdog: timeout exceeded, terminating process group")
            if callbacks is not None:
                await callbacks.on_timeout()
            await self._terminate_timed_out_process(process, is_windows=is_windows)
            return

    async def _terminate_timed_out_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        is_windows: bool,
    ) -> None:
        try:
            if is_windows:
                await self._terminate_windows_process(process)
            else:
                await self._terminate_unix_process(process)
        except (ProcessLookupError, OSError):
            return
        except Exception as exc:
            self._logger.debug(f"Error terminating process: {exc}")
            try:
                process.kill()
            except Exception:
                return

    async def _terminate_cancelled_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        is_windows: bool,
    ) -> None:
        self._logger.debug("Shell execution cancelled, terminating process group")
        await self._terminate_timed_out_process(process, is_windows=is_windows)

    async def _terminate_windows_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                process.send_signal(ctrl_break)
                if await self._wait_for_termination(process):
                    return
        except AttributeError:
            self._logger.debug("Watchdog: CTRL_BREAK_EVENT unsupported, skipping")
        except ValueError:
            self._logger.debug("Watchdog: no console attached for CTRL_BREAK_EVENT")
        except ProcessLookupError:
            return

        if process.returncode is None:
            process.terminate()
            if await self._wait_for_termination(process):
                return
        if process.returncode is None:
            process.kill()
            await process.wait()

    async def _terminate_unix_process(self, process: asyncio.subprocess.Process) -> None:
        os.killpg(process.pid, signal.SIGTERM)
        if await self._wait_for_termination(process):
            return
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    @staticmethod
    async def _wait_for_termination(process: asyncio.subprocess.Process) -> bool:
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_PROCESS_TERMINATION_GRACE_SECONDS,
            )
        except TimeoutError:
            return False
        return True

    async def _cancel_task_if_running(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _drain_output_tasks(
        self,
        tasks: list[asyncio.Task[None]],
        *,
        timeout_seconds: float,
    ) -> bool:
        done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        drain_timed_out = bool(pending)
        try:
            for task in done:
                await task
        finally:
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task
        return drain_timed_out

    async def _wait_for_process_exit(
        self,
        process: asyncio.subprocess.Process,
    ) -> int:
        while process.returncode is None:
            await _asyncio_sleep(_PROCESS_EXIT_POLL_SECONDS)
        return process.returncode

    async def close(self) -> None:
        return None


class LocalEnvironment(LocalShellExecutor):
    """Local shell environment with filesystem operations rooted at its cwd."""

    def resolve_path(self, path: str) -> str:
        return str(self._resolve_filesystem_path(path))

    async def read_text(self, path: str) -> str:
        return self._resolve_filesystem_path(path).read_text(encoding="utf-8", errors="replace")

    async def write_text(self, path: str, content: str) -> None:
        resolved = self._resolve_filesystem_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    async def read_bytes(self, path: str) -> bytes:
        return self._resolve_filesystem_path(path).read_bytes()

    async def write_bytes(self, path: str, content: bytes) -> None:
        resolved = self._resolve_filesystem_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)

    async def exists(self, path: str) -> bool:
        return self._resolve_filesystem_path(path).exists()

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        resolved_dir = self._resolve_filesystem_path(path)
        entries: list[EnvironmentFileEntry] = []
        for child in sorted(resolved_dir.iterdir(), key=lambda item: item.name):
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
                    path=str(child),
                    name=child.name,
                    kind=kind,
                )
            )
        return entries

    async def mkdir(self, path: str) -> None:
        self._resolve_filesystem_path(path).mkdir(parents=True, exist_ok=True)

    async def remove(self, path: str) -> None:
        resolved = self._resolve_filesystem_path(path)
        if resolved.is_dir():
            rmtree(resolved)
            return
        resolved.unlink()

    def _resolve_filesystem_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.working_directory() / candidate).resolve()


__all__ = [
    "LocalEnvironment",
    "LocalShellExecutor",
    "ShellEnvironmentLogger",
]
