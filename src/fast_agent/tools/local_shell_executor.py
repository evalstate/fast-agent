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
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.core.logging.logger import Logger
from fast_agent.home import build_child_environment
from fast_agent.tools.session_environment import ShellExecutionResult
from fast_agent.utils.shell_detection import shell_runtime_info
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.config import Settings

_STREAM_READ_CHUNK_SIZE = 4096
_MAX_PENDING_STREAM_BYTES = 65536
_IO_DRAIN_TIMEOUT_SECONDS = 2.0
_PROCESS_EXIT_POLL_SECONDS = 0.1
_asyncio_sleep = asyncio.sleep


class ShellExecutionCallbacks(Protocol):
    """Optional observer hooks for streaming shell execution."""

    async def on_stdout(self, text: str) -> None: ...

    async def on_stderr(self, text: str) -> None: ...

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None: ...

    async def on_timeout(self) -> None: ...


type ShellExecutorLogger = logging.Logger | Logger


@dataclass(frozen=True, slots=True)
class ShellExecutionOptions:
    timeout_seconds: float
    warning_interval_seconds: int


@dataclass(frozen=True, slots=True)
class LocalShellExecution:
    result: ShellExecutionResult
    options: ShellExecutionOptions
    timed_out: bool = False
    io_drain_timed_out: bool = False


@dataclass(frozen=True, slots=True)
class _ShellProcessPlan:
    working_dir: Path
    shell_name: str
    shell_path: str | None
    is_windows: bool
    process_kwargs: dict[str, Any]


@dataclass(slots=True)
class _ShellOutputCapture:
    stdout_segments: list[str] = field(default_factory=list)
    stderr_segments: list[str] = field(default_factory=list)
    last_output_time: float = field(default_factory=time.monotonic)
    timeout_occurred: bool = False
    exit_code: int = 0

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
        logger: ShellExecutorLogger,
        timeout_seconds: int = 90,
        warning_interval_seconds: int = 30,
        working_directory: Path | None = None,
        config: Settings | None = None,
    ) -> None:
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._working_directory = working_directory
        self._config = config

    @property
    def timeout_seconds(self) -> int:
        return self._timeout_seconds

    @property
    def warning_interval_seconds(self) -> int:
        return self._warning_interval_seconds

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

    def runtime_info(self) -> dict[str, str | None]:
        return shell_runtime_info()

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        execution = await self.execute(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        return execution.result

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        callbacks: ShellExecutionCallbacks | None = None,
    ) -> LocalShellExecution:
        options = ShellExecutionOptions(
            timeout_seconds=self._timeout_seconds if timeout is None else timeout,
            warning_interval_seconds=self._warning_interval_seconds,
        )
        configured_working_dir = self.working_directory() if cwd is None else Path(cwd)
        working_dir_error = self.validate_working_directory(configured_working_dir)
        if working_dir_error:
            raise ValueError(working_dir_error)

        plan = self._build_process_plan(configured_working_dir, env=env)
        process = await self._start_shell_process(command, plan)
        output = _ShellOutputCapture()

        stdout_task = asyncio.create_task(
            self._stream_process_output(
                process.stdout,
                output=output,
                callbacks=callbacks,
                is_stderr=False,
            )
        )
        stderr_task = asyncio.create_task(
            self._stream_process_output(
                process.stderr,
                output=output,
                callbacks=callbacks,
                is_stderr=True,
            )
        )
        watchdog_task = asyncio.create_task(
            self._watch_process_timeout(
                process,
                is_windows=plan.is_windows,
                options=options,
                output=output,
                callbacks=callbacks,
            )
        )

        output.exit_code = await self._wait_for_process_exit(process)
        await self._cancel_task_if_running(watchdog_task)
        drain_timed_out = await self._drain_output_tasks(
            [stdout_task, stderr_task],
            timeout_seconds=_IO_DRAIN_TIMEOUT_SECONDS,
        )
        return LocalShellExecution(
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
    ) -> _ShellProcessPlan:
        working_dir = self.resolve_working_directory(configured_working_dir)
        runtime_details = self.runtime_info()
        shell_name = strip_casefold(str(runtime_details.get("name") or ""))
        shell_path = runtime_details.get("path")
        is_windows = platform.system() == "Windows"
        child_env = build_child_environment(
            active_home=getattr(self._config, "_fast_agent_home", None),
            noenv=bool(getattr(self._config, "_fast_agent_noenv", False)),
        )
        if env is not None:
            child_env.update(env)
        process_kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
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
            output.stderr_segments.append(text)
            if callbacks is not None:
                await callbacks.on_stderr(text)
        else:
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
            await asyncio.sleep(1)
            if process.returncode is not None:
                self._logger.debug("Watchdog: process exited normally")
                return

            elapsed = time.monotonic() - output.last_output_time
            remaining = options.timeout_seconds - elapsed
            time_since_warning = elapsed - last_warning_time
            if time_since_warning >= options.warning_interval_seconds and remaining > 0:
                self._logger.debug(f"Watchdog: warning at {int(remaining)}s remaining")
                if callbacks is not None:
                    await callbacks.on_idle_warning(elapsed, remaining)
                last_warning_time = elapsed

            if elapsed < options.timeout_seconds:
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

    async def _terminate_windows_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                process.send_signal(ctrl_break)
            await asyncio.sleep(2)
        except AttributeError:
            self._logger.debug("Watchdog: CTRL_BREAK_EVENT unsupported, skipping")
        except ValueError:
            self._logger.debug("Watchdog: no console attached for CTRL_BREAK_EVENT")
        except ProcessLookupError:
            return

        if process.returncode is None:
            process.terminate()
            await asyncio.sleep(2)
        if process.returncode is None:
            process.kill()

    async def _terminate_unix_process(self, process: asyncio.subprocess.Process) -> None:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.sleep(2)
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)

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


__all__ = [
    "LocalShellExecution",
    "LocalShellExecutor",
    "ShellExecutionCallbacks",
    "ShellExecutionOptions",
]
