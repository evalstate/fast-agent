from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import sys
import time
from pathlib import Path
from shutil import rmtree

import pytest
from mcp.types import TextContent

import fast_agent.tools.shell_runtime as shell_runtime_module
from fast_agent.config import Settings, ShellSettings
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.shell_runtime import ShellRuntime


class _StartedCallbacks:
    def __init__(self, request: ShellExecutionRequest) -> None:
        self._request = request
        self.started = asyncio.Event()
        self.process_id: int | None = None
        self.output_spool_path: str | None = None

    async def on_started(self, process_id: int | None) -> None:
        self.process_id = process_id
        self.output_spool_path = self._request.output_spool_path
        self.started.set()

    async def on_stdout(self, text: str) -> None:
        del text

    async def on_stderr(self, text: str) -> None:
        del text

    async def on_idle_warning(self, elapsed: float, remaining: float) -> None:
        del elapsed, remaining

    async def on_timeout(self) -> None:
        return None


async def _wait_for_file_growth(path: Path, *, initial_size: int = -1) -> int:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1
        if size > initial_size:
            return size
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path} did not grow")


@pytest.mark.asyncio
async def test_persistent_background_output_reaches_poll_buffer(tmp_path: Path) -> None:
    script = tmp_path / "ticker.py"
    script.write_text(
        "\n".join(
            [
                "import sys, time",
                "time.sleep(0.3)",
                "print('ticker stdout', flush=True)",
                "print('ticker stderr', file=sys.stderr, flush=True)",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    try:
        await runtime.execute(
            {
                "command": f'"{sys.executable}" "{script}"',
                "background": True,
            }
        )
        result = await runtime.poll_process(
            {
                "process_id": "process-1",
                "wait_sec": 3,
                "wake_on_output": True,
            }
        )

        assert result.content
        assert isinstance(result.content[0], TextContent)
        assert "ticker stdout" in result.content[0].text
        assert "[stderr] ticker stderr" in result.content[0].text
        metadata = shell_runtime_module.process_result_metadata(result)
        assert metadata is not None
        assert metadata["output_bytes_since_last_poll"] > 0
        assert metadata["process_yield_reason"] == "output"
        snapshot = (await runtime.process_snapshots())[0]
        assert snapshot.total_output_bytes > 0
        assert snapshot.output_spool_path is not None
        assert Path(snapshot.output_spool_path).is_dir()

        spool_path = Path(snapshot.output_spool_path)
        await runtime.terminate_process({"process_id": "process-1"})
        assert not spool_path.exists()
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix process groups")
async def test_cancelling_persistent_execution_leaves_child_and_spool_running(
    tmp_path: Path,
) -> None:
    script = tmp_path / "persistent.py"
    script.write_text(
        "\n".join(
            [
                "import time",
                "while True:",
                "    print('tick', flush=True)",
                "    time.sleep(0.05)",
            ]
        ),
        encoding="utf-8",
    )
    executor = LocalShellExecutor(
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
    )
    request = ShellExecutionRequest(
        command=f'exec "{sys.executable}" "{script}"',
        terminate_after_idle=False,
        retain_output=False,
        terminate_on_cancel=False,
        detach=True,
    )
    callbacks = _StartedCallbacks(request)
    task = asyncio.create_task(executor.execute(request, callbacks=callbacks))
    await asyncio.wait_for(callbacks.started.wait(), timeout=3)
    assert callbacks.process_id is not None
    assert callbacks.output_spool_path is not None
    spool_path = Path(callbacks.output_spool_path)
    stdout_path = spool_path / "stdout.log"

    try:
        initial_size = await _wait_for_file_growth(stdout_path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        os.kill(callbacks.process_id, 0)
        await _wait_for_file_growth(stdout_path, initial_size=initial_size)
        assert request.output_spool_path == str(spool_path)
    finally:
        try:
            os.killpg(callbacks.process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        rmtree(spool_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_completed_detached_execution_removes_drained_spool(
    tmp_path: Path,
) -> None:
    executor = LocalShellExecutor(
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
    )
    request = ShellExecutionRequest(
        command=f'"{sys.executable}" -c "print(\'complete\')"',
        terminate_after_idle=False,
        detach=True,
    )
    callbacks = _StartedCallbacks(request)

    execution = await executor.execute(request, callbacks=callbacks)

    assert execution.result.stdout == "complete\n"
    assert callbacks.output_spool_path is not None
    assert not Path(callbacks.output_spool_path).exists()
    assert request.output_spool_path is None


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix process groups")
async def test_runtime_close_leaves_real_persistent_background_process_running(
    tmp_path: Path,
) -> None:
    script = tmp_path / "service.py"
    script.write_text(
        "\n".join(
            [
                "import time",
                "while True:",
                "    print('serving', flush=True)",
                "    time.sleep(0.05)",
            ]
        ),
        encoding="utf-8",
    )
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
        }
    )
    snapshot = (await runtime.process_snapshots())[0]
    assert snapshot.os_process_id is not None
    assert snapshot.output_spool_path is not None
    spool_path = Path(snapshot.output_spool_path)
    stdout_path = spool_path / "stdout.log"

    await runtime.close()

    try:
        os.kill(snapshot.os_process_id, 0)
        initial_size = await _wait_for_file_growth(stdout_path)
        await _wait_for_file_growth(stdout_path, initial_size=initial_size)
    finally:
        try:
            os.killpg(snapshot.os_process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        rmtree(spool_path, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.skipif(platform.system() == "Windows", reason="Unix process groups")
async def test_runtime_close_terminates_real_session_background_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "session.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    runtime = ShellRuntime(
        activation_reason="test",
        logger=logging.getLogger(__name__),
        working_directory=tmp_path,
        config=Settings(shell_execution=ShellSettings(show_bash=False)),
    )

    await runtime.execute(
        {
            "command": f'exec "{sys.executable}" "{script}"',
            "background": True,
            "lifecycle": "session",
        }
    )
    snapshot = (await runtime.process_snapshots())[0]
    assert snapshot.os_process_id is not None

    await runtime.close()

    with pytest.raises(ProcessLookupError):
        os.kill(snapshot.os_process_id, 0)
