from __future__ import annotations

import asyncio

import pytest

from fast_agent.tools.shell_output_spool import (
    ShellOutputSpoolPaths,
    ShellOutputSpoolTailer,
)


@pytest.mark.asyncio
async def test_tailer_captures_descendant_output_arriving_after_process_exit() -> None:
    contents = {
        "stdout": bytearray(b"parent\n"),
        "stderr": bytearray(),
    }
    stdout: list[str] = []
    stderr: list[str] = []

    async def read_chunk(path: str, offset: int, size: int) -> bytes:
        return bytes(contents[path][offset : offset + size])

    async def append_descendant_output() -> None:
        await asyncio.sleep(0.03)
        contents["stdout"].extend(b"descendant\n")

    tailer = ShellOutputSpoolTailer(
        ShellOutputSpoolPaths(
            directory="spool",
            stdout="stdout",
            stderr="stderr",
        ),
        read_chunk=read_chunk,
        on_stdout=lambda text: _append_output(stdout, text),
        on_stderr=lambda text: _append_output(stderr, text),
    )
    writer = asyncio.create_task(append_descendant_output())

    await tailer.tail_until(
        lambda: _process_exited(),
        poll_interval=0.01,
        final_grace_seconds=0.1,
    )
    await writer

    assert "".join(stdout) == "parent\ndescendant\n"
    assert stderr == []


@pytest.mark.asyncio
async def test_tailer_skips_final_grace_without_recent_output() -> None:
    async def read_chunk(path: str, offset: int, size: int) -> bytes:
        return b""

    tailer = ShellOutputSpoolTailer(
        ShellOutputSpoolPaths(
            directory="spool",
            stdout="stdout",
            stderr="stderr",
        ),
        read_chunk=read_chunk,
        on_stdout=lambda text: _append_output([], text),
        on_stderr=lambda text: _append_output([], text),
    )

    async with asyncio.timeout(0.1):
        await tailer.tail_until(
            lambda: _process_exited(),
            poll_interval=0,
            final_grace_seconds=1,
        )


@pytest.mark.asyncio
async def test_tailer_emits_complete_lines_across_read_boundaries() -> None:
    contents = {
        "stdout": bytearray(b"one\ntwo\nunterminated"),
        "stderr": bytearray(b"error one\nerror two\n"),
    }
    stdout: list[str] = []
    stderr: list[str] = []

    async def read_chunk(path: str, offset: int, size: int) -> bytes:
        return bytes(contents[path][offset : offset + size])

    tailer = ShellOutputSpoolTailer(
        ShellOutputSpoolPaths(
            directory="spool",
            stdout="stdout",
            stderr="stderr",
        ),
        read_chunk=read_chunk,
        on_stdout=lambda text: _append_output(stdout, text),
        on_stderr=lambda text: _append_output(stderr, text),
        chunk_size=3,
        chunks_per_poll=1,
    )

    await tailer.tail_until(
        lambda: _process_exited(),
        poll_interval=0,
        final_grace_seconds=0,
    )

    assert stdout == ["one\n", "two\n", "unterminated"]
    assert stderr == ["error one\n", "error two\n"]


@pytest.mark.asyncio
async def test_tailer_emits_carriage_return_progress_updates() -> None:
    contents = {
        "stdout": bytearray(b"step 1\rstep 2\rcomplete\n"),
        "stderr": bytearray(),
    }
    stdout: list[str] = []

    async def read_chunk(path: str, offset: int, size: int) -> bytes:
        return bytes(contents[path][offset : offset + size])

    tailer = ShellOutputSpoolTailer(
        ShellOutputSpoolPaths(
            directory="spool",
            stdout="stdout",
            stderr="stderr",
        ),
        read_chunk=read_chunk,
        on_stdout=lambda text: _append_output(stdout, text),
        on_stderr=lambda text: _append_output([], text),
    )

    await tailer.tail_until(
        lambda: _process_exited(),
        poll_interval=0,
        final_grace_seconds=0,
    )

    assert stdout == ["step 1\r", "step 2\r", "complete\n"]


@pytest.mark.asyncio
async def test_tailer_notifies_activity_for_partial_output() -> None:
    contents = {
        "stdout": bytearray(b"partial"),
        "stderr": bytearray(),
    }
    stdout: list[str] = []
    activity: list[int] = []

    async def read_chunk(path: str, offset: int, size: int) -> bytes:
        return bytes(contents[path][offset : offset + size])

    tailer = ShellOutputSpoolTailer(
        ShellOutputSpoolPaths(
            directory="spool",
            stdout="stdout",
            stderr="stderr",
        ),
        read_chunk=read_chunk,
        on_stdout=lambda text: _append_output(stdout, text),
        on_stderr=lambda text: _append_output([], text),
        on_stdout_activity=lambda byte_count: _append_output(activity, byte_count),
    )

    await tailer.tail_until(
        lambda: _process_exited(),
        poll_interval=0,
        final_grace_seconds=0,
    )

    assert activity == [len(contents["stdout"])]
    assert stdout == ["partial"]


async def _append_output[T](output: list[T], value: T) -> None:
    output.append(value)


async def _process_exited() -> bool:
    return True
