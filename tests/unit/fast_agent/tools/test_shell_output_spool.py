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


async def _append_output(output: list[str], text: str) -> None:
    output.append(text)


async def _process_exited() -> bool:
    return True
