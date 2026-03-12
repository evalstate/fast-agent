import asyncio
import time

import pytest

from fast_agent.tools.function_tool_loader import FastMCPTool


@pytest.mark.asyncio
async def test_sync_function_tool_runs_off_event_loop() -> None:
    def blocking_add(a: int, b: int) -> int:
        time.sleep(0.05)
        return a + b

    tool = FastMCPTool.from_function(blocking_add)

    async def probe() -> float:
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        return time.perf_counter() - started

    result, probe_elapsed = await asyncio.gather(
        tool.run({"a": 2, "b": 3}),
        probe(),
    )

    assert result == 5
    assert probe_elapsed < 0.03


@pytest.mark.asyncio
async def test_sync_function_tool_injects_context_kwarg() -> None:
    sentinel = object()

    def read_context(value: str, ctx: object) -> str:
        assert ctx is sentinel
        return value.upper()

    tool = FastMCPTool.from_function(read_context, context_kwarg="ctx")

    result = await tool.run({"value": "hello"}, context=sentinel)

    assert result == "HELLO"


@pytest.mark.asyncio
async def test_async_function_tool_still_runs_inline() -> None:
    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    tool = FastMCPTool.from_function(async_add)

    result = await tool.run({"a": 4, "b": 5})

    assert result == 9
