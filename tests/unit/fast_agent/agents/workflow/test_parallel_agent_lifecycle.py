import asyncio
from collections.abc import Awaitable, Callable

import pytest
from mcp import Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams


class CallbackAgent(LlmAgent):
    def __init__(self, name: str, callback: Callable[[], Awaitable[str]]) -> None:
        super().__init__(AgentConfig(name))
        self.callback = callback
        self.calls = 0

    async def generate_impl(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        self.calls += 1
        return PromptMessageExtended(
            role="assistant", content=[text_content(await self.callback())]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("branch failed"), asyncio.CancelledError()])
async def test_parallel_failure_cancels_and_waits_for_sibling(failure: BaseException) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_for_work() -> str:
        started.set()
        try:
            await release.wait()
            return "sibling result"
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            finished.set()

    async def fail() -> str:
        await started.wait()
        raise failure

    async def aggregate() -> str:
        return "combined"

    fan_in = CallbackAgent("fan_in", aggregate)
    parallel = ParallelAgent(
        AgentConfig("parallel"),
        fan_in,
        [CallbackAgent("failing", fail), CallbackAgent("waiting", wait_for_work)],
    )
    try:
        with pytest.raises(type(failure)) as raised:
            await asyncio.wait_for(parallel.generate("work"), timeout=2)
        assert raised.value is failure
        assert cancelled.is_set()
        assert finished.is_set()
        assert fan_in.calls == 0
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=2)


@pytest.mark.asyncio
async def test_parallel_caller_cancellation_waits_for_fan_out_cleanup() -> None:
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    finished = asyncio.Event()

    async def work() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            finished.set()

    async def aggregate() -> str:
        return "combined"

    fan_in = CallbackAgent("fan_in", aggregate)
    parallel = ParallelAgent(AgentConfig("parallel"), fan_in, [CallbackAgent("worker", work)])
    task = asyncio.create_task(parallel.generate("work"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert not task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert finished.is_set()
        assert fan_in.calls == 0
    finally:
        allow_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
