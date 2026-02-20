import asyncio
import io

import pytest
from rich.console import Console

from fast_agent.core.logging.events import EventFilter
from fast_agent.core.logging.listeners import FilteredListener, convert_log_event
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.event_progress import ProgressAction
from fast_agent.ui.rich_progress import RichProgressDisplay


async def _wait_for(
    predicate,
    *,
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Wait for an async-test condition to become true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while True:
        if predicate():
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(interval)


class ProgressDisplayProbe(FilteredListener):
    def __init__(self) -> None:
        event_filter = EventFilter(types={"info"}, namespaces={"fast_agent.mcp.mcp_aggregator"})
        super().__init__(event_filter=event_filter)
        self.display = RichProgressDisplay(console=Console(file=io.StringIO(), force_terminal=True))
        self.display.start()
        self.seen_correlation_ids: set[str] = set()
        self.max_task_count = 0

    async def handle_matched_event(self, event) -> None:
        progress_event = convert_log_event(event)
        if not progress_event:
            return
        if progress_event.action not in {ProgressAction.CALLING_TOOL, ProgressAction.TOOL_PROGRESS}:
            return

        self.display.update(progress_event)
        self.max_task_count = max(self.max_task_count, len(self.display._taskmap))
        if progress_event.correlation_id:
            self.seen_correlation_ids.add(progress_event.correlation_id)


class LLMToolCallProbe(FilteredListener):
    def __init__(self) -> None:
        event_filter = EventFilter(types={"info"}, namespaces={"fast_agent.llm.provider.openai"})
        super().__init__(event_filter=event_filter)
        self.display = RichProgressDisplay(console=Console(file=io.StringIO(), force_terminal=True))
        self.display.start()
        self.seen_correlation_ids: set[str] = set()
        self.max_task_count = 0

    async def handle_matched_event(self, event) -> None:
        progress_event = convert_log_event(event)
        if not progress_event:
            return
        if progress_event.action != ProgressAction.CALLING_TOOL:
            return

        self.display.update(progress_event)
        self.max_task_count = max(self.max_task_count, len(self.display._taskmap))
        if progress_event.correlation_id:
            self.seen_correlation_ids.add(progress_event.correlation_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_tool_progress_rows_are_tracked_and_cleaned_up(fast_agent) -> None:
    fast = fast_agent

    probe = ProgressDisplayProbe()
    bus = AsyncEventBus.get()
    listener_name = f"parallel_progress_probe_{id(probe)}"
    bus.add_listener(listener_name, probe)

    try:

        @fast.agent(
            name="test",
            instruction="Test parallel progress display behavior",
            servers=["progress_test"],
        )
        async def agent_function() -> None:
            async with fast.run() as app:

                async def run_one(index: int) -> None:
                    result = await app.test.call_tool(
                        "progress_task",
                        {"steps": 5},
                        tool_use_id=f"parallel-tool-{index}",
                    )
                    assert not result.isError

                await asyncio.gather(*(run_one(i) for i in range(3)))

                # Give async event listeners time to process final completion updates.
                await asyncio.sleep(0.2)

        await agent_function()

        assert await _wait_for(lambda: len(probe.seen_correlation_ids) >= 3)
        assert probe.max_task_count >= 3
        assert await _wait_for(lambda: len(probe.display._taskmap) == 0)

    finally:
        bus.remove_listener(listener_name)
        probe.display.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_tool_call_rows_cleanup_on_stop_events(fast_agent) -> None:
    fast = fast_agent
    probe = LLMToolCallProbe()
    bus = AsyncEventBus.get()
    listener_name = f"llm_tool_call_probe_{id(probe)}"
    bus.add_listener(listener_name, probe)
    llm_logger = get_logger("fast_agent.llm.provider.openai.codex_responses.dev")

    try:

        @fast.agent(name="test", instruction="Test websocket tool row cleanup", servers=[])
        async def agent_function() -> None:
            async with fast.run():
                tool_use_ids = [f"call_id_{idx}" for idx in range(3)]

                for tool_use_id in tool_use_ids:
                    llm_logger.info(
                        "Model started streaming tool call",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": "test",
                            "model": "gpt-5.3-codex",
                            "tool_name": "execute",
                            "tool_use_id": tool_use_id,
                            "tool_event": "start",
                        },
                    )

                for tool_use_id in tool_use_ids:
                    llm_logger.info(
                        "Model finished streaming tool call",
                        data={
                            "progress_action": ProgressAction.CALLING_TOOL,
                            "agent_name": "test",
                            "model": "gpt-5.3-codex",
                            "tool_name": "execute",
                            "tool_use_id": tool_use_id,
                            "tool_event": "stop",
                        },
                    )

                await asyncio.sleep(0.2)

        await agent_function()

        assert await _wait_for(lambda: len(probe.seen_correlation_ids) >= 3)
        assert probe.max_task_count >= 3
        assert await _wait_for(lambda: len(probe.display._taskmap) == 0)

    finally:
        bus.remove_listener(listener_name)
        probe.display.stop()
