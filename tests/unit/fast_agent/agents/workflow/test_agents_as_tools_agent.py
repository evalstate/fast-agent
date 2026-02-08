import asyncio
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams, PromptMessage, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
    HistoryMergeTarget,
    HistorySource,
)
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt_serialization import load_messages, save_messages
from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
from fast_agent.types import PromptMessageExtended, RequestParams


@pytest_asyncio.fixture(autouse=True)
async def cleanup_logging():
    yield
    from fast_agent.core.logging.logger import LoggingConfig
    from fast_agent.core.logging.transport import AsyncEventBus

    await LoggingConfig.shutdown()
    bus = AsyncEventBus._instance
    if bus is not None:
        await bus.stop()
    AsyncEventBus.reset()
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and getattr(task.get_coro(), "__qualname__", "") == "AsyncEventBus._process_events"
    ]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class FakeChildAgent(LlmAgent):
    """Minimal child agent stub for Agents-as-Tools tests."""

    def __init__(self, name: str, response_text: str = "ok", delay: float = 0):
        super().__init__(AgentConfig(name))
        self._response_text = response_text
        self._delay = delay

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        if self._delay:
            await asyncio.sleep(self._delay)
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(f"{self._response_text}")],
        )

    async def spawn_detached_instance(self, name: str | None = None):
        # Mutate name for instance labelling; reuse self to keep the stub small.
        self._name = name or self.name
        return self


class ErrorChannelChild(FakeChildAgent):
    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        return PromptMessageExtended(
            role="assistant",
            content=[],
            channels={FAST_AGENT_ERROR_CHANNEL: [text_content("err-block")]},
        )


class HistoryChild(LlmAgent):
    """Child stub that records loaded history and appends a response."""

    def __init__(self, name: str):
        super().__init__(AgentConfig(name))
        self.loaded_history: list[PromptMessageExtended] | None = None
        self.last_clone: HistoryChild | None = None

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.loaded_history = list(messages or [])
        super().load_message_history(messages)

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        response = PromptMessageExtended(
            role="assistant",
            content=[text_content("ok")],
        )
        self.message_history.append(response)
        return response

    async def spawn_detached_instance(self, name: str | None = None):
        clone = HistoryChild(name or self.name)
        clone.load_message_history(list(self.message_history))
        self.last_clone = clone
        return clone


class StubNestedAgentsAsTools(AgentsAsToolsAgent):
    """Stub AgentsAsToolsAgent that responds without hitting an LLM."""

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(f"{self.name}-reply")],
        )

    async def spawn_detached_instance(self, name: str | None = None):
        self._name = name or self.name
        return self


class RecordingToolHandler(ToolExecutionHandler):
    def __init__(self) -> None:
        self.starts: list[tuple[str, str, dict[str, Any] | None, str | None]] = []
        self.progress: list[tuple[str, float, float | None, str | None]] = []
        self.completes: list[
            tuple[str, bool, list[Any] | None, str | None]
        ] = []

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        self.starts.append((tool_name, server_name, arguments, tool_use_id))
        return "tool-call-1"

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        self.progress.append((tool_call_id, progress, total, message))

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[Any] | None,
        error: str | None,
    ) -> None:
        self.completes.append((tool_call_id, success, content, error))

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        return None

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        return None

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        return "tool-call-1"


class HookedChildAgent(LlmAgent):
    def __init__(self, name: str, response_text: str = "ok") -> None:
        super().__init__(AgentConfig(name))
        self._response_text = response_text
        self._tool_runner_hooks: ToolRunnerHooks | None = None

    @property
    def tool_runner_hooks(self) -> ToolRunnerHooks | None:
        return self._tool_runner_hooks

    @tool_runner_hooks.setter
    def tool_runner_hooks(self, value: ToolRunnerHooks | None) -> None:
        self._tool_runner_hooks = value

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        if self._tool_runner_hooks and self._tool_runner_hooks.before_llm_call:
            await self._tool_runner_hooks.before_llm_call(self, [])
        if self._tool_runner_hooks and self._tool_runner_hooks.before_tool_call:
            await self._tool_runner_hooks.before_tool_call(
                self,
                PromptMessageExtended(role="assistant", content=[]),
            )
        return PromptMessageExtended(
            role="assistant",
            content=[text_content(self._response_text)],
        )

    async def spawn_detached_instance(self, name: str | None = None):
        clone = HookedChildAgent(name or self.name, response_text=self._response_text)
        clone.tool_runner_hooks = self.tool_runner_hooks
        return clone


@pytest.mark.asyncio
async def test_list_tools_merges_base_and_child():
    child = FakeChildAgent("child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    # Inject a base MCP tool via the filtered MCP path to ensure merge behavior.
    base_tool = Tool(name="base_tool", description="base", inputSchema={"type": "object"})
    setattr(agent, "_get_filtered_mcp_tools", AsyncMock(return_value=[base_tool]))

    result = await agent.list_tools()
    tool_names = {t.name for t in result.tools}

    assert "base_tool" in tool_names
    assert "agent__child" in tool_names


@pytest.mark.asyncio
async def test_run_tools_respects_max_parallel_and_timeout():
    fast_child = FakeChildAgent("fast", response_text="fast")
    slow_child = FakeChildAgent("slow", response_text="slow", delay=0.05)

    options = AgentsAsToolsOptions(max_parallel=1, child_timeout_sec=0.01)
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [fast_child, slow_child], options=options)
    await agent.initialize()

    tool_calls = {
        "1": CallToolRequest(params=CallToolRequestParams(name="agent__fast", arguments={"text": "hi"})),
        "2": CallToolRequest(params=CallToolRequestParams(name="agent__slow", arguments={"text": "hi"})),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    result_message = await agent.run_tools(request)
    assert result_message.tool_results is not None

    fast_result = result_message.tool_results["1"]
    slow_result = result_message.tool_results["2"]

    assert not fast_result.isError
    # Skipped due to max_parallel cap.
    assert slow_result.isError
    assert slow_result.content is not None
    assert slow_result.content[0].type == "text"
    assert isinstance(slow_result.content[0], TextContent)
    assert "Skipped" in slow_result.content[0].text

    # Now ensure timeout path yields an error result when a single slow call runs.
    request_single = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={"3": CallToolRequest(params=CallToolRequestParams(name="agent__slow", arguments={"text": "hi"}))},
    )
    single_result = await agent.run_tools(request_single)
    assert single_result.tool_results is not None
    err_res = single_result.tool_results["3"]
    assert err_res.isError
    assert err_res.content is not None
    assert any(
        isinstance(block, TextContent) and "Tool execution failed" in (block.text or "")
        for block in err_res.content
    )


@pytest.mark.asyncio
async def test_run_tools_emits_progress_for_child_agent():
    child = HookedChildAgent("child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    handler = RecordingToolHandler()
    request_params = RequestParams(tool_execution_handler=handler)

    tool_calls = {
        "tool-use-1": CallToolRequest(
            params=CallToolRequestParams(name="agent__child", arguments={"message": "hi"})
        )
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    result_message = await agent.run_tools(request, request_params=request_params)
    assert result_message.tool_results is not None

    assert handler.starts == [("child", "agent", {"message": "hi"}, "tool-use-1")]
    assert handler.progress
    assert any(
        update[0] == "tool-call-1" and update[3] and "llm" in update[3]
        for update in handler.progress
    )
    assert handler.completes
    tool_call_id, success, content, error = handler.completes[0]
    assert tool_call_id == "tool-call-1"
    assert success is True
    assert error is None
    assert content is not None


@pytest.mark.asyncio
async def test_history_source_child_merges_back_to_child():
    child = HistoryChild("child")
    seed = PromptMessageExtended(role="user", content=[text_content("seed")])
    child.load_message_history([seed])

    options = AgentsAsToolsOptions(
        history_source=HistorySource.CHILD,
        history_merge_target=HistoryMergeTarget.CHILD,
    )
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child], options=options)
    await agent.initialize()

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(name="agent__child", arguments={"text": "hi"})
        ),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    await agent.run_tools(request)

    clone = child.last_clone
    assert clone is not None
    assert clone.loaded_history == [seed]
    assert len(child.message_history) == 2
    assert child.message_history[-1].role == "assistant"


@pytest.mark.asyncio
async def test_history_source_orchestrator_merges_back_to_orchestrator():
    child = HistoryChild("child")
    options = AgentsAsToolsOptions(
        history_source=HistorySource.ORCHESTRATOR,
        history_merge_target=HistoryMergeTarget.ORCHESTRATOR,
    )
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child], options=options)
    await agent.initialize()

    seed = PromptMessageExtended(role="user", content=[text_content("seed")])
    agent.load_message_history([seed])

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(name="agent__child", arguments={"text": "hi"})
        ),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    await agent.run_tools(request)

    clone = child.last_clone
    assert clone is not None
    assert clone.loaded_history == [seed]
    assert len(agent.message_history) == 2
    assert agent.message_history[-1].role == "assistant"


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="history_merge_target=messages is deferred until file merge is implemented",
)
async def test_history_merge_target_messages_updates_history_file(tmp_path):
    messages_path = tmp_path / "history.json"
    seed = PromptMessageExtended(role="user", content=[text_content("seed")])
    save_messages([seed], str(messages_path))

    child = HistoryChild("child")
    options = AgentsAsToolsOptions(history_merge_target=HistoryMergeTarget.MESSAGES)
    agent = AgentsAsToolsAgent(
        AgentConfig("parent"),
        [child],
        options=options,
        child_message_files={"child": [messages_path]},
    )
    await agent.initialize()

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(name="agent__child", arguments={"text": "hi"})
        ),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    await agent.run_tools(request)

    merged = load_messages(str(messages_path))
    assert len(merged) > 1

@pytest.mark.asyncio
async def test_invoke_child_appends_error_channel():
    child = ErrorChannelChild("err-child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    call_result = await agent._invoke_child_agent(child, {"text": "hi"})

    assert call_result.isError
    assert call_result.content is not None
    texts = [block.text for block in call_result.content if isinstance(block, TextContent)]
    assert "err-block" in texts


@pytest.mark.asyncio
async def test_nested_agents_as_tools_preserves_instance_labels():
    leaf = FakeChildAgent("leaf", response_text="leaf-ok")
    nested = StubNestedAgentsAsTools(AgentConfig("nested"), [leaf])
    parent = AgentsAsToolsAgent(AgentConfig("parent"), [nested])

    await nested.initialize()
    await parent.initialize()

    tool_calls = {
        "1": CallToolRequest(params=CallToolRequestParams(name="agent__nested", arguments={"text": "hi"})),
    }
    request = PromptMessageExtended(role="assistant", content=[], tool_calls=tool_calls)

    result_message = await parent.run_tools(request)
    assert result_message.tool_results is not None
    result = result_message.tool_results["1"]
    assert not result.isError
    # Reply should include the instance-suffixed nested agent name.
    assert result.content is not None
    assert any(
        isinstance(block, TextContent) and "nested[1]-reply" in (block.text or "")
        for block in result.content
    )
