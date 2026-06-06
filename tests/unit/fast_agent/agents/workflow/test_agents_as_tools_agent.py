import asyncio
import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from mcp import CallToolRequest, Tool
from mcp.types import CallToolRequestParams, CallToolResult, PromptMessage, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
    HistoryMergeTarget,
    HistorySource,
)
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.helpers.content_helpers import get_text, text_content
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
        bus_task = getattr(bus, "_task", None)
        await bus.stop()
        # bus.stop() is best-effort (it may swallow cancellation/timeouts). Ensure
        # the underlying processing task is fully awaited so pytest doesn't warn.
        if isinstance(bus_task, asyncio.Future) and not bus_task.done():
            bus_task.cancel()
            await asyncio.gather(bus_task, return_exceptions=True)
    AsyncEventBus.reset()
    pending = []
    for task in asyncio.all_tasks():
        if task is asyncio.current_task():
            continue
        qn = getattr(task.get_coro(), "__qualname__", "")
        if "AsyncEventBus._process_events" in qn and not task.done():
            pending.append(task)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.sleep(0)
        await asyncio.gather(*pending, return_exceptions=True)


class FakeChildAgent(LlmAgent):
    """Minimal child agent stub for Agents-as-Tools tests."""

    def __init__(self, name: str, response_text: str = "ok", delay: float = 0):
        super().__init__(AgentConfig(name))
        self._response_text = response_text
        self._delay = delay
        self.last_request_params: RequestParams | None = None

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        self.last_request_params = request_params
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


def test_agents_as_tools_rejects_duplicate_child_tool_names() -> None:
    first = FakeChildAgent("child")
    second = FakeChildAgent("child")

    with pytest.raises(
        AgentConfigError,
        match="Duplicate Agents-as-Tools tool name 'agent__child'",
    ):
        AgentsAsToolsAgent(AgentConfig("parent"), [first, second])


class StructuredInputChild(FakeChildAgent):
    def __init__(self, name: str, response_text: str = "ok") -> None:
        super().__init__(name, response_text=response_text)
        self.last_input_text: str | None = None

    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        if isinstance(messages, Sequence) and messages:
            first_message = messages[0]
            if isinstance(first_message, PromptMessageExtended) and first_message.content:
                first_block = first_message.content[0]
                if isinstance(first_block, TextContent):
                    self.last_input_text = first_block.text
        return await super().generate(messages, request_params=request_params, tools=tools)


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


class CancellingChild(FakeChildAgent):
    async def generate(
        self,
        messages: str
        | PromptMessage
        | PromptMessageExtended
        | Sequence[str | PromptMessage | PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        raise asyncio.CancelledError


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

    def _hook_runner(self) -> ToolRunner:
        """Narrow this lightweight test stub at the hook boundary."""
        return self  # ty: ignore[invalid-return-type]

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
            await self._tool_runner_hooks.before_llm_call(self._hook_runner(), [])
        if self._tool_runner_hooks and self._tool_runner_hooks.before_tool_call:
            await self._tool_runner_hooks.before_tool_call(
                self._hook_runner(),
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


def test_parallel_streaming_close_does_not_swallow_internal_attribute_errors() -> None:
    class BrokenStreamingCloseAgent(AgentsAsToolsAgent):
        def close_active_streaming_display(self, *, reason: str | None = None) -> bool:
            raise AttributeError("internal display state missing")

    child = FakeChildAgent("child")
    agent = BrokenStreamingCloseAgent(AgentConfig("parent"), [child])

    with pytest.raises(AttributeError, match="internal display state missing"):
        agent._close_streaming_for_parallel_child_tools(2)


@pytest.mark.asyncio
async def test_call_tool_routes_collision_to_advertised_base_tool(monkeypatch):
    child = StructuredInputChild("child", response_text="child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    base_tool = Tool(name="agent__child", description="base", inputSchema={"type": "object"})
    setattr(agent, "_get_filtered_mcp_tools", AsyncMock(return_value=[base_tool]))

    async def fake_base_call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        del self, arguments, tool_use_id, request_params
        return CallToolResult(content=[text_content(f"base:{name}")], isError=False)

    monkeypatch.setattr(McpAgent, "call_tool", fake_base_call_tool)

    result = await agent.call_tool("agent__child", {"message": "hi"})

    assert result.isError is False
    assert result.content is not None
    assert get_text(result.content[0]) == "base:agent__child"
    assert child.last_input_text is None


@pytest.mark.asyncio
async def test_run_tools_routes_collision_to_advertised_base_tool(monkeypatch):
    child = StructuredInputChild("child", response_text="child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    base_tool = Tool(name="agent__child", description="base", inputSchema={"type": "object"})
    setattr(agent, "_get_filtered_mcp_tools", AsyncMock(return_value=[base_tool]))

    async def fake_base_run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        del self, request_params
        return PromptMessageExtended(
            role="user",
            tool_results={
                cid: CallToolResult(
                    content=[text_content(f"base:{tool.params.name}")],
                    isError=False,
                )
                for cid, tool in (request.tool_calls or {}).items()
            },
        )

    monkeypatch.setattr(McpAgent, "run_tools", fake_base_run_tools)

    request = PromptMessageExtended(
        role="assistant",
        tool_calls={
            "1": CallToolRequest(
                params=CallToolRequestParams(name="agent__child", arguments={"message": "hi"})
            )
        },
    )

    result = await agent.run_tools(request)

    assert result.tool_results is not None
    assert get_text(result.tool_results["1"].content[0]) == "base:agent__child"
    assert child.last_input_text is None


@pytest.mark.asyncio
async def test_list_tools_uses_child_tool_input_schema():
    child = FakeChildAgent("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
        },
        "required": ["query"],
    }
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent.list_tools()
    child_tool = next(tool for tool in result.tools if tool.name == "agent__child")

    assert child_tool.inputSchema == child.config.tool_input_schema


@pytest.mark.asyncio
async def test_list_tools_adds_response_mode_when_child_tool_result_mode_is_selectable():
    child = FakeChildAgent("child")
    child.config.default_request_params = RequestParams(tool_result_mode="selectable")

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent.list_tools()
    child_tool = next(tool for tool in result.tools if tool.name == "agent__child")
    properties = child_tool.inputSchema.get("properties", {})

    assert child_tool.inputSchema.get("required") == ["message"]
    assert properties.get("response_mode") == {
        "type": "string",
        "description": "Override how the child agent returns tool results for this call.",
        "enum": ["inherit", "postprocess", "passthrough"],
        "default": "inherit",
    }


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
    # max_parallel limits concurrency without dropping requested calls; the slow
    # call still runs and then hits the per-child timeout.
    assert slow_result.isError
    assert slow_result.content is not None
    assert slow_result.content[0].type == "text"
    assert isinstance(slow_result.content[0], TextContent)

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
async def test_run_tools_preserves_interleaved_child_and_mcp_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = FakeChildAgent("child", response_text="child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    async def fake_base_run_tools(
        self: McpAgent,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        del self, request_params
        assert request.tool_calls is not None
        return PromptMessageExtended(
            role="user",
            tool_results={
                correlation_id: CallToolResult(
                    content=[text_content(f"mcp:{correlation_id}")],
                    isError=False,
                )
                for correlation_id in request.tool_calls
            },
        )

    monkeypatch.setattr(McpAgent, "run_tools", fake_base_run_tools)

    request = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "mcp-1": CallToolRequest(
                params=CallToolRequestParams(name="base_tool", arguments={})
            ),
            "child-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="agent__child",
                    arguments={"message": "hi"},
                )
            ),
            "mcp-2": CallToolRequest(
                params=CallToolRequestParams(name="other_tool", arguments={})
            ),
        },
    )

    result_message = await agent.run_tools(request)

    assert result_message.tool_results is not None
    assert list(result_message.tool_results) == ["mcp-1", "child-1", "mcp-2"]


@pytest.mark.asyncio
async def test_invoke_child_uses_structured_json_input_for_custom_schema():
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent._invoke_child_agent(
        child,
        {"query": "find updates", "sources": ["docs.fast-agent.ai"]},
    )

    assert result.isError is False
    assert child.last_input_text is not None
    assert json.loads(child.last_input_text) == {
        "query": "find updates",
        "sources": ["docs.fast-agent.ai"],
    }


@pytest.mark.asyncio
async def test_invoke_child_uses_structured_json_input_for_mixed_message_schema():
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "User message context",
            },
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
            "filters": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent._invoke_child_agent(
        child,
        {"message": "context", "query": "find updates", "filters": ["docs", "code"]},
    )

    assert result.isError is False
    assert child.last_input_text is not None
    assert json.loads(child.last_input_text) == {
        "message": "context",
        "query": "find updates",
        "filters": ["docs", "code"],
    }


@pytest.mark.asyncio
async def test_invoke_child_uses_legacy_message_input_for_message_only_schema():
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to send to the agent",
            },
        },
        "required": ["message"],
    }

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent._invoke_child_agent(child, {"message": "hello child"})

    assert result.isError is False
    assert child.last_input_text == "hello child"


@pytest.mark.asyncio
async def test_child_delegation_keeps_workflow_settings_without_parent_llm_defaults() -> None:
    child = FakeChildAgent("child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    parent_request_params = RequestParams(
        model="parent-model",
        maxTokens=123,
        tool_result_mode="passthrough",
    )
    await agent._invoke_child_agent(
        child,
        {"message": "hello child"},
        request_params=parent_request_params,
    )

    assert child.last_request_params is not None
    assert child.last_request_params.model is None
    assert "maxTokens" not in child.last_request_params.model_dump(exclude_unset=True)
    assert child.last_request_params.tool_result_mode == "passthrough"


@pytest.mark.asyncio
async def test_child_response_mode_overrides_inherited_passthrough_and_is_stripped() -> None:
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
        },
        "required": ["query"],
    }
    child.config.default_request_params = RequestParams(tool_result_mode="selectable")

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    parent_request_params = RequestParams(
        model="parent-model",
        maxTokens=123,
        tool_result_mode="passthrough",
    )
    await agent._invoke_child_agent(
        child,
        {
            "query": "find updates",
            "response_mode": " PostProcess ",
        },
        request_params=parent_request_params,
    )

    assert child.last_request_params is not None
    assert child.last_request_params.model is None
    assert "maxTokens" not in child.last_request_params.model_dump(exclude_unset=True)
    assert child.last_request_params.tool_result_mode == "postprocess"
    assert child.last_input_text is not None
    assert json.loads(child.last_input_text) == {"query": "find updates"}


@pytest.mark.asyncio
async def test_child_response_mode_rejects_invalid_value_when_control_enabled() -> None:
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }
    child.config.default_request_params = RequestParams(tool_result_mode="selectable")

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    result = await agent._invoke_child_agent(
        child,
        {
            "query": "find updates",
            "response_mode": "passthru",
        },
    )

    assert result.isError is True
    assert result.content is not None
    error_text = get_text(result.content[0])
    assert error_text is not None
    assert "Invalid response_mode 'passthru'" in error_text
    assert child.last_input_text is None


@pytest.mark.asyncio
async def test_child_response_mode_field_is_preserved_when_control_disabled() -> None:
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
        },
        "required": ["query"],
    }

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    parent_request_params = RequestParams(
        model="parent-model",
        tool_result_mode="passthrough",
    )
    await agent._invoke_child_agent(
        child,
        {
            "query": "find updates",
            "response_mode": "surprise",
        },
        request_params=parent_request_params,
    )

    assert child.last_request_params is not parent_request_params
    assert child.last_request_params is not None
    assert child.last_request_params.systemPrompt is None
    assert child.last_request_params.model is None
    assert child.last_request_params.tool_result_mode == "passthrough"
    assert child.last_input_text is not None
    assert json.loads(child.last_input_text) == {
        "query": "find updates",
        "response_mode": "surprise",
    }


@pytest.mark.asyncio
async def test_child_owned_response_mode_field_is_preserved_when_selectable() -> None:
    child = StructuredInputChild("child")
    child.config.tool_input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
            "response_mode": {
                "type": "string",
                "description": "A domain-specific mode consumed by the child agent",
            },
        },
        "required": ["query"],
    }
    child.config.default_request_params = RequestParams(tool_result_mode="selectable")

    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    listed_tools = await agent.list_tools()
    child_tool = next(tool for tool in listed_tools.tools if tool.name == "agent__child")

    assert child_tool.inputSchema == child.config.tool_input_schema

    await agent._invoke_child_agent(
        child,
        {
            "query": "find updates",
            "response_mode": "domain-value",
        },
    )

    assert child.last_request_params is None
    assert child.last_input_text is not None
    assert json.loads(child.last_input_text) == {
        "query": "find updates",
        "response_mode": "domain-value",
    }


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

    assert handler.starts == [("child[1]", "agent", {"message": "hi"}, "tool-use-1")]
    assert handler.progress
    # Progress updates are intentionally minimal; title already includes the agent instance.
    assert any(update[0] == "tool-call-1" for update in handler.progress)
    assert handler.completes
    tool_call_id, success, content, error = handler.completes[0]
    assert tool_call_id == "tool-call-1"
    assert success is True
    assert error is None
    assert content is not None


@pytest.mark.asyncio
async def test_invoke_child_completes_tool_call_on_cancellation() -> None:
    child = CancellingChild("child")
    agent = AgentsAsToolsAgent(AgentConfig("parent"), [child])
    await agent.initialize()

    handler = RecordingToolHandler()
    request_params = RequestParams(tool_execution_handler=handler)

    with pytest.raises(asyncio.CancelledError):
        await agent._invoke_child_agent(
            child,
            {"message": "hi"},
            request_params=request_params,
            tool_use_id="tool-use-1",
        )

    assert handler.starts == [("child", "agent", {"message": "hi"}, "tool-use-1")]
    assert handler.completes == [
        ("tool-call-1", False, None, "Child agent tool call cancelled")
    ]


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
async def test_history_source_none_and_merge_none_leave_histories_unchanged():
    child = HistoryChild("child")
    seed = PromptMessageExtended(role="user", content=[text_content("seed")])
    child.load_message_history([seed])

    options = AgentsAsToolsOptions(
        history_source=HistorySource.NONE,
        history_merge_target=HistoryMergeTarget.NONE,
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
    assert clone.loaded_history == []
    assert child.message_history == [seed]
    assert agent.message_history == []


def test_history_options_reject_invalid_history_source() -> None:
    invalid_source: Any = "chlid"

    with pytest.raises(ValueError, match="history_source must be one of"):
        AgentsAsToolsOptions(history_source=invalid_source)


def test_history_options_reject_unsupported_message_merge_target() -> None:
    message_target: Any = "messages"

    with pytest.raises(ValueError, match="history_merge_target=messages is not supported"):
        AgentsAsToolsOptions(history_merge_target=message_target)


def test_child_request_params_strip_parent_system_prompt() -> None:
    request_params = RequestParams(
        systemPrompt="parent instruction",
        model="passthrough",
        tool_result_mode="selectable",
    )

    child_params = AgentsAsToolsAgent._build_child_request_params(
        request_params,
        tool_result_mode_override=None,
    )

    assert child_params is not None
    assert child_params.systemPrompt is None
    assert child_params.model is None
    assert child_params.tool_result_mode == "selectable"


def test_child_request_params_strip_parent_system_prompt_with_response_mode_override() -> None:
    request_params = RequestParams(
        systemPrompt="parent instruction",
        model="passthrough",
        tool_result_mode="selectable",
    )

    child_params = AgentsAsToolsAgent._build_child_request_params(
        request_params,
        tool_result_mode_override="passthrough",
    )

    assert child_params is not None
    assert child_params.systemPrompt is None
    assert child_params.model is None
    assert child_params.tool_result_mode == "passthrough"


def test_child_request_params_preserve_explicit_no_history() -> None:
    request_params = RequestParams(
        systemPrompt="parent instruction",
        model="passthrough",
        use_history=False,
    )

    child_params = AgentsAsToolsAgent._build_child_request_params(
        request_params,
        tool_result_mode_override=None,
    )

    assert child_params is not None
    assert child_params.systemPrompt is None
    assert child_params.model is None
    assert child_params.use_history is False


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
