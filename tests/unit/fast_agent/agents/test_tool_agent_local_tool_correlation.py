import asyncio
from typing import Any

import pytest
from fastmcp.tools import ToolResult
from mcp_types import CallToolRequest, CallToolRequestParams, ContentBlock, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.tools.invocation_context import (
    LocalToolInvocationContext,
    get_local_tool_invocation_context,
)
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams


class LocalToolCallSimulator(PassthroughLLM):
    def __init__(self, tool_calls: dict[str, CallToolRequest]) -> None:
        super().__init__()
        self._tool_calls = tool_calls
        self._turn = 0

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del multipart_messages, request_params, tools, is_template
        self._turn += 1
        if self._turn == 1:
            return Prompt.assistant(
                "use local tools",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls=self._tool_calls,
            )
        return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)


class ToolExecutionSimulator(NoOpToolExecutionHandler):
    def __init__(self) -> None:
        self.starts: list[tuple[str, str | None]] = []
        self.completions: list[tuple[str, bool, list[ContentBlock] | None, str | None]] = []

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
        tool_use_id: str | None = None,
    ) -> str:
        del server_name, arguments
        self.starts.append((tool_name, tool_use_id))
        return f"local-{tool_name}"

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[ContentBlock] | None,
        error: str | None,
    ) -> None:
        self.completions.append((tool_call_id, success, content, error))


def _tool_call(name: str) -> CallToolRequest:
    return CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments={}),
    )


def _context() -> LocalToolInvocationContext:
    context = get_local_tool_invocation_context()
    assert context is not None
    return context


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sequential_local_tools_receive_correlation_context_and_reset() -> None:
    observed: list[LocalToolInvocationContext] = []

    def local_tool() -> str:
        observed.append(_context())
        return "ok"

    handler = ToolExecutionSimulator()
    agent = ToolAgent(AgentConfig("local"), [local_tool])
    agent._llm = LocalToolCallSimulator({"sequential-id": _tool_call("local_tool")})

    schemas = await agent.list_tools()
    assert schemas.tools[0].input_schema["properties"] == {}

    result = await agent.generate(
        "run",
        request_params=RequestParams(tool_execution_handler=handler),
    )

    assert result.last_text() == "done"
    assert observed == [
        LocalToolInvocationContext(
            tool_name="local_tool",
            arguments={},
            tool_use_id="sequential-id",
        )
    ]
    assert handler.starts == [("local_tool", "sequential-id")]
    assert get_local_tool_invocation_context() is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_local_tools_keep_independent_correlation_contexts() -> None:
    observed: dict[str, LocalToolInvocationContext] = {}

    async def first_tool() -> str:
        context = _context()
        await asyncio.sleep(0)
        observed["first"] = _context()
        assert observed["first"] == context
        return "first"

    async def second_tool() -> str:
        context = _context()
        await asyncio.sleep(0)
        observed["second"] = _context()
        assert observed["second"] == context
        return "second"

    handler = ToolExecutionSimulator()
    agent = ToolAgent(AgentConfig("local"), [first_tool, second_tool])
    agent._llm = LocalToolCallSimulator(
        {
            "parallel-first": _tool_call("first_tool"),
            "parallel-second": _tool_call("second_tool"),
        }
    )

    await agent.generate(
        "run",
        request_params=RequestParams(tool_execution_handler=handler),
    )

    assert observed == {
        "first": LocalToolInvocationContext(
            tool_name="first_tool",
            arguments={},
            tool_use_id="parallel-first",
        ),
        "second": LocalToolInvocationContext(
            tool_name="second_tool",
            arguments={},
            tool_use_id="parallel-second",
        ),
    }
    assert set(handler.starts) == {
        ("first_tool", "parallel-first"),
        ("second_tool", "parallel-second"),
    }
    assert get_local_tool_invocation_context() is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_native_error_result_reports_failed_tool_completion() -> None:
    def error_tool() -> ToolResult:
        return ToolResult(
            content=[text_content("simulated tool failure")],
            is_error=True,
        )

    handler = ToolExecutionSimulator()
    agent = ToolAgent(AgentConfig("local"), [error_tool])

    result = await agent.call_tool(
        "error_tool",
        {},
        tool_use_id="error-call",
        request_params=RequestParams(tool_execution_handler=handler),
    )

    assert result.is_error is True
    assert handler.starts == [("error_tool", "error-call")]
    assert handler.completions == [("local-error_tool", False, None, "simulated tool failure")]
