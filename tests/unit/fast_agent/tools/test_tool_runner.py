import asyncio

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, Tool

from fast_agent.tools.tool_runner import ToolRunner
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


def _assistant_tool_call(tool_calls: dict[str, CallToolRequest]) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        tool_calls=tool_calls,
        stop_reason=LlmStopReason.TOOL_USE,
    )


def _assistant_text(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.END_TURN,
        tool_calls=None,
    )


@pytest.mark.asyncio
async def test_tool_runner_no_tool_calls():
    # generate_fn returns an assistant message with no tool use
    async def generate_fn(_msgs):
        return _assistant_text("done")

    runner = ToolRunner(
        generate_fn=generate_fn,
        tools=[],
        messages=[PromptMessageExtended(role="user", content=[])],
        max_iterations=3,
        parallel_tool_calls=True,
        tool_executor=lambda name, args, tool_use_id: CallToolResult(content=[]),  # pragma: no cover
    )

    final = await runner.run_until_done()
    assert final.stop_reason == LlmStopReason.END_TURN
    assert final.tool_calls is None


@pytest.mark.asyncio
async def test_tool_runner_single_tool_call():
    # First response requests a tool, second response ends the turn
    responses = [
        _assistant_tool_call(
            {
                "t1": CallToolRequest(
                    params=CallToolRequestParams(name="do_it", arguments={"x": 1}),
                ),
            }
        ),
        _assistant_text("done"),
    ]

    async def generate_fn(_msgs):
        return responses.pop(0)

    async def exec_tool(name, args, tool_use_id):
        assert name == "do_it"
        assert args == {"x": 1}
        return CallToolResult(content=[])

    runner = ToolRunner(
        generate_fn=generate_fn,
        tools=[Tool(name="do_it", description="", inputSchema={"type": "object"})],
        messages=[PromptMessageExtended(role="user", content=[])],
        max_iterations=3,
        parallel_tool_calls=True,
        tool_executor=exec_tool,
    )

    final = await runner.run_until_done()
    assert final.stop_reason == LlmStopReason.END_TURN
    assert final.content == []


@pytest.mark.asyncio
async def test_tool_runner_parallel_tool_calls_and_error():
    tool_calls = {
        "a": CallToolRequest(params=CallToolRequestParams(name="t_a", arguments={"y": 2})),
        "b": CallToolRequest(params=CallToolRequestParams(name="t_b", arguments={"z": 3})),
    }
    responses = [
        _assistant_tool_call(tool_calls),
        _assistant_text("done"),
    ]

    async def generate_fn(_msgs):
        return responses.pop(0)

    call_order: list[str] = []

    async def exec_tool(name, args, tool_use_id):
        call_order.append(name)
        if name == "t_b":
            raise RuntimeError("boom")
        await asyncio.sleep(0)  # ensure scheduling happens
        return CallToolResult(content=[])

    runner = ToolRunner(
        generate_fn=generate_fn,
        tools=[
            Tool(name="t_a", description="", inputSchema={"type": "object"}),
            Tool(name="t_b", description="", inputSchema={"type": "object"}),
        ],
        messages=[PromptMessageExtended(role="user", content=[])],
        max_iterations=3,
        parallel_tool_calls=True,
        tool_executor=exec_tool,
    )

    final = await runner.run_until_done()

    # Both calls attempted; one failed but the loop continued
    assert set(call_order) == {"t_a", "t_b"}
    # Runner should still reach the end-turn message
    assert final.stop_reason == LlmStopReason.END_TURN
