"""Typed values from function tools: PTC code gets the return value, direct calls get text."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from fastmcp.tools import FunctionTool
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.ptc.runtime import ProgrammaticToolRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytest.importorskip("pydantic_monty")


def add(a: int, b: int) -> int:
    return a + b


async def code() -> str:
    return "137"


def payload() -> str:
    return '{"a": 1}'


def ready() -> bool:
    return False


def pair() -> tuple[int, int]:
    return (1, 2)


def multiply(a: int, b: int) -> int:
    return a * b


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _value(result: CallToolResult) -> Any:
    assert not result.is_error, _text(result)
    return json.loads(_text(result).split("Result:\n", 1)[1])


def _agent(*, ptc: bool) -> McpAgent:
    return McpAgent(
        AgentConfig(name="typed", servers=[]),
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=ptc))),
        # A FastMCP tool built with its own output schema wraps the int as {"result": ...}.
        tools=[add, code, payload, ready, pair, FunctionTool.from_function(multiply)],
    )


@pytest_asyncio.fixture
async def agent() -> AsyncIterator[McpAgent]:
    agent = _agent(ptc=True)
    await agent.list_tools()
    yield agent
    await agent.shutdown()


@pytest.mark.asyncio
async def test_code_gets_typed_return_values(agent: McpAgent) -> None:
    result = await agent.call_tool(
        PTC_TOOL_NAME,
        {
            "code": (
                "n = await add(a=100, b=37)\n"
                "[n + 1, await code(), await payload(), await ready(), await pair(),\n"
                " await multiply(a=6, b=7) + 1]"
            )
        },
        "call-1",
    )

    assert _value(result) == [138, "137", '{"a": 1}', False, [1, 2], 43]


@pytest.mark.asyncio
async def test_stubs_declare_the_return_type(agent: McpAgent) -> None:
    surface = await agent.tool_surface()
    stubs = {
        name: tool.stub or ""
        for name in ("add", "code", "ready", "multiply")
        if (tool := surface.find(name)) is not None
    }

    assert "async def add(*, a: int, b: int) -> int:" in stubs["add"]
    assert "async def code() -> str:" in stubs["code"]
    assert "async def ready() -> bool:" in stubs["ready"]
    assert "async def multiply(*, a: int, b: int) -> int:" in stubs["multiply"]
    assert "Returns:" not in stubs["multiply"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ptc", [False, True])
async def test_direct_calls_return_unchanged_text(ptc: bool) -> None:
    agent = _agent(ptc=ptc)
    try:
        results = [
            await agent.call_tool("add", {"a": 100, "b": 37}, "call-1"),
            await agent.call_tool("ready", {}, "call-2"),
        ]
    finally:
        await agent.shutdown()

    assert [result.model_dump(by_alias=True) for result in results] == [
        CallToolResult(content=[TextContent(type="text", text=text)]).model_dump(by_alias=True)
        for text in ("137", "false")
    ]


@pytest.mark.asyncio
async def test_mcp_text_stays_text_and_wrapped_results_unwrap() -> None:
    wrapped = {
        "type": "object",
        "properties": {"result": {"type": "array", "items": {"type": "integer"}}},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
    }
    tools = [
        Tool(name="label", input_schema={"type": "object"}),
        Tool(name="primes", input_schema={"type": "object"}, output_schema=wrapped),
    ]
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    description = runtime.model_tools(tools)[0].description or ""

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        if name == "label":
            return CallToolResult(content=[TextContent(type="text", text="137")])
        return CallToolResult(
            content=[TextContent(type="text", text="[2,3,5]")],
            structured_content={"result": [2, 3, 5]},
        )

    try:
        result = await runtime.execute({"code": "[await label(), sum(await primes())]"}, dispatch)
    finally:
        await runtime.close()

    assert "async def primes() -> list[int]:" in description
    assert _value(result) == ["137", 10]
