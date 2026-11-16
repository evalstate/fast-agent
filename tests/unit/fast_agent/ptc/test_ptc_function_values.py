"""Typed values from function tools: PTC code gets the return value, direct calls get text.

A stub's return type must hold for every value the runtime can hand sandbox code; when the
value can differ (strings parsed as JSON, serialised models, structured results with media),
the stub says ``Any`` or includes the alternative.
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List

import pytest
import pytest_asyncio
from mcp_types import CallToolResult, ImageContent, TextContent, Tool
from pydantic import PlainSerializer, RootModel

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
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


class Level(IntEnum):
    HIGH = 3


def level() -> int:
    return Level.HIGH


def ready() -> bool:
    return False


def maybe() -> int | None:
    return None


def pair() -> tuple[int, int]:
    return (1, 2)


def rootint() -> RootModel[int]:
    return RootModel[int](3)


def bare_list() -> List:
    return [1, "a"]


def bare_dict() -> Dict:
    return {"a": 1}


def nested_bare() -> list[List]:
    return [[1]]


def serialized() -> Annotated[int, PlainSerializer(str, return_type=str)]:
    return 3


UNRECOGNISED = [bare_list, bare_dict, nested_bare, serialized]


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _value(result: CallToolResult) -> Any:
    assert not result.is_error, _text(result)
    return json.loads(_text(result).split("Result:\n", 1)[1])


def _agent(*, ptc: bool) -> McpAgent:
    return McpAgent(
        AgentConfig(name="typed", servers=[]),
        context=Context(config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=ptc))),
        tools=[add, code, payload, level, ready, maybe, pair, rootint, *UNRECOGNISED],
    )


@pytest_asyncio.fixture
async def agent() -> AsyncIterator[McpAgent]:
    agent = _agent(ptc=True)
    await agent.list_tools()
    yield agent
    await agent.shutdown()


async def _run(agent: McpAgent, code: str) -> Any:
    return _value(await agent.call_tool(PTC_TOOL_NAME, {"code": code}, "call-1"))


@pytest.mark.asyncio
async def test_code_gets_json_return_values(agent: McpAgent) -> None:
    value = await _run(
        agent,
        "n = await add(a=100, b=37)\n"
        "[n + 1, await code(), (await payload())['a'], await level() + 1, await ready(),\n"
        " await maybe(), await pair(), (await rootint()).isdigit()]",
    )

    # An IntEnum returned for `-> int` arrives as a plain int. Strings keep the text rules:
    # "137" stays a str, JSON text is parsed. A RootModel and a tuple are not JSON values,
    # so they keep the text path too.
    assert value == [138, "137", 1, 4, False, None, [1, 2], True]


@pytest.mark.asyncio
async def test_stubs_narrow_only_values_passed_through(agent: McpAgent) -> None:
    surface = await agent.tool_surface()
    signatures = {
        name: tool.signature
        for name in ("add", "code", "payload", "ready", "maybe", "pair", "rootint")
        if (tool := surface.find(name)) is not None
    }

    assert signatures == {
        "add": "add(*, a: int, b: int) -> int",
        "code": "code() -> Any",
        "payload": "payload() -> Any",
        "ready": "ready() -> bool",
        "maybe": "maybe() -> int | None",
        "pair": "pair() -> Any",
        "rootint": "rootint() -> Any",
    }


@pytest.mark.asyncio
async def test_agent_tools_keep_json_text_parsing() -> None:
    agent = _agent(ptc=True)
    child = ToolAgent(AgentConfig(name="echo", servers=[]), tools=[], context=Context())
    try:
        await child.attach_llm(ModelFactory.create_factory("passthrough"))
        agent.add_agent_tool(child, name="echo")
        await agent.list_tools()
        assert await _run(agent, "(await echo(message='{\"a\":1}'))['a']") == 1
    finally:
        await agent.shutdown()
        await child.shutdown()


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
        Tool(name="chart", input_schema={"type": "object"}, output_schema=wrapped),
        Tool(name="odd", input_schema={"type": "object"}, output_schema=wrapped),
    ]
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    description = runtime.model_tools(tools)[0].description or ""
    image = ImageContent(type="image", mime_type="image/png", data="aGk=")

    async def dispatch(name: str, arguments: dict[str, Any], call_id: str) -> CallToolResult:
        text = [TextContent(type="text", text="[2,3,5]")]
        match name:
            case "label":
                return CallToolResult(content=[TextContent(type="text", text="137")])
            case "primes":
                return CallToolResult(content=text, structured_content={"result": [2, 3, 5]})
            case "chart":
                return CallToolResult(content=[*text, image], structured_content={"result": [2]})
        return CallToolResult(content=text, structured_content={"other": 1})

    try:
        result = await runtime.execute(
            {
                "code": (
                    "out = [await label()]\n"
                    "for value in [await primes(), await chart(), await odd()]:\n"
                    "    if isinstance(value, list):\n"
                    "        out.append(sum(value))\n"
                    "    else:\n"
                    "        out.append(sorted(value))\n"
                    "out"
                )
            },
            dispatch,
        )
    finally:
        await runtime.close()

    assert "async def primes() -> list[int] | dict[str, Any]:" in description
    # Media is kept in the structured_content/content envelope the stub allows for.
    assert _value(result) == ["137", 10, ["content", "structured_content"], ["other"]]


@pytest.mark.asyncio
async def test_unrecognised_annotations_register_and_keep_the_text_path(agent: McpAgent) -> None:
    # Unparameterised typing aliases and Annotated serializers are typed Any, and their values
    # follow the text rules exactly as before typed values: the serializer's "3" stays "3".
    surface = await agent.tool_surface()
    signatures = [
        tool.signature
        for function in UNRECOGNISED
        if (tool := surface.find(function.__name__)) is not None
    ]
    value = await _run(
        agent,
        "[await bare_list(), await bare_dict(), await nested_bare(), (await serialized()).upper()]",
    )
    direct = [await agent.call_tool(function.__name__, {}, "call-2") for function in UNRECOGNISED]

    assert signatures == [f"{function.__name__}() -> Any" for function in UNRECOGNISED]
    assert value == [[1, "a"], {"a": 1}, [[1]], "3"]
    assert [_text(result) for result in direct] == ['[1,"a"]', '{"a":1}', "[[1]]", "3"]
