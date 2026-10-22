"""Contract tests for /tools when programmatic tool calling hides tools inside the code tool."""

from __future__ import annotations

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.core.harness_tools import SLASH_COMMAND_TOOL_NAME, set_harness_tools
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.tools.tool_surface import CHARS_PER_TOKEN, LARGE_STUB_TOKENS, ToolExposure

pytest.importorskip("pydantic_monty")


async def lookup_order(order_id: str) -> str:
    return order_id


async def list_items(source: str = "all") -> str:
    return source


async def big_report() -> str:
    return "report"


def _agent(*, ptc: bool) -> McpAgent:
    big_description = "Large report. " * (LARGE_STUB_TOKENS * CHARS_PER_TOKEN // 10)
    agent = McpAgent(
        AgentConfig(name="ptc-tools", servers=[], harness_tools=True),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=ptc, direct_tools=["lookup_*"])
            )
        ),
        tools=[
            build_default_function_tool(lookup_order),
            build_default_function_tool(list_items, name="list"),
            build_default_function_tool(big_report, description=big_description),
        ],
    )
    set_harness_tools(agent, True)
    return agent


async def _tools(agent: McpAgent, arguments: str = "", *, ok: bool = True) -> str:
    result = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": f"/tools {arguments}"})
    text = get_text(result.content[0])
    assert result.is_error is not ok, text
    assert text is not None
    # The harness renders for a markdown client; compare against the unescaped text.
    return text.replace("\\", "")


@pytest.mark.asyncio
async def test_every_tool_is_reachable_exactly_once_in_the_surface() -> None:
    direct_names = sorted(tool.name for tool in (await _agent(ptc=False).list_tools()).tools)

    surface = await _agent(ptc=True).tool_surface()
    surface_names = sorted(
        tool.tool.name for tool in surface.tools if tool.tool.name != PTC_TOOL_NAME
    )

    assert surface_names == direct_names
    model_names = {tool.tool.name for tool in surface.exposed(ToolExposure.MODEL)}
    assert model_names == {PTC_TOOL_NAME, SLASH_COMMAND_TOOL_NAME, "lookup_order"}


@pytest.mark.asyncio
async def test_tools_summary_separates_model_tools_from_code_functions() -> None:
    rendered = await _tools(_agent(ptc=True))

    assert "Programmatic tool calling: on" in rendered
    assert "Sent to the model (3" in rendered
    assert "declared by tool" in rendered
    assert 'ptc.direct_tools "lookup_*"' in rendered
    assert f"Callable from {PTC_TOOL_NAME} (3" in rendered
    assert "list_tool" in rendered
    assert "/tools --all" in rendered
    assert "Large stubs" in rendered and "big_report" in rendered


@pytest.mark.asyncio
async def test_tools_all_lists_code_function_signatures() -> None:
    rendered = await _tools(_agent(ptc=True), "--all")

    assert "await list_tool(*, source: str = ...) -> Any" in rendered
    assert "tool: list" in rendered
    assert "/tools --all" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["list", "list_tool"])
async def test_tools_detail_resolves_code_functions_by_tool_or_python_name(name: str) -> None:
    rendered = await _tools(_agent(ptc=True), name)

    assert "# Tool schema: list" in rendered
    assert f"Callable from `{PTC_TOOL_NAME}` as `await list_tool(...)`" in rendered
    assert "async def list_tool(*, source: str = ...) -> Any:" in rendered


@pytest.mark.asyncio
async def test_tools_detail_for_code_tool_lists_functions_without_stub_dump() -> None:
    rendered = await _tools(_agent(ptc=True), PTC_TOOL_NAME)

    assert "## Instructions" in rendered
    assert "## Callable functions (3" in rendered
    assert "Available functions:" not in rendered


@pytest.mark.asyncio
async def test_tools_without_ptc_lists_only_model_tools() -> None:
    rendered = await _tools(_agent(ptc=False))

    assert "Sent to the model (5" in rendered
    assert "Callable from" not in rendered
    assert "Programmatic tool calling" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["one two", "--bogus"])
async def test_tools_rejects_invalid_arguments(arguments: str) -> None:
    rendered = await _tools(_agent(ptc=True), arguments, ok=False)

    assert "Usage: /tools [--all|<tool-name>]" in rendered
