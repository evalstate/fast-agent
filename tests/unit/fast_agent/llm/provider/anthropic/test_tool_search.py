"""Tests for Anthropic tool search (deferred tool definitions)."""

import pytest
from mcp import Tool

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    ANTHROPIC_TOOL_SEARCH_TYPE,
    TOOL_SEARCH_AUTO_TOOL_THRESHOLD,
    AnthropicLLM,
)
from fast_agent.llm.provider.anthropic.llm_anthropic_vertex import AnthropicVertexLLM
from fast_agent.llm.request_params import RequestParams


def _make_llm() -> AnthropicLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(api_key="test-key")
    context = Context(config=settings)
    return AnthropicLLM(context=context, model="claude-sonnet-4-6", name="test-agent")


def _make_vertex_llm() -> AnthropicVertexLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(api_key="test-key")
    context = Context(config=settings)
    return AnthropicVertexLLM(context=context, model="claude-sonnet-4-6", name="test-agent")


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Test tool {name}.",
        input_schema={"type": "object", "properties": {}},
    )


@pytest.mark.asyncio
async def test_default_policy_sends_full_definitions() -> None:
    llm = _make_llm()
    tools = [_tool("alpha"), _tool("beta")]

    prepared = await llm._prepare_tools("claude-sonnet-4-6", tools=tools)

    assert len(prepared) == 2
    assert all("defer_loading" not in tool for tool in prepared)
    assert [tool["name"] for tool in prepared] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_tool_search_defers_definitions_and_appends_search_tool() -> None:
    llm = _make_llm()
    tools = [_tool("alpha"), _tool("beta")]

    prepared = await llm._prepare_tools("claude-sonnet-4-6", tools=tools, tool_search_active=True)

    assert len(prepared) == 3
    assert all(tool.get("defer_loading") is True for tool in prepared[:2])
    assert prepared[0]["name"] == "alpha"
    assert prepared[0]["description"] == "Test tool alpha."
    assert prepared[0]["input_schema"] == {"type": "object", "properties": {}}
    search_tool = prepared[-1]
    assert search_tool["type"] == ANTHROPIC_TOOL_SEARCH_TYPE
    assert search_tool["name"] == "tool_search_tool_bm25"


@pytest.mark.asyncio
async def test_structured_tool_use_suppresses_tool_search() -> None:
    llm = _make_llm()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    prepared = await llm._prepare_tools(
        "claude-sonnet-4-6",
        structured_schema=schema,
        tools=[_tool("alpha")],
        structured_mode="tool_use",
        tool_search_active=True,
    )

    assert len(prepared) == 1
    assert prepared[0]["name"] == "return_structured_output"
    assert "defer_loading" not in prepared[0]


@pytest.mark.asyncio
async def test_json_structured_mode_keeps_deferred_tools() -> None:
    llm = _make_llm()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    prepared = await llm._prepare_tools(
        "claude-sonnet-4-6",
        structured_schema=schema,
        tools=[_tool("alpha")],
        structured_mode="json",
        tool_search_active=True,
    )

    assert len(prepared) == 2
    assert prepared[0]["name"] == "alpha"
    assert prepared[0].get("defer_loading") is True


def test_resolve_policy_off_by_default() -> None:
    llm = _make_llm()

    assert RequestParams().tool_search == "off"
    assert llm._resolve_tool_search_active(None, 100) is False


def test_resolve_policy_always_active_at_any_tool_count() -> None:
    llm = _make_llm()

    assert llm._resolve_tool_search_active(RequestParams(tool_search="always"), 1) is True


def test_resolve_policy_auto_uses_tool_threshold() -> None:
    llm = _make_llm()
    params = RequestParams(tool_search="auto")
    tools = [_tool(f"tool_{i}") for i in range(TOOL_SEARCH_AUTO_TOOL_THRESHOLD)]

    assert llm._resolve_tool_search_active(params, len(tools) - 1) is False
    assert llm._resolve_tool_search_active(params, len(tools)) is False
    assert llm._resolve_tool_search_active(params, len(tools) + 1) is True


def test_resolve_policy_uses_default_request_params() -> None:
    llm = _make_llm()
    llm.default_request_params = RequestParams(tool_search="always")

    assert llm._resolve_tool_search_active(None, 1) is True


def test_vertex_does_not_support_tool_search() -> None:
    llm = _make_vertex_llm()

    assert llm._resolve_tool_search_active(RequestParams(tool_search="always"), 50) is False
