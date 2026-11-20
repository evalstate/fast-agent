"""PTC functions stay bound to their tool's full destination and origin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import CallToolResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.common import McpDestination, create_namespaced_name
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import MCPAggregator, NamespacedTool
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc.catalog import ToolRoute
from fast_agent.ptc.runtime import ProgrammaticToolRuntime
from fast_agent.session.snapshot import SessionAgentSnapshot
from fast_agent.tools.function_tool_loader import build_default_function_tool

if TYPE_CHECKING:
    from collections.abc import Callable

pytest.importorskip("pydantic_monty")

# Two remote names that differ only after the 64-character namespaced-name limit.
LONG_A = "x" * 61 + "A"
LONG_B = "x" * 61 + "B"


def _discover(aggregator: MCPAggregator, server: str, names: list[str]) -> None:
    """Commit ``names`` as ``server``'s discovered tools, named as discovery names them."""
    discovered = [
        NamespacedTool(
            tool=Tool(name=name, input_schema={"type": "object"}),
            server_name=server,
            namespaced_tool_name=create_namespaced_name(server, name),
        )
        for name in names
    ]
    aggregator._namespaced_tool_map = {tool.namespaced_tool_name: tool for tool in discovered}
    aggregator._server_to_tool_map = {server: discovered}
    aggregator.initialized = True


def _record_server_calls(
    aggregator: MCPAggregator, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    async def execute(*, server_name: str, operation_name: str, **kwargs: Any) -> CallToolResult:
        calls.append((server_name, operation_name))
        return CallToolResult(content=[TextContent(type="text", text=operation_name)])

    monkeypatch.setattr(aggregator, "_execute_on_server", execute)
    return calls


async def _agent() -> McpAgent:
    agent = McpAgent(
        AgentConfig("routes", skills=[], servers=[]),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True)),
            server_registry=ServerRegistry(),
        ),
    )
    await agent.initialize()
    await agent.attach_llm(ModelFactory.create_factory("passthrough"))
    return agent


async def _run(agent: McpAgent, code: str) -> str:
    result = await agent.call_tool("exec_sandbox", {"code": code})
    return get_text(result.content[0]) or ""


async def _python_names(agent: McpAgent) -> dict[str, str]:
    """Python function name by tool name."""
    surface = await agent.tool_surface()
    return {tool.tool.name: tool.python_name for tool in surface.tools if tool.python_name}


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_mid_run", [False, True])
async def test_a_replaced_tool_sharing_a_truncated_name_is_not_called(
    monkeypatch: pytest.MonkeyPatch, refresh_mid_run: bool
) -> None:
    agent = await _agent()
    try:
        aggregator = agent._aggregator
        calls = _record_server_calls(aggregator, monkeypatch)
        _discover(aggregator, "svc", [LONG_A])
        label = create_namespaced_name("svc", LONG_A)
        assert label == create_namespaced_name("svc", LONG_B)

        def replace() -> str:
            """Replace the server's tool while the script runs."""
            _discover(aggregator, "svc", [LONG_B])
            return "replaced"

        agent.add_tool(build_default_function_tool(replace))
        old = (await _python_names(agent))[label]
        if not refresh_mid_run:
            replace()
            await agent.list_tools()

        text = await _run(agent, f"await replace()\nprint(await settle({old}()))")
        assert "is no longer available" in text
        assert calls == []

        # The replacement is a new tool: it is listed and bound under fresh names.
        (new,) = set((await _python_names(agent)).values()) - {old, "replace"}
        await _run(agent, f"await {new}()")
        assert calls == [("svc", LONG_B)]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_resumed_bindings_do_not_reach_a_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = await _agent()
    resumed = await _agent()
    try:
        _discover(saved._aggregator, "svc", [LONG_A])
        old = (await _python_names(saved))[create_namespaced_name("svc", LONG_A)]
        snapshot = SessionAgentSnapshot.model_validate_json(
            SessionAgentSnapshot(ptc_bindings=saved.ptc_bindings).model_dump_json()
        )
        assert snapshot.ptc_bindings == saved.ptc_bindings

        calls = _record_server_calls(resumed._aggregator, monkeypatch)
        _discover(resumed._aggregator, "svc", [LONG_B])
        assert snapshot.ptc_bindings is not None
        assert await resumed.restore_ptc_bindings(snapshot.ptc_bindings) == {old}
        text = await _run(resumed, f"print(await settle({old}()))")
        assert "is no longer available" in text
        assert calls == []
    finally:
        await saved.shutdown()
        await resumed.shutdown()


@pytest.mark.asyncio
async def test_aggregator_does_not_call_an_undiscovered_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = await _agent()
    try:
        aggregator = agent._aggregator
        calls = _record_server_calls(aggregator, monkeypatch)
        _discover(aggregator, "svc", [LONG_B])
        label = create_namespaced_name("svc", LONG_A)

        stale = await aggregator.call_tool(label, {}, destination=McpDestination("svc", LONG_A))
        assert stale.is_error and calls == []
        await aggregator.call_tool(label, {}, destination=McpDestination("svc", LONG_B))
        assert calls == [("svc", LONG_B)]
    finally:
        await agent.shutdown()


def _local(name: str) -> Callable[[], str]:
    def tool() -> str:
        """A local tool."""
        return name

    return tool


@pytest.mark.asyncio
async def test_local_function_with_a_server_like_name_is_searched_as_local() -> None:
    agent = await _agent()
    assert agent.context is not None
    agent.context.ptc_search_forced = True
    try:
        _discover(agent._aggregator, "billing", ["invoice", "refund_status", "quote", "credit"])
        agent.add_tool(
            build_default_function_tool(_local("billing__refund"), name="billing__refund")
        )
        surface = await agent.tool_surface()
        sources = {tool.tool.name: tool.source for tool in surface.tools if tool.python_name}
        assert sources["billing__refund"] == "local"
        assert sources["billing__invoice"] == "billing"

        local = await _run(agent, 'print(await search_tools(server="local"))')
        billing = await _run(agent, 'print(await search_tools(server="billing"))')
        assert "billing__refund(" in local and "billing__invoice(" not in local
        assert "billing__invoice(" in billing and "billing__refund(" not in billing
    finally:
        await agent.shutdown()


def test_route_origin_decides_the_source_group() -> None:
    tools = [
        Tool(name=name, input_schema={"type": "object"})
        for name in ("billing__refund", "billing__invoice", "card__tool", "plain", "other")
    ]
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools(
        tools,
        sources={"card__tool": "card"},
        routes={
            "billing__refund": ToolRoute("local", "billing__refund"),
            "billing__invoice": ToolRoute(
                "mcp", "billing__invoice", McpDestination("billing", "invoice")
            ),
            "card__tool": ToolRoute("local", "card__tool"),
        },
    )
    assert runtime.catalog.sources == {
        "billing__refund": "local",
        "billing__invoice": "billing",
        "card__tool": "card",
        "plain": "local",
        "other": "local",
    }

    # Without routes, a namespaced name is still read as an MCP tool.
    runtime = ProgrammaticToolRuntime(ProgrammaticToolCallingSettings(enabled=True))
    runtime.model_tools(tools[:1])
    assert runtime.catalog.sources == {"billing__refund": "billing"}
