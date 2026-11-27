"""PTC integration tests for the built-in ``events`` tool and ``<fastagent:tool-sources>``."""

from __future__ import annotations

import pytest
from mcp_types import ServerCapabilities, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.mcp.common import get_resource_name, get_server_name
from fast_agent.mcp.events_extension import EVENTS_EXTENSION
from fast_agent.mcp.mcp_aggregator import MCPAggregator, NamespacedTool
from fast_agent.ptc.catalog import build_roster
from fast_agent.ptc.vocabulary import PTC_BUILTIN_ORDER
from fast_agent.tools.events_tool import EVENTS_TOOL_NAME

pytest.importorskip("pydantic_monty")


def _events_capabilities() -> ServerCapabilities:
    return ServerCapabilities(extensions={EVENTS_EXTENSION: {}})


def _tools(server: str, count: int) -> list[Tool]:
    return [
        Tool(
            name=f"{server}__op_{index}",
            description="x" * 5_000,
            input_schema={"type": "object"},
        )
        for index in range(count)
    ]


def _install_mcp_tools(aggregator: MCPAggregator, tools: list[Tool]) -> None:
    discovered = [
        NamespacedTool(
            tool=tool.model_copy(update={"name": get_resource_name(tool.name)}),
            server_name=get_server_name(tool.name),
            namespaced_tool_name=tool.name,
        )
        for tool in tools
    ]
    aggregator._namespaced_tool_map = {tool.namespaced_tool_name: tool for tool in discovered}
    aggregator._server_to_tool_map = {}
    for tool in discovered:
        aggregator._server_to_tool_map.setdefault(tool.server_name, []).append(tool)


def test_events_in_ptc_builtin_order_right_after_process() -> None:
    assert "events" in PTC_BUILTIN_ORDER
    assert PTC_BUILTIN_ORDER.index("events") == PTC_BUILTIN_ORDER.index("process") + 1


def test_build_roster_annotates_events_sources_only() -> None:
    roster = build_roster(
        sources={
            "jobs_run": "jobs",
            "jobs_cancel": "jobs",
            "plain_get": "plain",
        },
        deferred={"jobs_run", "jobs_cancel", "plain_get"},
        unavailable={"offline_ev": "no functions loaded", "offline_plain": "no functions loaded"},
        events_sources={"jobs", "offline_ev", "upfront_only_ev"},
    )
    rendered = roster.render()
    assert (
        '- jobs: 2 functions (supports events — use events(action="list", server="jobs"))'
        in rendered
    )
    assert "- plain: 1 function\n" in rendered
    assert (
        '- offline_ev: 0 functions (no functions loaded; supports events — use events(action="list", server="offline_ev"))'
        in rendered
    )
    assert "- offline_plain: 0 functions (no functions loaded)" in rendered
    assert (
        '- upfront_only_ev: 0 functions (supports events — use events(action="list", server="upfront_only_ev"))'
        in rendered
    )


@pytest.mark.asyncio
async def test_mcp_agent_exposes_events_tool_and_annotates_ptc_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fast_agent.ptc.runtime.PTC_SEARCH_EXTRA_BYTE_THRESHOLD", 10_000)

    agent = McpAgent(
        AgentConfig(name="ev-agent", servers=["jobs", "plain"], shell=True),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(search="on")),
            ptc_enabled=True,
        ),
    )
    try:
        # Mark initialized with cached tools and capabilities; no network probes should occur.
        agent._aggregator.initialized = True
        _install_mcp_tools(agent._aggregator, [*_tools("jobs", 3), *_tools("plain", 2)])

        # Without cached events capabilities, events tool is NOT exposed.
        assert agent.events_capable_servers() == frozenset()
        await agent.list_tools()
        ptc = agent._programmatic_tool_runtime()
        assert ptc is not None
        assert EVENTS_TOOL_NAME not in ptc.catalog.functions
        assert "supports events" not in ptc.catalog.roster.render()

        # Once cached capabilities declare MCP Events on 'jobs', events tool + roster hint appear.
        agent._aggregator._capabilities_cache["jobs"] = _events_capabilities()
        assert agent.events_capable_servers() == frozenset({"jobs"})

        await agent.list_tools()
        assert EVENTS_TOOL_NAME in ptc.catalog.functions
        assert EVENTS_TOOL_NAME not in ptc.catalog.deferred
        upfront_names = [fn.python_name for fn in ptc.catalog.upfront]
        assert upfront_names.index("events") == upfront_names.index("process") + 1

        roster_text = ptc.catalog.roster.render()
        assert (
            '- jobs: 3 functions (supports events — use events(action="list", server="jobs"))'
            in roster_text
        )
        assert "- plain: 2 functions\n" in roster_text

        # Direct dispatch via call_tool("events", ...) works cleanly.
        res = await agent.call_tool(EVENTS_TOOL_NAME, {"action": "status"})
        assert not res.is_error
        assert res.structured_content == {"subscriptions": [], "wake_paused": False}
    finally:
        await agent.shutdown()
