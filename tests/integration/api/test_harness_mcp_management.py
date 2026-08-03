from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings
from fast_agent.context import initialize_context
from fast_agent.core.harness_tools import SLASH_COMMAND_TOOL_NAME, set_harness_tools
from fast_agent.mcp.helpers.content_helpers import get_text


@pytest.mark.asyncio
async def test_harness_tool_connects_and_disconnects_live_mcp_server() -> None:
    context = await initialize_context(
        Settings(default_model="passthrough"),
        store_globally=False,
    )
    agent = McpAgent(
        config=AgentConfig(
            "harness-smoke",
            instruction="{{serverInstructions}}",
            harness_tools=True,
            shell=True,
        ),
        context=context,
    )
    await agent.initialize()
    set_harness_tools(agent, True)

    server = Path(__file__).with_name("mcp_tools_server.py")
    try:
        connected = await agent.call_tool(
            SLASH_COMMAND_TOOL_NAME,
            {
                "command": (
                    f"/mcp connect --name smoke --no-oauth uv run {shlex.quote(str(server))}"
                )
            },
        )
        assert not connected.is_error, get_text(connected.content[0])
        assert "smoke__check_weather" in {tool.name for tool in (await agent.list_tools()).tools}
        assert "Here is how to use this server" in agent.instruction

        disconnected = await agent.call_tool(
            SLASH_COMMAND_TOOL_NAME,
            {"command": "/mcp disconnect smoke"},
        )
        assert not disconnected.is_error, get_text(disconnected.content[0])
        assert "smoke__check_weather" not in {
            tool.name for tool in (await agent.list_tools()).tools
        }
        assert "Here is how to use this server" not in agent.instruction
    finally:
        await agent.shutdown()
