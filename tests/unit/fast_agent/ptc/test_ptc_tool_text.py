"""The code tool's name and what its description claims about the sandbox."""

from __future__ import annotations

import pytest
from mcp_types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings, ShellSettings
from fast_agent.context import Context
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_TOOL_NAME

pytest.importorskip("pydantic_monty")


@pytest.mark.asyncio
async def test_code_tool_is_exec_sandbox_end_to_end() -> None:
    assert PTC_TOOL_NAME == "exec_sandbox"
    agent = McpAgent(
        AgentConfig(name="ptc-name", servers=[], shell=True),
        context=Context(
            config=Settings(
                ptc=ProgrammaticToolCallingSettings(enabled=True),
                shell_execution=ShellSettings(show_bash=False),
            ),
            server_registry=ServerRegistry(),
        ),
    )
    try:
        await agent.initialize()
        tools = (await agent.list_tools()).tools
        assert [tool.name for tool in tools] == ["exec_sandbox"]
        assert "`exec_sandbox` invocation" in agent.instruction
        result = await agent.call_tool("exec_sandbox", {"code": "print(6 * 7)"})
        assert not result.is_error
        assert "42" in "\n".join(b.text for b in result.content if isinstance(b, TextContent))
    finally:
        await agent.shutdown()
