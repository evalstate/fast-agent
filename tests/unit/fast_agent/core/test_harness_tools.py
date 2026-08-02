from __future__ import annotations

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.harness_tools import (
    GET_RESOURCE_TOOL_NAME,
    HARNESS_TOOL_NAMES,
    SLASH_COMMAND_TOOL_NAME,
    set_harness_tools,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.function_tool_loader import build_default_function_tool


def _tool_names(agent: ToolAgent) -> set[str]:
    return set(agent._execution_tools)


@pytest.mark.asyncio
async def test_harness_tools_install_execute_and_disable() -> None:
    agent = ToolAgent(AgentConfig("dev", instruction="System details", harness_tools=True))

    assert set_harness_tools(agent, True)
    assert HARNESS_TOOL_NAMES <= _tool_names(agent)
    assert agent._execution_tools[SLASH_COMMAND_TOOL_NAME].parameters["properties"]["command"] == {
        "type": "string"
    }
    assert agent._execution_tools[GET_RESOURCE_TOOL_NAME].parameters["properties"]["uri"] == {
        "type": "string"
    }

    commands = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": "/commands"})
    resource = await agent.call_tool(
        GET_RESOURCE_TOOL_NAME,
        {"uri": "internal://fast-agent/agent-cards"},
    )

    commands_text = get_text(commands.content[0])
    resource_text = get_text(resource.content[0])
    assert commands_text is not None
    assert resource_text is not None
    assert "/usage" in commands_text
    assert "AgentCard" in resource_text

    for command, expected in (
        ("/usage", "No usage data available."),
        ("/system", "System details"),
        ("/status", "No MCP status is available for this agent."),
    ):
        result = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": command})
        result_text = get_text(result.content[0])
        assert not result.is_error
        assert result_text is not None
        assert expected in result_text

    removed_status_alias = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcpstatus"},
    )
    removed_status_alias_text = get_text(removed_status_alias.content[0])
    assert removed_status_alias.is_error
    assert removed_status_alias_text is not None
    assert "Unsupported harness command" in removed_status_alias_text

    rejected = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect https://example.com/mcp"},
    )
    rejected_text = get_text(rejected.content[0])
    assert rejected.is_error
    assert rejected_text is not None
    assert "Unsupported harness command" in rejected_text

    assert not set_harness_tools(agent, False)
    assert HARNESS_TOOL_NAMES.isdisjoint(_tool_names(agent))


def test_harness_tools_do_not_replace_user_tools() -> None:
    def slash_command(command: str) -> str:
        return command

    agent = ToolAgent(
        AgentConfig("dev"),
        tools=[build_default_function_tool(slash_command)],
    )

    with pytest.raises(AgentConfigError, match="reserved"):
        set_harness_tools(agent, True)

    assert SLASH_COMMAND_TOOL_NAME in _tool_names(agent)


@pytest.mark.asyncio
async def test_isolated_clone_does_not_inherit_harness_tools() -> None:
    parent = ToolAgent(AgentConfig("dev", harness_tools=True))
    set_harness_tools(parent, True)

    clone = await parent.spawn_isolated_instance()
    try:
        assert clone.config.harness_tools is False
        assert HARNESS_TOOL_NAMES.isdisjoint(_tool_names(clone))
        assert HARNESS_TOOL_NAMES <= _tool_names(parent)
    finally:
        await clone.shutdown()
