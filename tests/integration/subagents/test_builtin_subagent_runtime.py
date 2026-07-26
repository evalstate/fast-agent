"""Managed-runtime coverage for the built-in subagent tool."""

import pytest

from fast_agent.agents.subagent_tool import SUBAGENT_TOOL_NAME
from fast_agent.mcp.helpers.content_helpers import get_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enabled_agent_can_run_builtin_subagent(fast_agent) -> None:
    @fast_agent.agent(
        name="parent",
        instruction="Complete the requested task.",
        model="passthrough",
        subagents=True,
    )
    async def run_scenario() -> None:
        async with fast_agent.run() as app:
            parent = app.parent
            tools = await parent.list_tools()
            assert "subagent" in {tool.name for tool in tools.tools}

            result = await parent._execution_tools["subagent"].run({"message": "hello"})
            assert get_text(result.content[0]) == "hello"

    await run_scenario()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_builtin_subagent_is_installed_on_each_enabled_visible_agent(fast_agent) -> None:
    @fast_agent.agent(name="first", model="passthrough", subagents=True)
    @fast_agent.agent(name="second", model="passthrough", subagents=True)
    async def run_scenario() -> None:
        async with fast_agent.run() as app:
            assert "subagent" in {tool.name for tool in (await app.first.list_tools()).tools}
            assert "subagent" in {tool.name for tool in (await app.second.list_tools()).tools}

    await run_scenario()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_managed_runtime_respects_subagent_activation_controls(fast_agent) -> None:
    @fast_agent.agent(name="inherited", model="passthrough")
    @fast_agent.agent(name="enabled", model="passthrough", subagents=True)
    @fast_agent.agent(name="disabled", model="passthrough", subagents=False)
    async def run_scenario() -> None:
        async with fast_agent.run() as app:
            assert SUBAGENT_TOOL_NAME not in {
                tool.name for tool in (await app.inherited.list_tools()).tools
            }
            assert SUBAGENT_TOOL_NAME in {
                tool.name for tool in (await app.enabled.list_tools()).tools
            }
            assert SUBAGENT_TOOL_NAME not in {
                tool.name for tool in (await app.disabled.list_tools()).tools
            }

    await run_scenario()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_managed_runtime_activates_from_hidden_instruction_directive(fast_agent) -> None:
    instruction = "<!-- fast-agent-subagents -->\nComplete the requested task."

    @fast_agent.agent(name="directive", instruction=instruction, model="passthrough")
    @fast_agent.agent(
        name="disabled",
        instruction=instruction,
        model="passthrough",
        subagents=False,
    )
    async def run_scenario() -> None:
        async with fast_agent.run() as app:
            assert SUBAGENT_TOOL_NAME in {
                tool.name for tool in (await app.directive.list_tools()).tools
            }
            assert app.directive.config.subagent_activation_source == "instruction"
            assert "fast-agent-subagents" not in app.directive.instruction

            assert SUBAGENT_TOOL_NAME not in {
                tool.name for tool in (await app.disabled.list_tools()).tools
            }
            assert "fast-agent-subagents" not in app.disabled.instruction

    await run_scenario()
