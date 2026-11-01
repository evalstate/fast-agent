"""PTC batching guidance reaches agent instructions without contaminating templates."""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_BATCHING_INSTRUCTION, PTC_TOOL_NAME
from fast_agent.types import RequestParams


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("template", ["", "Answer briefly."])
async def test_batching_guidance_is_only_added_with_ptc(enabled: bool, template: str) -> None:
    config = AgentConfig(
        "ptc-instructions", instruction=template, default_request_params=RequestParams()
    )
    agent = McpAgent(
        config,
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=enabled)),
            server_registry=ServerRegistry(),
        ),
    )
    try:
        await agent.initialize()
        assert (PTC_BATCHING_INSTRUCTION in agent.instruction) is enabled
        assert agent.instruction_template == template
        assert config.instruction == template
        assert config.default_request_params is not None
        assert config.default_request_params.system_prompt == agent.instruction
        if not enabled:
            assert agent.instruction == template
            return

        tools = (await agent.list_tools()).tools
        code_tool = next(tool for tool in tools if tool.name == PTC_TOOL_NAME)
        assert code_tool.description is not None
        assert code_tool.description.index(PTC_BATCHING_INSTRUCTION) < code_tool.description.index(
            "Sandbox:"
        )
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_batching_guidance_survives_refresh_and_cloning_without_duplication() -> None:
    template = "Workspace: {{workspaceRoot}}"
    settings = Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))
    agent = McpAgent(
        AgentConfig("ptc-instructions", instruction=template),
        context=Context(config=settings, server_registry=ServerRegistry()),
    )
    agent.set_instruction_context({"workspaceRoot": "/first"})
    try:
        await agent.initialize()
        for workspace in ("/second", "/third"):
            await rebuild_agent_instruction(agent, context={"workspaceRoot": workspace})
            assert agent.instruction.startswith(f"Workspace: {workspace}")
            assert agent.instruction.count(PTC_BATCHING_INSTRUCTION) == 1
            assert agent.instruction_template == template

        for for_subagent in (False, True):
            clone = await agent.spawn_isolated_instance(for_subagent=for_subagent)
            try:
                assert clone.instruction == agent.instruction
                assert clone.config.instruction == template
            finally:
                await clone.shutdown()

        settings.ptc.enabled = False
        await rebuild_agent_instruction(agent)
        assert agent.instruction == "Workspace: /third"
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_context_refresh", [False, True])
async def test_empty_template_guidance_tracks_ptc_when_refreshed(use_context_refresh: bool) -> None:
    settings = Settings(ptc=ProgrammaticToolCallingSettings(enabled=True))
    agent = McpAgent(
        AgentConfig("ptc-instructions", instruction=""),
        context=Context(config=settings, server_registry=ServerRegistry()),
    )
    try:
        await agent.initialize()
        for enabled in (False, True, False):
            settings.ptc.enabled = enabled
            if use_context_refresh:
                await apply_instruction_context([agent], {})
            else:
                await rebuild_agent_instruction(agent)
            assert agent.instruction == (PTC_BATCHING_INSTRUCTION if enabled else "")
            assert agent.instruction_template == ""
    finally:
        await agent.shutdown()
