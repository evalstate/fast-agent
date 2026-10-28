"""Unchanged PTC clone instructions preserve explicit LLM system prompts."""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_BATCHING_INSTRUCTION
from fast_agent.types import RequestParams


@pytest.mark.asyncio
@pytest.mark.parametrize("for_subagent", [False, True])
@pytest.mark.parametrize("template", ["Answer briefly.", "Workspace: {{workspaceRoot}}"])
@pytest.mark.parametrize("system_prompt", ["Explicit caller prompt", ""])
async def test_ptc_clone_preserves_explicit_system_prompt(
    for_subagent: bool, template: str, system_prompt: str
) -> None:
    agent = McpAgent(
        AgentConfig("ptc-clone", instruction=template, skills=[]),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=True)),
            server_registry=ServerRegistry(),
        ),
    )
    agent.set_instruction_context({"workspaceRoot": "/workspace"})
    try:
        await agent.initialize()
        llm = await agent.attach_llm(
            ModelFactory.create_factory("passthrough"),
            request_params=RequestParams(system_prompt=system_prompt),
        )
        assert llm.default_request_params.system_prompt == system_prompt

        clone = await agent.spawn_isolated_instance(for_subagent=for_subagent)
        try:
            assert clone.instruction == agent.instruction
            assert clone.instruction.count(PTC_BATCHING_INSTRUCTION) == 1
            assert "{{workspaceRoot}}" not in clone.instruction
            assert clone.instruction_template == template
            assert clone.llm is not None
            assert clone.llm is not llm
            assert clone.llm.default_request_params.system_prompt == system_prompt
            assert llm.default_request_params.system_prompt == system_prompt
        finally:
            await clone.shutdown()
    finally:
        await agent.shutdown()
