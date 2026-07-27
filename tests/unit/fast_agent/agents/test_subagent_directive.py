from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_directive import (
    SUBAGENT_DIRECTIVE,
    resolve_subagent_directive,
)
from fast_agent.agents.subagent_tool import SUBAGENT_TOOL_NAME, install_subagent_tool
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.instruction import InstructionBuilder
from fast_agent.core.instruction_refresh import rebuild_agent_instruction


@pytest.mark.parametrize(
    "directive",
    [
        SUBAGENT_DIRECTIVE,
        f"  {SUBAGENT_DIRECTIVE}  ",
        f"<!-- {SUBAGENT_DIRECTIVE} -->",
        f"  <!--   {SUBAGENT_DIRECTIVE}   -->  ",
    ],
)
def test_resolve_subagent_directive_strips_exact_standalone_lines(directive: str) -> None:
    resolved = resolve_subagent_directive(f"Before\n{directive}\nAfter")

    assert resolved.found is True
    assert resolved.instruction == "Before\nAfter"


def test_resolve_subagent_directive_ignores_prose_and_longer_tokens() -> None:
    instruction = (
        "This project does not use fast-agent-subagents.\n"
        "The fast-agent-subagents-disabled setting is unrelated."
    )

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is False
    assert resolved.instruction == instruction


@pytest.mark.asyncio
async def test_agents_file_directive_enables_tool_and_is_hidden_from_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "<!-- fast-agent-subagents -->\nUse concise delegated reviews.\n",
        encoding="utf-8",
    )
    builder = InstructionBuilder("Project rules:\n{{file_silent:AGENTS.md}}")
    builder.set("workspaceRoot", str(tmp_path))
    instruction = await builder.build()
    agent = ToolAgent(AgentConfig("dev", instruction=instruction))

    assert install_subagent_tool(agent) is True
    assert agent.config.subagents is True
    assert agent.config.subagent_activation_source == "instruction"
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert "Use concise delegated reviews." in agent.instruction
    assert SUBAGENT_TOOL_NAME in {tool.name for tool in (await agent.list_tools()).tools}


@pytest.mark.asyncio
async def test_explicit_disable_wins_but_directive_is_still_hidden() -> None:
    agent = ToolAgent(
        AgentConfig(
            "dev",
            instruction="<!-- fast-agent-subagents -->\nBe concise.",
            subagents=False,
        )
    )

    assert install_subagent_tool(agent) is False
    assert agent.config.subagents is False
    assert agent.config.subagent_activation_source == "configuration"
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert SUBAGENT_TOOL_NAME not in {tool.name for tool in (await agent.list_tools()).tools}


def test_explicit_enable_keeps_configuration_as_activation_source() -> None:
    agent = ToolAgent(
        AgentConfig(
            "dev",
            instruction="fast-agent-subagents\nBe concise.",
            subagents=True,
        )
    )

    assert install_subagent_tool(agent) is True
    assert agent.config.subagent_activation_source == "configuration"


@pytest.mark.asyncio
async def test_mcp_directive_is_removed_from_source_template_and_clones() -> None:
    template = "fast-agent-subagents\nWorkspace: {{workspaceRoot}}"
    config = AgentConfig("dev", instruction=template)
    agent = McpAgent(config)
    agent.set_instruction_context({"workspaceRoot": "/first"})
    await agent.initialize()

    assert install_subagent_tool(agent) is True
    assert SUBAGENT_DIRECTIVE not in agent.instruction_template
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert config.instruction == template

    await rebuild_agent_instruction(agent, context={"workspaceRoot": "/second"})
    assert agent.instruction == "Workspace: /second"

    clone = await agent.spawn_isolated_instance()
    try:
        assert clone.instruction_template == "Workspace: {{workspaceRoot}}"
        assert clone.instruction == "Workspace: /second"
    finally:
        await clone.shutdown()
        await agent.shutdown()


@pytest.mark.asyncio
async def test_mcp_directive_from_include_is_stripped_after_every_render(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "fast-agent-subagents\nIncluded rules.",
        encoding="utf-8",
    )
    template = "Project rules:\n{{file_silent:AGENTS.md}}"
    agent = McpAgent(AgentConfig("dev", instruction=template))
    agent.set_instruction_context({"workspaceRoot": str(tmp_path)})
    await agent.initialize()

    assert install_subagent_tool(agent) is True
    assert agent.instruction_template == template
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert "Included rules." in agent.instruction

    await rebuild_agent_instruction(agent)
    assert SUBAGENT_DIRECTIVE not in agent.instruction

    clone = await agent.spawn_isolated_instance()
    try:
        assert clone.instruction_template == template
        assert SUBAGENT_DIRECTIVE not in clone.instruction
        assert "Included rules." in clone.instruction
    finally:
        await clone.shutdown()
        await agent.shutdown()
