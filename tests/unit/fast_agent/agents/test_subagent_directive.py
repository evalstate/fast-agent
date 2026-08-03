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
        f"<!-- {SUBAGENT_DIRECTIVE} --!>",
        f"  <!--   {SUBAGENT_DIRECTIVE}   -->  ",
    ],
)
def test_resolve_subagent_directive_strips_exact_standalone_lines(directive: str) -> None:
    resolved = resolve_subagent_directive(f"Before\n{directive}\nAfter")

    assert resolved.found is True
    assert resolved.instruction == "Before\nAfter"
    assert resolved.subagent_instruction == "Before\nAfter"


def test_resolve_subagent_directive_ignores_prose_and_longer_tokens() -> None:
    instruction = (
        "This project does not use fast-agent-subagents.\n"
        "The fast-agent-subagents-disabled setting is unrelated."
    )

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is False
    assert resolved.instruction == instruction
    assert resolved.subagent_instruction == instruction


@pytest.mark.parametrize("closer", ["-->", "--!>"])
def test_multiline_directive_body_is_parent_only(closer: str) -> None:
    instruction = f"Before\n<!-- fast-agent-subagents\nuse terra for analysis\n{closer}\nAfter"

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is True
    assert resolved.instruction == "Before\nuse terra for analysis\nAfter"
    assert resolved.subagent_instruction == "Before\nAfter"


@pytest.mark.parametrize("closer", ["-->", "--!>"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_multiline_directive_accepts_inline_closer(line_ending: str, closer: str) -> None:
    instruction = (
        f"Before{line_ending}"
        f"<!-- fast-agent-subagents{line_ending}"
        f"use terra for analysis \t{closer} \t{line_ending}"
        "After"
    )

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is True
    assert resolved.instruction == (f"Before{line_ending}use terra for analysis{line_ending}After")
    assert resolved.subagent_instruction == f"Before{line_ending}After"


@pytest.mark.parametrize("closer", ["-->", "--!>"])
def test_multiline_directive_accepts_inline_closer_at_eof(closer: str) -> None:
    instruction = f"Before\n<!-- fast-agent-subagents\nuse terra for analysis {closer}"

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is True
    assert resolved.instruction == "Before\nuse terra for analysis"
    assert resolved.subagent_instruction == "Before\n"


@pytest.mark.parametrize(
    "instruction",
    [
        "<!-- fast-agent-subagents\nuse terra for analysis",
        "<!-- fast-agent-subagents\nuse terra for analysis --> trailing text",
        "<!-- fast-agent-subagents\nuse terra for analysis --!> trailing text",
    ],
)
def test_invalid_multiline_directive_is_ignored(instruction: str) -> None:
    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is False
    assert resolved.instruction == instruction
    assert resolved.subagent_instruction == instruction


@pytest.mark.parametrize(
    "marker",
    [
        "fast-agent-subagents",
        "<!-- fast-agent-subagents -->",
    ],
)
def test_unclosed_multiline_directive_does_not_hide_later_marker(marker: str) -> None:
    instruction = f"<!-- fast-agent-subagents\nunclosed directive\n{marker}\nIncluded rules."

    resolved = resolve_subagent_directive(instruction)

    assert resolved.found is True
    assert resolved.instruction == (
        "<!-- fast-agent-subagents\nunclosed directive\nIncluded rules."
    )
    assert resolved.subagent_instruction == resolved.instruction


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
async def test_mcp_directive_template_is_projected_when_rendered() -> None:
    template = "fast-agent-subagents\nWorkspace: {{workspaceRoot}}"
    config = AgentConfig("dev", instruction=template)
    agent = McpAgent(config)
    agent.set_instruction_context({"workspaceRoot": "/first"})
    await agent.initialize()

    assert install_subagent_tool(agent) is True
    assert agent.instruction_template == template
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert config.instruction == template

    await rebuild_agent_instruction(agent, context={"workspaceRoot": "/second"})
    assert agent.instruction == "Workspace: /second"

    clone = await agent.spawn_isolated_instance()
    try:
        assert clone.instruction_template == template
        assert clone.instruction == "Workspace: /second"
    finally:
        await clone.shutdown()
        await agent.shutdown()


@pytest.mark.asyncio
async def test_mcp_directive_from_include_is_stripped_after_every_render(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "<!-- fast-agent-subagents\nUse terra for analysis.\n-->\nIncluded rules.",
        encoding="utf-8",
    )
    template = "Project rules:\n{{file_silent:AGENTS.md}}"
    agent = McpAgent(AgentConfig("dev", instruction=template))
    agent.set_instruction_context({"workspaceRoot": str(tmp_path)})
    await agent.initialize()

    assert install_subagent_tool(agent) is True
    assert agent.instruction_template == template
    assert SUBAGENT_DIRECTIVE not in agent.instruction
    assert "Use terra for analysis." in agent.instruction
    assert "Included rules." in agent.instruction

    await rebuild_agent_instruction(agent)
    assert SUBAGENT_DIRECTIVE not in agent.instruction

    clone = await agent.spawn_isolated_instance()
    try:
        assert clone.instruction_template == template
        assert SUBAGENT_DIRECTIVE not in clone.instruction
        assert "Use terra for analysis." in clone.instruction
        assert "Included rules." in clone.instruction
    finally:
        await clone.shutdown()

    child = await agent.spawn_isolated_instance(for_subagent=True)
    try:
        assert child.config.subagent_child is True
        assert child.config.subagents is False
        assert SUBAGENT_DIRECTIVE not in child.instruction
        assert "Use terra for analysis." not in child.instruction
        assert "Included rules." in child.instruction
    finally:
        await child.shutdown()
        await agent.shutdown()


@pytest.mark.asyncio
async def test_mcp_refresh_can_enable_subagents_from_updated_include(tmp_path: Path) -> None:
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("Included rules.", encoding="utf-8")
    agent = McpAgent(AgentConfig("dev", instruction="Project rules:\n{{file_silent:AGENTS.md}}"))
    agent.set_instruction_context({"workspaceRoot": str(tmp_path)})
    await agent.initialize()

    assert install_subagent_tool(agent) is False
    agents_file.write_text(
        "<!-- fast-agent-subagents\nUse terra for analysis.\n-->\nIncluded rules.",
        encoding="utf-8",
    )

    await rebuild_agent_instruction(agent)

    assert agent.config.subagents is True
    assert "Use terra for analysis." in agent.instruction
    assert SUBAGENT_TOOL_NAME in {tool.name for tool in (await agent.list_tools()).tools}
    await agent.shutdown()
