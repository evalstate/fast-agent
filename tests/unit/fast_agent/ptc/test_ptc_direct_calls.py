"""Real dispatch contracts: code-only tools cannot be reached by direct calls."""

from pathlib import Path
from typing import Literal

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import (
    MCPServerSettings,
    MCPSettings,
    ProgrammaticToolCallingSettings,
    Settings,
)
from fast_agent.context import Context
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.types import PromptMessageExtended


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nested", "direct", "off"])
@pytest.mark.parametrize("names", ["canonical", "profile"])
async def test_mcp_direct_and_code_dispatch(
    tmp_path: Path, mode: str, names: Literal["canonical", "profile"]
) -> None:
    record = tmp_path / "received"
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(
            enabled=mode != "off",
            function_names=names,
            direct_tools=["recording__record_call"] if mode == "direct" else [],
        ),
        mcp=MCPSettings(
            servers={
                "recording": MCPServerSettings(
                    command="uv",
                    args=[
                        "run",
                        str(Path(__file__).parent / "fixtures" / "recording_server.py"),
                        str(record),
                    ],
                )
            }
        ),
    )
    agent = McpAgent(
        AgentConfig("dispatch", model="passthrough", servers=["recording"], skills=[]),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        # Exercise both the public dispatch boundary and the model's turn planner,
        # including the unprefixed alias accepted by the MCP router.
        for name in ("recording__record_call", "record_call"):
            result = await agent.call_tool(name, {"value": "direct"})
            assert bool(result.is_error) is (mode == "nested")
            request = PromptMessageExtended(
                role="assistant",
                content=[],
                tool_calls={
                    "call": CallToolRequest(
                        params=CallToolRequestParams(name=name, arguments={"value": "turn"})
                    )
                },
            )
            turn = await agent.run_tools(request)
            assert turn.tool_results is not None
            assert bool(turn.tool_results["call"].is_error) is (mode == "nested")
            if mode == "nested":
                for rejected in (result, turn.tool_results["call"]):
                    text = "\n".join(b.text for b in rejected.content if isinstance(b, TextContent))
                    assert "not a direct tool" in text
                    assert PTC_TOOL_NAME in text
                    assert "await recording__record_call(" in text
                assert not record.exists()
        if mode == "nested":
            result = await agent.call_tool(
                PTC_TOOL_NAME, {"code": "await recording__record_call(value='nested')"}
            )
            assert not result.is_error, result
            assert record.read_text().splitlines() == ["nested"]
        else:
            assert record.read_text().splitlines() == ["direct", "turn", "direct", "turn"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_function_and_card_tools_reject_direct_calls(wrapped: bool) -> None:
    received: list[str] = []

    def card_tool(value: str) -> str:
        received.append(value)
        return value

    agent = McpAgent(
        AgentConfig("local", model="passthrough", servers=[], skills=[]),
        tools=[build_default_function_tool(card_tool)] if wrapped else [card_tool],
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        result = await agent.call_tool("card_tool", {"value": "direct"})
        assert result.is_error and "await card_tool(" in str(result.content)
        assert received == []
        result = await agent.call_tool(PTC_TOOL_NAME, {"code": "await card_tool(value='nested')"})
        assert not result.is_error, result
        assert received == ["nested"]
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("no_shell", [False, True])
async def test_skill_readers_are_code_only(tmp_path: Path, no_shell: bool) -> None:
    root = tmp_path / "skills"
    skill = root / "alpha" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: alpha\ndescription: Test skill\n---\nSkill body\n")
    agent = McpAgent(
        AgentConfig(
            "skills",
            model="passthrough",
            servers=[],
            skills=root,
            skill_manifests=SkillRegistry.load_directory(root),
        ),
        context=Context(config=Settings(), ptc_enabled=True, no_shell=no_shell),
    )
    try:
        name = "read_skill" if no_shell else "read_text_file"
        arguments = {"path": str(skill)}
        result = await agent.call_tool(name, arguments)
        assert result.is_error and f"await {name}(" in str(result.content)
        result = await agent.call_tool(
            PTC_TOOL_NAME, {"code": f"await {name}(path={str(skill)!r})"}
        )
        assert not result.is_error, result
        assert "Skill body" in str(result.content)
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_builtin_guard_and_direct_harness_tool() -> None:
    agent = McpAgent(
        AgentConfig("builtins", model="passthrough", servers=[], shell=True, skills=[]),
        context=Context(config=Settings(), ptc_enabled=True),
    )
    try:
        set_harness_tools(agent, True)
        for name in ("shell", "bash", "process", "read_text_file", "get_resource"):
            result = await agent.call_tool(name, {})
            assert result.is_error and "not a direct tool" in str(result.content)
        result = await agent.call_tool("slash_command", {"command": "/status"})
        assert not result.is_error, result
    finally:
        await agent.shutdown()
