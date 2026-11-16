"""Only model direct calls are constrained by the code-only PTC surface."""

from pathlib import Path
from typing import Literal

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import (
    MCPServerSettings,
    MCPSettings,
    ProgrammaticToolCallingSettings,
    Settings,
    ShellSettings,
)
from fast_agent.context import Context
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_TOOL_NAME
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime
from fast_agent.types import PromptMessageExtended


def _recording_server(record: Path) -> MCPServerSettings:
    return MCPServerSettings(
        command="uv",
        args=["run", str(Path(__file__).parent / "fixtures" / "recording_server.py"), str(record)],
    )


async def _model_call(agent: McpAgent, name: str, arguments: dict[str, str]) -> CallToolResult:
    turn = await agent.run_tools(
        PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "call": CallToolRequest(
                    params=CallToolRequestParams(name=name, arguments=arguments)
                )
            },
        )
    )
    assert turn.tool_results is not None
    return turn.tool_results["call"]


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
        mcp=MCPSettings(servers={"recording": _recording_server(record)}),
    )
    agent = McpAgent(
        AgentConfig(
            "dispatch",
            model="passthrough",
            servers=["recording"],
            skills=[],
            tools={"recording": ["record_call"]},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        # Host calls remain unrestricted, including the MCP router's unprefixed alias.
        received: list[str] = []
        for name in ("recording__record_call", "record_call"):
            result = await agent.call_tool(name, {"value": "host"})
            assert not result.is_error, result
            received.append("host")
            assert record.read_text().splitlines() == received

            result = await _model_call(agent, name, {"value": "turn"})
            assert bool(result.is_error) is (mode == "nested")
            if mode == "nested":
                text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
                assert "not a direct tool" in text
                assert PTC_TOOL_NAME in text
                assert "await recording__record_call(" in text
            else:
                received.append("turn")
            # A rejected model call must not reach the server.
            assert record.read_text().splitlines() == received
        if mode == "nested":
            result = await _model_call(
                agent, PTC_TOOL_NAME, {"code": "await recording__record_call(value='nested')"}
            )
            assert not result.is_error, result
            assert record.read_text().splitlines() == [*received, "nested"]
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
        result = await _model_call(agent, "card_tool", {"value": "direct"})
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
        result = await _model_call(agent, name, arguments)
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
        context=Context(
            config=Settings(shell_execution=ShellSettings(enable_attach_media="on")),
            ptc_enabled=True,
        ),
    )
    try:
        set_harness_tools(agent, True)
        for name in ("shell", "bash", "process", "read_text_file", "get_resource", "attach_media"):
            result = await _model_call(agent, name, {})
            assert result.is_error and "not a direct tool" in str(result.content)
        result = await _model_call(agent, "slash_command", {"command": "/status"})
        assert not result.is_error, result
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "names"), [(False, "canonical"), (True, "canonical"), (True, "profile")]
)
@pytest.mark.parametrize(
    ("patterns", "direct_tools", "allowed", "hidden"),
    [
        (None, [], True, True),
        (["allowed_call"], [], True, False),
        (["allowed_*"], [], True, False),
        (["*call"], [], True, True),
        ([], [], False, False),
        (["allowed_call"], ["recording__hidden_call"], True, False),
        (["allowed_call"], ["*"], True, False),
    ],
)
async def test_mcp_filter_constrains_model_routes(
    tmp_path: Path,
    enabled: bool,
    names: Literal["canonical", "profile"],
    patterns: list[str] | None,
    direct_tools: list[str],
    allowed: bool,
    hidden: bool,
) -> None:
    record = tmp_path / "received"
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(
            enabled=enabled, function_names=names, direct_tools=direct_tools
        ),
        mcp=MCPSettings(servers={"recording": _recording_server(record)}),
    )
    agent = McpAgent(
        AgentConfig(
            "filtered",
            model="passthrough",
            servers=["recording"],
            skills=[],
            tools={} if patterns is None else {"recording": patterns},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    received: list[str] = []

    def receipts() -> list[str]:
        return record.read_text().splitlines() if record.exists() else []

    try:
        await agent.initialize()
        for local_name, permitted in (("hidden_call", hidden), ("allowed_call", allowed)):
            for name in (
                f"recording__{local_name}",
                local_name,
                f"RECORDING__{local_name.upper()}",
                local_name.upper(),
            ):
                # Only advertised (namespaced) names get case correction today.
                executes = permitted and name != local_name.upper()
                result = await _model_call(agent, name, {"value": name})
                assert bool(result.is_error) is not executes, result
                if executes:
                    received.append(name)
                else:
                    assert "not available" in str(result.content)
                assert receipts() == received

                if enabled:
                    # Sandbox functions have their advertised names, not router aliases.
                    direct_only = "*" in direct_tools or f"recording__{local_name}" in direct_tools
                    nested_executes = (
                        permitted and name == f"recording__{local_name}" and not direct_only
                    )
                    result = await _model_call(
                        agent, PTC_TOOL_NAME, {"code": f"await {name}(value='nested')"}
                    )
                    assert bool(result.is_error) is not nested_executes, result
                    if nested_executes:
                        received.append("nested")
                    assert receipts() == received

        # tools: constrains model access, not trusted host calls.
        for name in ("recording__hidden_call", "hidden_call"):
            result = await agent.call_tool(name, {"value": "host"})
            assert not result.is_error, result
            received.append("host")
            assert receipts() == received
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_mcp_alias_checks_destination_not_other_servers(tmp_path: Path) -> None:
    denied_record = tmp_path / "denied"
    allowed_record = tmp_path / "allowed"
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(enabled=False),
        mcp=MCPSettings(
            servers={
                "first": _recording_server(denied_record),
                "second": _recording_server(allowed_record),
            }
        ),
    )
    agent = McpAgent(
        AgentConfig(
            "collision",
            model="passthrough",
            servers=["first", "second"],
            tools={"first": []},
            skills=[],
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        result = await _model_call(agent, "allowed_call", {"value": "alias"})
        assert result.is_error and "not available" in str(result.content)
        assert not denied_record.exists()
        assert not allowed_record.exists()

        # An omitted server filter still allows its tools, but cannot grant another
        # server's alias access or change the router's first-server precedence.
        result = await _model_call(agent, "second__allowed_call", {"value": "namespaced"})
        assert not result.is_error, result
        assert allowed_record.read_text().splitlines() == ["namespaced"]
        assert not denied_record.exists()
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "names"), [(False, "canonical"), (True, "canonical"), (True, "profile")]
)
async def test_mcp_filter_blocks_filesystem_collision(
    tmp_path: Path, enabled: bool, names: Literal["canonical", "profile"]
) -> None:
    record = tmp_path / "received"
    local_file = tmp_path / "local.txt"
    local_file.write_text("local content")
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(enabled=enabled, function_names=names),
        mcp=MCPSettings(servers={"recording": _recording_server(record)}),
    )
    agent = McpAgent(
        AgentConfig(
            "filesystem",
            model="passthrough",
            servers=["recording"],
            skills=[],
            tools={"recording": []},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        agent.set_filesystem_runtime(
            LocalFilesystemRuntime(agent.logger, working_directory=tmp_path)
        )
        for name in ("read_text_file", "READ_TEXT_FILE", "recording__read_text_file"):
            result = await _model_call(agent, name, {"path": str(local_file)})
            assert result.is_error and "not available" in str(result.content)
            assert not record.exists()

        # The allowed local implementation is still callable from code and by host.
        if enabled:
            result = await _model_call(
                agent, PTC_TOOL_NAME, {"code": f"await read_text_file(path={str(local_file)!r})"}
            )
            assert not result.is_error and "local content" in str(result.content)
        result = await agent.call_tool("read_text_file", {"path": str(local_file)})
        assert not result.is_error and "local content" in str(result.content)
        assert not record.exists()
    finally:
        await agent.shutdown()
