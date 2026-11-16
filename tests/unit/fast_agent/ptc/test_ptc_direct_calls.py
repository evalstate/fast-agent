"""Only model direct calls are constrained by the code-only PTC surface."""

from pathlib import Path
from typing import Literal

import pytest
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ReadResourceResult,
    TextContent,
)

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
            instruction="{{serverInstructions}}",
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
        # Server instructions list only the tools the filter allows.
        assert ("recording__hidden_call" in agent.instruction) is hidden
        assert ("recording__allowed_call" in agent.instruction) is allowed
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
                    assert str(result.content).count("not available") == 1
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
@pytest.mark.parametrize("patterns", [[], ["read_text_file"], None])
@pytest.mark.parametrize("direct", [False, True])
async def test_mcp_filter_preserves_filesystem_collision_dispatch(
    tmp_path: Path,
    enabled: bool,
    names: Literal["canonical", "profile"],
    patterns: list[str] | None,
    direct: bool,
) -> None:
    record = tmp_path / "received"
    local_file = tmp_path / "local.txt"
    local_file.write_text("LOCALCONTENT")
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(
            enabled=enabled,
            function_names=names,
            direct_tools=["read_text_file"] if direct else [],
        ),
        mcp=MCPSettings(servers={"recording": _recording_server(record)}),
    )
    agent = McpAgent(
        AgentConfig(
            "filesystem",
            model="passthrough",
            servers=["recording"],
            skills=[],
            tools={} if patterns is None else {"recording": patterns},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        agent.set_filesystem_runtime(
            LocalFilesystemRuntime(agent.logger, working_directory=tmp_path)
        )
        remote_allowed = patterns is None or bool(patterns)
        for name in ("read_text_file", "READ_TEXT_FILE"):
            result = await _model_call(agent, name, {"path": str(local_file)})
            local_advertised = not enabled or direct
            executes = local_advertised or (remote_allowed and name == "read_text_file")
            assert bool(result.is_error) is not executes, result
            if not executes:
                assert str(result.content).count("not available") == 1
                assert "recording__read_text_file" not in str(result.content) or remote_allowed
                assert not record.exists()
            elif remote_allowed:
                assert record.exists()
                record.unlink()
            else:
                assert "LOCALCONTENT" in str(result.content)
                assert "recording__read_text_file" not in str(result.content)
                assert not record.exists()

        result = await _model_call(agent, "recording__read_text_file", {"path": str(local_file)})
        assert bool(result.is_error) is not remote_allowed, result
        if remote_allowed:
            assert record.exists()
            record.unlink()
        else:
            assert str(result.content).count("not available") == 1
            assert not record.exists()

        # The local implementation remains callable from code and by host.
        if enabled and not direct:
            result = await _model_call(
                agent, PTC_TOOL_NAME, {"code": f"await read_text_file(path={str(local_file)!r})"}
            )
            assert not result.is_error and "LOCALCONTENT" in str(result.content)
        result = await agent.call_tool("read_text_file", {"path": str(local_file)})
        assert not result.is_error and "LOCALCONTENT" in str(result.content)
        assert not record.exists()
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("ptc", [False, True])
@pytest.mark.parametrize(
    ("patterns", "permitted"),
    [
        (None, {"allowed/doc", "allowed/item/7", "hidden/doc"}),
        (["recording://allowed/doc"], {"allowed/doc"}),
        (["recording://allowed/*"], {"allowed/doc", "allowed/item/7"}),
        ([], set()),
    ],
)
async def test_resource_filter_constrains_model_reads(
    tmp_path: Path, ptc: bool, patterns: list[str] | None, permitted: set[str]
) -> None:
    record = tmp_path / "received"
    other_record = tmp_path / "other"
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(enabled=ptc),
        mcp=MCPSettings(
            servers={
                "recording": _recording_server(record),
                "other": _recording_server(other_record),
            }
        ),
    )
    agent = McpAgent(
        AgentConfig(
            "resources",
            model="passthrough",
            servers=["recording", "other"],
            skills=[],
            resources={} if patterns is None else {"recording": patterns},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )

    def receipts(path: Path) -> list[str]:
        return path.read_text().splitlines() if path.exists() else []

    async def model_read(arguments: dict[str, str]) -> CallToolResult:
        if not ptc:
            return await _model_call(agent, "get_resource", arguments)
        code = f"await get_resource({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
        return await _model_call(agent, PTC_TOOL_NAME, {"code": code})

    received: list[str] = []
    other_received: list[str] = []
    try:
        await agent.initialize()
        set_harness_tools(agent, True)
        for path in ("allowed/doc", "allowed/item/7", "hidden/doc"):
            uri = f"recording://{path}"
            result = await model_read({"uri": uri, "server_name": "recording"})
            assert bool(result.is_error) is (path not in permitted), result
            if path in permitted:
                received.append(path)
            assert receipts(record) == received

            # An omitted server skips servers whose filter denies the URI.
            result = await model_read({"uri": uri})
            assert not result.is_error, result
            if path in permitted:
                received.append(path)
            else:
                other_received.append(path)
            assert receipts(record) == received
            assert receipts(other_record) == other_received

        result = await model_read({"uri": "internal://fast-agent/agent-cards"})
        assert not result.is_error and "AgentCard" in str(result.content)

        # resources: constrains model reads, not trusted host APIs.
        for server in ("recording", None):
            hidden = await agent.get_resource("recording://hidden/doc", server)
            assert "hidden/doc" in str(hidden.contents)
            received.append("hidden/doc")
        await agent.get_embedded_resources("recording://hidden/doc", "recording")
        received.append("hidden/doc")
        assert receipts(record) == received
    finally:
        await agent.shutdown()


_CANONICAL_FILTER = ["https://example.test/allowed/*", "recording://host/allowed/*"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ptc", [False, True])
@pytest.mark.parametrize("server_name", ["recording", None])
@pytest.mark.parametrize("patterns", [_CANONICAL_FILTER, []])
async def test_resource_filter_checks_the_uri_the_server_receives(
    tmp_path: Path, ptc: bool, server_name: str | None, patterns: list[str]
) -> None:
    record = tmp_path / "received"
    other_record = tmp_path / "other"
    settings = Settings(
        ptc=ProgrammaticToolCallingSettings(enabled=ptc),
        mcp=MCPSettings(
            servers={
                "recording": _recording_server(record),
                "other": _recording_server(other_record),
            }
        ),
    )
    # Every server is filtered, so an omitted server has nowhere to fall back to.
    agent = McpAgent(
        AgentConfig(
            "canonical",
            model="passthrough",
            servers=["recording", "other"],
            skills=[],
            resources={"recording": patterns, "other": patterns},
        ),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )

    async def model_read(uri: str) -> CallToolResult:
        arguments = (
            {"uri": uri} if server_name is None else {"uri": uri, "server_name": server_name}
        )
        if not ptc:
            return await _model_call(agent, "get_resource", arguments)
        code = f"await get_resource({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
        return await _model_call(agent, PTC_TOOL_NAME, {"code": code})

    def receipts() -> list[str]:
        return [
            line
            for path in (record, other_record)
            if path.exists()
            for line in path.read_text().splitlines()
        ]

    try:
        await agent.initialize()
        set_harness_tools(agent, True)
        for uri in (
            "https://example.test/hidden/doc",
            "https://example.test/allowed/../hidden/doc",
            "https://example.test/allowed/%2e%2e/hidden/doc",
            "https://example.test/allowed/%2E./hidden/doc",
            "https://example.test/allowed/..\\hidden/doc",
            "recording://host/hidden/doc",
            "recording://host/allowed/../hidden/doc",
            "recording://host/allowed/%2e%2e/hidden/doc",
        ):
            result = await model_read(uri)
            assert result.is_error, (uri, result)
            assert receipts() == [], uri

        # The allowed decision is made on the same URI that is sent.
        result = await model_read("https://example.test/hidden/../allowed/doc")
        assert bool(result.is_error) is (not patterns), result
        assert receipts() == (["https/allowed/doc"] if patterns else [])
    finally:
        await agent.shutdown()


class _NamespaceOverrideAgent(McpAgent):
    """Overrides get_resource with the AgentProtocol signature (namespace only)."""

    async def get_resource(  # ty: ignore[invalid-method-override]
        self, resource_uri: str, namespace: str | None = None
    ) -> ReadResourceResult:
        return await super().get_resource(resource_uri, namespace=namespace)


@pytest.mark.asyncio
@pytest.mark.parametrize("resources", [{}, {"recording": ["https://example.test/allowed/*"]}])
async def test_model_reads_call_namespace_only_get_resource_override(
    tmp_path: Path, resources: dict[str, list[str]]
) -> None:
    record = tmp_path / "received"
    settings = Settings(mcp=MCPSettings(servers={"recording": _recording_server(record)}))
    agent = _NamespaceOverrideAgent(
        AgentConfig("override", servers=["recording"], skills=[], resources=resources),
        context=Context(config=settings, server_registry=ServerRegistry(settings)),
    )
    try:
        await agent.initialize()
        set_harness_tools(agent, True)
        uri = "https://example.test/allowed/doc"
        for arguments in ({"uri": uri, "server_name": "recording"}, {"uri": uri}):
            result = await _model_call(agent, "get_resource", arguments)
            assert not result.is_error, result
        assert record.read_text().splitlines() == ["https/allowed/doc"] * 2
    finally:
        await agent.shutdown()
