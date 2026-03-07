import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.context import Context
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model
from fast_agent.mcp.mcp_aggregator import NamespacedTool
from fast_agent.skills.registry import SkillRegistry
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay


class CaptureDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.calls: list[dict[str, object]] = []

    async def show_assistant_message(
        self,
        message_text: str | Text | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        self.calls.append(
            {
                "bottom_items": bottom_items,
                "highlight_index": highlight_index,
                "additional_message": additional_message,
            }
        )


def _make_agent_config() -> AgentConfig:
    return AgentConfig(name="test-agent", instruction="do things", servers=[])


def _create_skill(directory, name: str, description: str = "desc") -> None:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n",
        encoding="utf-8",
    )


class StubLLM:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.instruction = ""
        self.default_request_params = RequestParams()


def _stub_llm_factory(model_name: str):
    def _factory(**_: object) -> StubLLM:
        return StubLLM(model_name)

    return _factory


@pytest.mark.asyncio
async def test_local_tools_listed_and_callable() -> None:
    calls: list[dict[str, Any]] = []

    def sample_tool(video_id: str) -> str:
        calls.append({"video_id": video_id})
        return f"transcript for {video_id}"

    config = _make_agent_config()
    context = Context()

    class LocalToolAgent(McpAgent):
        def __init__(self) -> None:
            super().__init__(
                config=config,
                connection_persistence=False,
                context=context,
                tools=[sample_tool],
            )

    agent = LocalToolAgent()

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "sample_tool" in tool_names

    result: CallToolResult = await agent.call_tool("sample_tool", {"video_id": "1234"})
    assert not result.isError
    assert calls == [{"video_id": "1234"}]
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "transcript for 1234"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_card_tools_label_highlighted_on_use() -> None:
    def sample_tool(video_id: str) -> str:
        return f"transcript for {video_id}"

    config = _make_agent_config()
    context = Context()

    class LocalToolAgent(McpAgent):
        def __init__(self) -> None:
            super().__init__(
                config=config,
                connection_persistence=False,
                context=context,
                tools=[sample_tool],
            )

    agent = LocalToolAgent()
    capture_display = CaptureDisplay()
    agent.display = capture_display

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(
                name="sample_tool",
                arguments={"video_id": "1234"},
            )
        )
    }
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="response")],
        tool_calls=tool_calls,
    )

    await agent.show_assistant_message(message)

    assert capture_display.calls
    call = capture_display.calls[-1]
    assert call["bottom_items"] == ["card_tools"]
    assert call["highlight_index"] == 0

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_skills_tool_listed_and_highlighted(tmp_path) -> None:
    skills_root = tmp_path / "skills"
    _create_skill(skills_root, "alpha")

    manifests = SkillRegistry.load_directory(skills_root)
    context = Context()
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        skills=skills_root,
    )
    config.skill_manifests = manifests

    agent = McpAgent(config=config, context=context)
    capture_display = CaptureDisplay()
    agent.display = capture_display

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(
                name="read_skill",
                arguments={"path": str(manifests[0].path)},
            )
        )
    }
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="response")],
        tool_calls=tool_calls,
    )

    await agent.show_assistant_message(message)

    assert capture_display.calls
    call = capture_display.calls[-1]
    assert call["bottom_items"] is not None
    assert "skill" in call["bottom_items"]
    assert call["highlight_index"] == call["bottom_items"].index("skill")

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_output_limit_refreshes_after_llm_attach() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

    await agent.attach_llm(_stub_llm_factory("claude-opus-4-6"), model="opus")

    assert shell_runtime.output_byte_limit == calculate_terminal_output_limit_for_model(
        "claude-opus-4-6"
    )

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_local_read_text_file_when_enabled(tmp_path: Path) -> None:
    test_file = tmp_path / "notes.txt"
    test_file.write_text("one\ntwo\nthree\n", encoding="utf-8")

    settings = Settings(shell_execution=ShellSettings(enable_read_text_file=True))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "execute" in tool_names
    assert "read_text_file" in tool_names
    assert "write_text_file" in tool_names

    result = await agent.call_tool(
        "read_text_file",
        {"path": str(test_file), "line": 2, "limit": 1},
    )
    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "two"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_local_write_text_file_when_enabled(tmp_path: Path) -> None:
    output_file = tmp_path / "nested" / "notes.txt"

    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names

    result = await agent.call_tool(
        "write_text_file",
        {"path": str(output_file), "content": "hello from write tool"},
    )
    assert result.isError is False
    assert output_file.read_text(encoding="utf-8") == "hello from write tool"
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Successfully wrote" in result.content[0].text

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_apply_patch_when_model_prefers_it(tmp_path: Path) -> None:
    target_file = tmp_path / "notes.txt"
    target_file.write_text("one\ntwo\n", encoding="utf-8")

    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="gpt-5.4",
        cwd=tmp_path,
    )
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "apply_patch" in tool_names
    assert "write_text_file" not in tool_names

    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: notes.txt\n"
        "@@\n"
        "-one\n"
        "+ONE\n"
        " two\n"
        "*** End Patch\n"
    )
    result = await agent.call_tool("apply_patch", {"input": patch_text})

    assert result.isError is False
    assert target_file.read_text(encoding="utf-8") == "ONE\ntwo\n"
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Success. Updated the following files:" in result.content[0].text

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_read_text_file_option_requires_shell_runtime() -> None:
    settings = Settings(shell_execution=ShellSettings(enable_read_text_file=True))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" not in tool_names
    assert "write_text_file" not in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_read_text_file_option_is_enabled_by_default() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "execute" in tool_names
    assert "read_text_file" in tool_names
    assert "write_text_file" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["codexplan", "gpt-5.4", "responses.gpt-5.4"])
async def test_write_text_file_auto_mode_prefers_apply_patch_for_codex_family_models(
    model_name: str,
) -> None:
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model=model_name,
    )
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in tool_names
    assert "write_text_file" not in tool_names
    assert "apply_patch" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_write_text_file_auto_mode_remains_enabled_for_qwen35() -> None:
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="qwen35",
    )
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_write_text_file_auto_mode_uses_context_default_model_when_agent_model_missing() -> (
    None
):
    settings = Settings(default_model="codexplan")
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" not in tool_names
    assert "apply_patch" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_apply_patch_mode_explicitly_enables_tool() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="apply_patch"))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="qwen35",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "apply_patch" in tool_names
    assert "write_text_file" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_write_text_file_mode_on_enables_tool_for_codex_models() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="codexplan",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_write_text_file_mode_off_disables_tool_even_for_non_codex_models() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="off"))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="qwen35",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_write_text_file_option_can_be_disabled() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="off"))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in tool_names
    assert "write_text_file" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_skills_fallback_to_read_skill_when_local_read_text_file_disabled(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    _create_skill(skills_root, "alpha")
    manifests = SkillRegistry.load_directory(skills_root)

    settings = Settings(
        shell_execution=ShellSettings(enable_read_text_file=False, write_text_file_mode="on")
    )
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        skills=skills_root,
    )
    config.skill_manifests = manifests
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names
    assert "read_text_file" not in tool_names
    assert "read_skill" in tool_names
    assert agent.skill_read_tool_name == "read_skill"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_acp_filesystem_runtime_injection_replaces_local_runtime_tools() -> None:
    class ReadOnlyACPFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="ACP read tool",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ]

        async def read_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(content=[TextContent(type="text", text="acp")], isError=False)

        async def write_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="write unavailable")],
                isError=True,
            )

        def metadata(self) -> dict[str, Any]:
            return {"variant": "acp_filesystem", "tools": ["read_text_file"]}

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    initial_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in initial_tool_names
    assert "write_text_file" in initial_tool_names

    acp_runtime = ReadOnlyACPFilesystemRuntime()
    agent.set_filesystem_runtime(cast("Any", acp_runtime))

    replaced_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in replaced_tool_names
    assert "write_text_file" not in replaced_tool_names

    result = await agent.call_tool("read_text_file", {"path": "/tmp/anything"})
    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "acp"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_unprefixed_read_text_file_routes_to_namespaced_mcp_when_local_fs_available() -> None:
    class RecordingFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="Local read tool",
                    inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ]
            self.read_calls: list[dict[str, Any] | None] = []

        async def read_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del tool_use_id
            self.read_calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="local")], isError=False)

        async def write_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="write unsupported")],
                isError=True,
            )

        def metadata(self) -> dict[str, Any]:
            return {"variant": "local_filesystem"}

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())

    mcp_tool = Tool(
        name="read_text_file",
        description="MCP read tool",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    namespaced_tool = NamespacedTool(
        tool=mcp_tool,
        server_name="docs",
        namespaced_tool_name="docs__read_text_file",
    )
    agent._aggregator._namespaced_tool_map = {namespaced_tool.namespaced_tool_name: namespaced_tool}
    agent._aggregator._server_to_tool_map = {namespaced_tool.server_name: [namespaced_tool]}

    mcp_calls: list[str] = []

    async def fake_list_tools() -> ListToolsResult:
        return ListToolsResult(
            tools=[
                mcp_tool.model_copy(
                    deep=True, update={"name": namespaced_tool.namespaced_tool_name}
                )
            ]
        )

    async def fake_call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: Any | None = None,
    ) -> CallToolResult:
        del arguments, tool_use_id, request_tool_handler
        mcp_calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="mcp")], isError=False)

    async def fake_get_skybridge_config(server_name: str) -> None:
        del server_name
        return None

    agent._aggregator.list_tools = cast("Any", fake_list_tools)
    agent._aggregator.call_tool = cast("Any", fake_call_tool)
    agent._aggregator.get_skybridge_config = cast("Any", fake_get_skybridge_config)

    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="use the tool")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="read_text_file",
                    arguments={"path": "/tmp/example.txt"},
                )
            )
        },
    )
    result = await agent.run_tools(request)

    assert mcp_calls == ["docs__read_text_file"]
    filesystem_runtime = cast("RecordingFilesystemRuntime", agent._filesystem_runtime)
    assert filesystem_runtime.read_calls == []
    assert result.tool_results is not None
    assert "call-1" in result.tool_results
    tool_result = result.tool_results["call-1"]
    assert tool_result.content is not None
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "mcp"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_unprefixed_write_text_file_routes_to_namespaced_mcp_when_local_fs_available() -> (
    None
):
    class RecordingFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="write_text_file",
                    description="Local write tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                )
            ]
            self.write_calls: list[dict[str, Any] | None] = []

        async def read_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="read unsupported")],
                isError=True,
            )

        async def write_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del tool_use_id
            self.write_calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="local")], isError=False)

        def metadata(self) -> dict[str, Any]:
            return {"variant": "local_filesystem"}

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())

    mcp_tool = Tool(
        name="write_text_file",
        description="MCP write tool",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        },
    )
    namespaced_tool = NamespacedTool(
        tool=mcp_tool,
        server_name="docs",
        namespaced_tool_name="docs__write_text_file",
    )
    agent._aggregator._namespaced_tool_map = {namespaced_tool.namespaced_tool_name: namespaced_tool}
    agent._aggregator._server_to_tool_map = {namespaced_tool.server_name: [namespaced_tool]}

    mcp_calls: list[str] = []

    async def fake_list_tools() -> ListToolsResult:
        return ListToolsResult(
            tools=[
                mcp_tool.model_copy(
                    deep=True, update={"name": namespaced_tool.namespaced_tool_name}
                )
            ]
        )

    async def fake_call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: Any | None = None,
    ) -> CallToolResult:
        del arguments, tool_use_id, request_tool_handler
        mcp_calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="mcp")], isError=False)

    async def fake_get_skybridge_config(server_name: str) -> None:
        del server_name
        return None

    agent._aggregator.list_tools = cast("Any", fake_list_tools)
    agent._aggregator.call_tool = cast("Any", fake_call_tool)
    agent._aggregator.get_skybridge_config = cast("Any", fake_get_skybridge_config)

    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="use the tool")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="write_text_file",
                    arguments={"path": "/tmp/example.txt", "content": "hello"},
                )
            )
        },
    )
    result = await agent.run_tools(request)

    assert mcp_calls == ["docs__write_text_file"]
    filesystem_runtime = cast("RecordingFilesystemRuntime", agent._filesystem_runtime)
    assert filesystem_runtime.write_calls == []
    assert result.tool_results is not None
    assert "call-1" in result.tool_results
    tool_result = result.tool_results["call-1"]
    assert tool_result.content is not None
    assert isinstance(tool_result.content[0], TextContent)
    assert tool_result.content[0].text == "mcp"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_tool_use_turn_hides_bottom_bar_and_mentions_shell_access() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    capture_display = CaptureDisplay()
    agent.display = capture_display

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(
                name="execute",
                arguments={"command": "pwd"},
            )
        )
    }
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls=tool_calls,
        stop_reason=LlmStopReason.TOOL_USE,
    )

    await agent.show_assistant_message(message)

    assert capture_display.calls
    call = capture_display.calls[-1]
    assert call["bottom_items"] is None
    assert call["highlight_index"] is None
    additional = call["additional_message"]
    assert isinstance(additional, Text)
    assert "requested shell access" in additional.plain

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_read_text_file_tool_use_turn_hides_bottom_bar_without_extra_message() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    capture_display = CaptureDisplay()
    agent.display = capture_display

    tool_calls = {
        "1": CallToolRequest(
            params=CallToolRequestParams(
                name="read_text_file",
                arguments={"path": "/tmp/example.txt", "line": 93, "limit": 30},
            )
        )
    }
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls=tool_calls,
        stop_reason=LlmStopReason.TOOL_USE,
    )

    await agent.show_assistant_message(message)

    assert capture_display.calls
    call = capture_display.calls[-1]
    assert call["bottom_items"] is None
    assert call["highlight_index"] is None
    assert call["additional_message"] is None

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_output_limit_override_is_preserved_after_llm_attach() -> None:
    settings = Settings(shell_execution=ShellSettings(output_byte_limit=9000))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == 9000

    await agent.attach_llm(_stub_llm_factory("claude-opus-4-6"), model="opus")

    assert shell_runtime.output_byte_limit == 9000

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_startup_warns_when_configured_cwd_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing-shell-cwd"
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        cwd=missing_dir,
    )
    agent = McpAgent(config=config, context=Context())

    assert any("shell cwd that does not exist" in warning for warning in agent.warnings)

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_startup_warns_when_configured_cwd_is_file(tmp_path: Path) -> None:
    file_path = tmp_path / "shell-cwd-file.txt"
    file_path.write_text("x", encoding="utf-8")
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        cwd=file_path,
    )
    agent = McpAgent(config=config, context=Context())

    assert any("shell cwd that is not a directory" in warning for warning in agent.warnings)

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_call_forwards_parallel_display_flags() -> None:
    class RecordingShellRuntime:
        def __init__(self) -> None:
            self.tool = Tool(
                name="execute",
                description="Run shell command",
                inputSchema={"type": "object", "properties": {}},
            )
            self.calls: list[dict[str, object]] = []

        def metadata(self, command: str | None) -> dict[str, object]:
            return {
                "variant": "shell",
                "command": command,
                "shell_name": "shell",
                "shell_path": "/bin/bash",
            }

        async def execute(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
            *,
            show_tool_call_id: bool = False,
            defer_display_to_tool_result: bool = False,
        ):
            self.calls.append(
                {
                    "arguments": arguments,
                    "tool_use_id": tool_use_id,
                    "show_tool_call_id": show_tool_call_id,
                    "defer_display_to_tool_result": defer_display_to_tool_result,
                }
            )
            return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    shell_runtime = RecordingShellRuntime()
    agent._shell_runtime = cast("Any", shell_runtime)
    agent._shell_runtime_enabled = True

    await agent.call_tool("execute", {"command": "echo hello"}, "call-1")
    assert shell_runtime.calls[-1]["show_tool_call_id"] is False
    assert shell_runtime.calls[-1]["defer_display_to_tool_result"] is False

    agent._show_shell_tool_call_id = True
    agent._defer_shell_display_to_tool_result = True
    await agent.call_tool("execute", {"command": "echo hello"}, "call-2")
    assert shell_runtime.calls[-1]["show_tool_call_id"] is True
    assert shell_runtime.calls[-1]["defer_display_to_tool_result"] is True

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_parallel_shell_results_display_in_tool_call_order() -> None:
    class RecordingShellRuntime:
        def __init__(self) -> None:
            self.tool = Tool(
                name="execute",
                description="Run shell command",
                inputSchema={"type": "object", "properties": {}},
            )

        def metadata(self, command: str | None) -> dict[str, object]:
            return {
                "variant": "shell",
                "command": command,
                "shell_name": "shell",
                "shell_path": "/bin/bash",
            }

        async def execute(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
            *,
            show_tool_call_id: bool = False,
            defer_display_to_tool_result: bool = False,
        ) -> CallToolResult:
            command = str((arguments or {}).get("command", ""))
            if command == "first":
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)

            result = CallToolResult(
                content=[TextContent(type="text", text=f"{command}\nprocess exit code was 0")],
                isError=False,
            )
            setattr(result, "_suppress_display", not defer_display_to_tool_result)
            return result

    class RecordingDisplay:
        def __init__(self) -> None:
            self.result_ids: list[str | None] = []
            self.result_text: list[str] = []

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            return None

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            self.result_ids.append(kwargs.get("tool_call_id"))
            result = kwargs.get("result")
            if isinstance(result, CallToolResult) and result.content:
                block = result.content[0]
                if isinstance(block, TextContent):
                    self.result_text.append(block.text)

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    agent._shell_runtime = cast("Any", RecordingShellRuntime())
    agent._shell_runtime_enabled = True
    recording_display = RecordingDisplay()
    agent.display = cast("Any", recording_display)

    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="run tools")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(name="execute", arguments={"command": "first"})
            ),
            "call-2": CallToolRequest(
                params=CallToolRequestParams(name="execute", arguments={"command": "second"})
            ),
        },
    )

    await agent.run_tools(request)

    # Even though "second" completes sooner, display order should follow tool-call order.
    assert recording_display.result_ids == ["call-1", "call-2"]
    assert recording_display.result_text[0].startswith("first")
    assert recording_display.result_text[1].startswith("second")

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_read_text_file_tool_call_header_is_suppressed() -> None:
    class RecordingFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="Read file",
                    inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ]

        async def read_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="line-1\nline-2")],
                isError=False,
            )

        async def write_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="unsupported")],
                isError=True,
            )

        def metadata(self) -> dict[str, Any]:
            return {"variant": "local_filesystem"}

    class RecordingDisplay:
        def __init__(self) -> None:
            self.tool_call_count = 0
            self.result_count = 0

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.tool_call_count += 1

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.result_count += 1

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())
    recording_display = RecordingDisplay()
    agent.display = cast("Any", recording_display)

    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="read a file")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="read_text_file",
                    arguments={"path": "/tmp/example.txt", "line": 93, "limit": 30},
                )
            )
        },
    )

    await agent.run_tools(request)

    assert recording_display.tool_call_count == 0
    assert recording_display.result_count == 1

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_parallel_read_text_file_results_use_file_read_label_without_ids() -> None:
    class RecordingFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="Read file",
                    inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
                )
            ]

        async def read_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="line-1\nline-2")],
                isError=False,
            )

        async def write_text_file(
            self,
            arguments: dict[str, Any] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="unsupported")],
                isError=True,
            )

        def metadata(self) -> dict[str, Any]:
            return {"variant": "local_filesystem"}

    class RecordingDisplay:
        def __init__(self) -> None:
            self.result_tool_call_ids: list[str | None] = []
            self.result_type_labels: list[str | None] = []
            self.results: list[CallToolResult] = []

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            return None

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            self.results.append(cast("CallToolResult", kwargs["result"]))
            self.result_tool_call_ids.append(cast("str | None", kwargs.get("tool_call_id")))
            self.result_type_labels.append(cast("str | None", kwargs.get("type_label")))

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())
    recording_display = RecordingDisplay()
    agent.display = cast("Any", recording_display)

    request = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="read two files")],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="read_text_file",
                    arguments={"path": "/tmp/example-1.txt", "line": 1, "limit": 20},
                )
            ),
            "call-2": CallToolRequest(
                params=CallToolRequestParams(
                    name="read_text_file",
                    arguments={"path": "/tmp/example-2.txt", "line": 1, "limit": 20},
                )
            ),
        },
    )

    await agent.run_tools(request)

    assert recording_display.result_type_labels == ["file read", "file read"]
    assert recording_display.result_tool_call_ids == [None, None]
    assert [getattr(result, "read_text_file_line", None) for result in recording_display.results] == [
        1,
        1,
    ]
    assert [getattr(result, "read_text_file_limit", None) for result in recording_display.results] == [
        20,
        20,
    ]

    await agent._aggregator.close()
