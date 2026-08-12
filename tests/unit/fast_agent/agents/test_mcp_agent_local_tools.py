import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest
from fastmcp.tools import FunctionTool, ToolResult
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import (
    McpAgent,
    ShellEditToolFlags,
    ShellEditToolMode,
    _effective_configured_servers,
)
from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.terminal_output_limits import (
    calculate_terminal_output_limit_for_model,
)
from fast_agent.mcp.mcp_aggregator import NamespacedTool
from fast_agent.mcp.tool_result_metadata import (
    tool_result_display_metadata,
    update_tool_result_display_metadata,
)
from fast_agent.skills.registry import SkillRegistry
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.tool_display import ToolCallDisplayRequest, ToolResultDisplayRequest
from fast_agent.utils.tool_names import (
    BASH_TOOL_NAME,
    EXECUTE_TOOL_NAME,
    GROK_SHELL_TOOL_NAME,
    LUNA_EXEC_TOOL_NAME,
    PROCESS_TOOL_NAME,
)


def test_runtime_mcp_overlay_does_not_mutate_agent_config() -> None:
    config = AgentConfig(name="main", servers=["configured"])
    context = Context(
        runtime_mcp_server_names={
            "main": ("configured", "startup"),
        }
    )

    assert config.servers == ["configured"]
    assert _effective_configured_servers(config, context) == ("configured", "startup")


class _DisplayCall(TypedDict):
    bottom_items: list[str] | None
    highlight_indexes: list[int] | None
    additional_message: Text | None


class CaptureDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.calls: list[_DisplayCall] = []

    async def show_assistant_message(
        self,
        message_text: str | Text | PromptMessageExtended,
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        pre_content=None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
        show_reprint_banner: bool = False,
    ) -> None:
        del (
            message_text,
            max_item_length,
            name,
            model,
            pre_content,
            render_markdown,
            show_hook_indicator,
            show_reprint_banner,
        )
        self.calls.append(
            {
                "bottom_items": bottom_items,
                "highlight_indexes": highlight_indexes,
                "additional_message": additional_message,
            }
        )


def _bottom_items(call: _DisplayCall) -> list[str]:
    bottom_items = call["bottom_items"]
    assert bottom_items is not None
    return bottom_items


def test_shell_edit_tool_flags_follow_mode_contract() -> None:
    assert ShellEditToolFlags.from_mode(ShellEditToolMode.WRITE_TEXT_FILE) == ShellEditToolFlags(
        write_text_file=True,
        apply_patch=False,
        edit_file=True,
    )
    assert ShellEditToolFlags.from_mode(ShellEditToolMode.EDIT_FILE) == ShellEditToolFlags(
        write_text_file=False,
        apply_patch=False,
        edit_file=True,
    )
    assert ShellEditToolFlags.from_mode(ShellEditToolMode.APPLY_PATCH) == ShellEditToolFlags(
        write_text_file=False,
        apply_patch=True,
        edit_file=False,
    )
    assert ShellEditToolFlags.from_mode(ShellEditToolMode.OFF) == ShellEditToolFlags(
        write_text_file=False,
        apply_patch=False,
        edit_file=False,
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
        self.resolved_model = SimpleNamespace(
            max_output_tokens=ModelDatabase.get_max_output_tokens(model_name),
            model_params=ModelDatabase.get_model_params(model_name),
        )
        self.instruction = ""
        self.default_request_params = RequestParams()

    @property
    def model_info(self) -> ModelInfo | None:
        return ModelInfo.from_name(self.model_name)


class StubLLMWithoutResolvedModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.instruction = ""
        self.default_request_params = RequestParams()

    @property
    def model_info(self) -> ModelInfo | None:
        return ModelInfo.from_name(self.model_name)


def _stub_llm_factory(model_name: str):
    def _factory(**_: object) -> StubLLM:
        return StubLLM(model_name)

    return _factory


@pytest.mark.asyncio
async def test_local_tools_listed_and_callable() -> None:
    calls: list[dict[str, str]] = []

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
    assert not result.is_error
    assert calls == [{"video_id": "1234"}]
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "transcript for 1234"
    assert result.structured_content is None

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_model_query_poll_period_updates_shell_runtime_on_model_switch() -> None:
    agent = McpAgent(
        config=AgentConfig(name="poll-period", shell=True),
        context=Context(config=Settings()),
    )

    await agent.attach_llm(ModelFactory.create_factory("silent?poll_period=90"))

    assert agent._shell_runtime is not None
    assert agent._shell_runtime._process_poll_default_wait_seconds == 90
    assert agent._shell_runtime._minimal_process_wait_seconds() == 90

    await agent.set_model("silent?poll_period=120")

    assert agent._shell_runtime._process_poll_default_wait_seconds == 120
    assert agent._shell_runtime._minimal_process_wait_seconds() == 120
    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_model_query_poll_period_above_operator_ceiling_rejects_switch_atomically() -> None:
    agent = McpAgent(
        config=AgentConfig(
            name="poll-period-limit",
            model="silent?poll_period=30",
            shell=True,
        ),
        context=Context(
            config=Settings(shell_execution=ShellSettings(process_poll_max_wait_seconds=60))
        ),
    )
    await agent.attach_llm(ModelFactory.create_factory("silent?poll_period=30"))
    previous_llm = agent.llm

    with pytest.raises(
        ModelConfigError,
        match=("poll_period=90 exceeds shell_execution.process_poll_max_wait_seconds=60"),
    ):
        await agent.set_model("silent?poll_period=90")

    assert agent.llm is previous_llm
    assert agent.config.model == "silent?poll_period=30"
    assert agent._shell_runtime is not None
    assert agent._shell_runtime._process_poll_default_wait_seconds == 30
    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_catalog_poll_default_remains_capped_by_operator_ceiling() -> None:
    agent = McpAgent(
        config=AgentConfig(name="catalog-poll-limit", shell=True),
        context=Context(
            config=Settings(shell_execution=ShellSettings(process_poll_max_wait_seconds=60))
        ),
    )

    await agent.attach_llm(_stub_llm_factory("grok-4.5"))

    assert agent._shell_runtime is not None
    assert agent._shell_runtime._process_poll_default_wait_seconds == 60
    assert agent._shell_runtime._minimal_process_wait_seconds() == 60
    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_plain_dict_tool_suppresses_structured_content() -> None:
    def summarize() -> dict[str, str]:
        return {"status": "ok"}

    agent = McpAgent(
        config=_make_agent_config(),
        connection_persistence=False,
        context=Context(),
        tools=[summarize],
    )

    result = await agent.call_tool("summarize", {})

    assert result.is_error is False
    assert result.structured_content is None
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == '{"status":"ok"}'

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_explicit_function_tool_preserves_native_structured_content() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    agent = McpAgent(
        config=_make_agent_config(),
        connection_persistence=False,
        context=Context(),
        tools=[FunctionTool.from_function(add)],
    )

    result = await agent.call_tool("add", {"a": 2, "b": 3})

    assert result.is_error is False
    assert result.structured_content == {"result": 5}

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_tool_result_preserves_explicit_structured_content() -> None:
    def summarize() -> ToolResult:
        return ToolResult(
            content={"status": "ok"},
            structured_content={"status": "ok"},
        )

    agent = McpAgent(
        config=_make_agent_config(),
        connection_persistence=False,
        context=Context(),
        tools=[summarize],
    )

    result = await agent.call_tool("summarize", {})

    assert result.is_error is False
    assert result.structured_content == {"status": "ok"}
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == '{"status":"ok"}'

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
    assert call["highlight_indexes"] == [0]

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_and_card_tools_are_both_highlighted() -> None:
    def lsp_diagnostics(file_path: str) -> str:
        return file_path

    agent = McpAgent(
        config=AgentConfig(
            name="test-agent",
            instruction="do things",
            servers=[],
            shell=True,
        ),
        connection_persistence=False,
        context=Context(),
        tools=[lsp_diagnostics],
    )
    capture_display = CaptureDisplay()
    agent.display = capture_display

    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="response")],
        tool_calls={
            "shell": CallToolRequest(
                params=CallToolRequestParams(name="bash", arguments={"command": "pwd"})
            ),
            "lsp": CallToolRequest(
                params=CallToolRequestParams(
                    name="lsp_diagnostics",
                    arguments={"file_path": "app.py"},
                )
            ),
        },
    )

    await agent.show_assistant_message(message)

    call = capture_display.calls[-1]
    bottom_items = _bottom_items(call)
    assert call["highlight_indexes"] == [
        0,
        bottom_items.index("card_tools"),
    ]

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
    bottom_items = _bottom_items(call)
    assert "skill" in bottom_items
    assert call["highlight_indexes"] == [bottom_items.index("skill")]

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
async def test_shell_stream_metadata_is_available_before_arguments_complete() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    shell_tool = shell_runtime.tool
    assert shell_tool is not None

    metadata = agent.resolve_stream_tool_metadata(shell_tool.name)

    assert metadata is not None
    assert metadata["variant"] == "shell"
    assert metadata["shell_name"]
    assert metadata["command"] is None

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_process_stream_metadata_is_not_rendered_as_shell_code() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())
    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    shell_tool = shell_runtime.tool
    assert shell_tool is not None
    process_tool = next(tool for tool in shell_runtime.tools if tool.name != shell_tool.name)

    metadata = agent.resolve_stream_tool_metadata(process_tool.name)

    assert metadata is not None
    assert metadata["variant"] == "shell_process"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_attach_media_auto_enables_after_anthropic_llm_attach() -> None:
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="sonnet",
    )
    agent = McpAgent(config=config, context=Context())

    initial_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "attach_media" not in initial_tool_names

    await agent.attach_llm(_stub_llm_factory("claude-sonnet-5"), model="sonnet")

    attached_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "attach_media" in attached_tool_names
    assert "write_text_file" not in attached_tool_names
    assert "edit_file" in attached_tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_output_limit_falls_back_when_llm_has_no_resolved_model() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None

    agent._on_llm_attached(cast("Any", StubLLMWithoutResolvedModel("gpt-4.1")))

    assert shell_runtime.output_byte_limit == calculate_terminal_output_limit_for_model("gpt-4.1")

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_local_read_text_file_when_enabled(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "notes.txt"
    test_file.write_text("one\ntwo\nthree\n", encoding="utf-8")

    settings = Settings(shell_execution=ShellSettings(enable_read_text_file=True))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "bash" in tool_names
    assert "process" in tool_names
    assert "read_text_file" in tool_names
    assert "write_text_file" in tool_names
    assert "edit_file" in tool_names

    result = await agent.call_tool(
        "read_text_file",
        {"path": str(test_file), "line": 2, "limit": 1},
    )
    assert result.is_error is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "two"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_local_write_text_file_when_enabled(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "nested" / "notes.txt"

    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    result = await agent.call_tool(
        "write_text_file",
        {"path": str(output_file), "content": "hello from write tool"},
    )
    assert result.is_error is False
    assert output_file.read_text(encoding="utf-8") == "hello from write tool"
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Successfully wrote" in result.content[0].text

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_call_edit_file_when_local_filesystem_runtime_is_enabled(
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "notes.txt"
    target_file.write_text("hello world\n", encoding="utf-8")

    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        cwd=tmp_path,
    )
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    result = await agent.call_tool(
        "edit_file",
        {
            "path": "notes.txt",
            "old_string": "world",
            "new_string": "there",
        },
    )

    assert result.is_error is False
    assert target_file.read_text(encoding="utf-8") == "hello there\n"
    assert result.structured_content is not None
    assert result.structured_content["success"] is True

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_filesystem_edit_tools_report_completion_to_tool_handler(
    tmp_path: Path,
) -> None:
    class RecordingToolHandler:
        def __init__(self) -> None:
            self.starts: list[tuple[str, str, dict[str, object] | None, str | None]] = []
            self.completions: list[tuple[str, bool, str | None]] = []

        async def on_tool_start(
            self,
            tool_name: str,
            server_name: str,
            arguments: dict | None,
            tool_use_id: str | None = None,
        ) -> str:
            self.starts.append((tool_name, server_name, arguments, tool_use_id))
            return f"call-{len(self.starts)}"

        async def on_tool_progress(
            self,
            tool_call_id: str,
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            del tool_call_id, progress, total, message

        async def on_tool_complete(
            self,
            tool_call_id: str,
            success: bool,
            content: list[object] | None,
            error: str | None,
        ) -> None:
            del content
            self.completions.append((tool_call_id, success, error))

        async def on_tool_permission_denied(
            self,
            tool_name: str,
            server_name: str,
            tool_use_id: str | None,
            error: str | None = None,
        ) -> None:
            del tool_name, server_name, tool_use_id, error

        async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
            del tool_use_id
            return None

        async def ensure_tool_call_exists(
            self,
            tool_use_id: str,
            tool_name: str,
            server_name: str,
            arguments: dict | None = None,
        ) -> str:
            del tool_use_id, tool_name, server_name, arguments
            return "ensured"

    edit_target = tmp_path / "edit.txt"
    edit_target.write_text("hello world\n", encoding="utf-8")
    patch_target = tmp_path / "patch.txt"
    patch_target.write_text("one\ntwo\n", encoding="utf-8")

    edit_agent = McpAgent(
        config=AgentConfig(
            name="test", instruction="Instruction", servers=[], shell=True, cwd=tmp_path
        ),
        context=Context(),
    )
    patch_agent = McpAgent(
        config=AgentConfig(
            name="test",
            instruction="Instruction",
            servers=[],
            shell=True,
            model="gpt-5.4",
            cwd=tmp_path,
        ),
        context=Context(),
    )
    handler = RecordingToolHandler()
    params = RequestParams(tool_execution_handler=cast("Any", handler))

    edit_result = await edit_agent.call_tool(
        "edit_file",
        {"path": "edit.txt", "old_string": "world", "new_string": "there"},
        tool_use_id="edit-use-1",
        request_params=params,
    )
    patch_result = await patch_agent.call_tool(
        "apply_patch",
        {
            "input": (
                "*** Begin Patch\n*** Update File: patch.txt\n@@\n-one\n+ONE\n two\n*** End Patch\n"
            )
        },
        tool_use_id="patch-use-1",
        request_params=params,
    )

    assert edit_result.is_error is False
    assert patch_result.is_error is False
    assert handler.starts == [
        (
            "edit_file",
            "local",
            {"path": "edit.txt", "old_string": "world", "new_string": "there"},
            "edit-use-1",
        ),
        (
            "apply_patch",
            "local",
            {
                "input": (
                    "*** Begin Patch\n"
                    "*** Update File: patch.txt\n"
                    "@@\n"
                    "-one\n"
                    "+ONE\n"
                    " two\n"
                    "*** End Patch\n"
                )
            },
            "patch-use-1",
        ),
    ]
    assert handler.completions == [
        ("call-1", True, None),
        ("call-2", True, None),
    ]

    await edit_agent._aggregator.close()
    await patch_agent._aggregator.close()


@pytest.mark.asyncio
async def test_shell_can_include_apply_patch_when_model_prefers_it(
    tmp_path: Path,
) -> None:
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
    assert "edit_file" not in tool_names

    patch_text = (
        "*** Begin Patch\n*** Update File: notes.txt\n@@\n-one\n+ONE\n two\n*** End Patch\n"
    )
    result = await agent.call_tool("apply_patch", {"input": patch_text})

    assert result.is_error is False
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
    assert "edit_file" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_read_text_file_option_is_enabled_by_default() -> None:
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "bash" in tool_names
    assert "process" in tool_names
    assert "read_text_file" in tool_names
    assert "write_text_file" in tool_names
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["codexplan", "gpt-5.2", "gpt-5.4", "responses.gpt-5.4"],
)
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
    assert "edit_file" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-5", "gpt-5.0", "gpt-5.1"])
async def test_write_text_file_auto_mode_keeps_write_and_edit_for_pre_52_gpt5_models(
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
    assert "write_text_file" in tool_names
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "sonnet",
        "claude-3-5-haiku",
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-fable-5",
        "anthropic-vertex.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
async def test_write_text_file_auto_mode_uses_edit_only_for_anthropic_series_models(
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
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek.deepseek-v4-flash",
        "deepseek.deepseek-v4-pro",
        "hf.deepseek-ai/DeepSeek-V4-Flash-0731?reasoning=max",
    ],
)
async def test_deepseek_uses_catalog_driven_shell_and_writer_editor_contract(
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

    tools = {tool.name: tool for tool in (await agent.list_tools()).tools}
    assert "Shell" in tools
    assert "bash" not in tools
    assert "write_text_file" in tools
    assert "edit_file" in tools
    assert "apply_patch" not in tools
    assert set(tools["Shell"].input_schema["properties"]) == {
        "command",
        "run_in_background",
    }
    assert tools["Shell"].input_schema["required"] == ["command"]

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_explicit_edit_mode_overrides_deepseek_writer_editor_default() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="edit_file"))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="deepseek.deepseek-v4-flash",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" not in tool_names
    assert "edit_file" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_attaching_deepseek_rebuilds_shell_and_file_tool_contract() -> None:
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="qwen35",
    )
    agent = McpAgent(config=config, context=Context())

    initial = {tool.name for tool in (await agent.list_tools()).tools}
    assert "bash" in initial
    assert "write_text_file" in initial

    agent._on_llm_attached(cast("Any", StubLLM("deepseek-v4-flash")))

    attached = {tool.name for tool in (await agent.list_tools()).tools}
    assert "Shell" in attached
    assert "bash" not in attached
    assert "write_text_file" in attached
    assert "edit_file" in attached

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_explicit_edit_file_mode_hides_writer_for_other_models() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="edit_file"))
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
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

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
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

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
    assert "edit_file" not in tool_names

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
    assert "edit_file" not in tool_names

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
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_write_text_file_mode_on_restores_tool_for_anthropic_models() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="on"))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="sonnet",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "write_text_file" in tool_names
    assert "edit_file" in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_default_shell_profile_exposes_facades_with_file_tools() -> None:
    settings = Settings()
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="sonnet",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "bash" in tool_names
    assert "process" in tool_names
    assert "execute" not in tool_names
    assert "poll_process" not in tool_names
    assert "terminate_process" not in tool_names
    assert "read_text_file" in tool_names
    assert "edit_file" in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expects_extended_guidance"),
    [
        ("openai/gpt-5.6-sol", True),
        ("deepseek.deepseek-v4-flash", False),
        ("sonnet", False),
    ],
)
async def test_minimal_shell_extended_guidance_is_gpt56_specific(
    model: str,
    expects_extended_guidance: bool,
) -> None:
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model=model,
    )
    agent = McpAgent(config=config, context=Context(config=Settings()))

    tools = {tool.name: tool for tool in (await agent.list_tools()).tools}
    shell_tool_name = "Shell" if model == "deepseek.deepseek-v4-flash" else "bash"
    bash_description = tools[shell_tool_name].description or ""
    process_description = tools["process"].description or ""

    assert ("task-relevant verification" in bash_description) is expects_extended_guidance
    assert (
        "before relying on its result or ending the task" in process_description
    ) is expects_extended_guidance

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_minimal_process_planned_metadata_matches_runtime_dispatch() -> None:
    settings = Settings()
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    bash_metadata = agent._metadata_for_planned_tool(
        tool_name="bash",
        tool_args={"command": "service", "run_in_background": True},
        local_tool=None,
        is_external_runtime_tool=False,
        is_filesystem_runtime_tool=False,
        route_to_namespaced_candidate=False,
    )
    assert bash_metadata is not None
    assert bash_metadata["background"] is True
    assert bash_metadata["lifecycle"] == "persistent"

    status_metadata = agent._metadata_for_planned_tool(
        tool_name="process",
        tool_args={"process_id": "process-1", "action": "status"},
        local_tool=None,
        is_external_runtime_tool=False,
        is_filesystem_runtime_tool=False,
        route_to_namespaced_candidate=False,
    )
    assert status_metadata is not None
    assert status_metadata["action"] == "poll"
    assert status_metadata["wait_sec"] == 0

    stop_metadata = agent._metadata_for_planned_tool(
        tool_name="process",
        tool_args={"process_id": "process-1", "action": "stop"},
        local_tool=None,
        is_external_runtime_tool=False,
        is_filesystem_runtime_tool=False,
        route_to_namespaced_candidate=False,
    )
    assert stop_metadata is not None
    assert stop_metadata["action"] == "terminate"

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
    assert "edit_file" not in tool_names
    assert "apply_patch" not in tool_names

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_local_write_text_file_option_can_be_disabled() -> None:
    settings = Settings(shell_execution=ShellSettings(write_text_file_mode="off"))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in tool_names
    assert "write_text_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "apply_patch" not in tool_names

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
    assert READ_SKILL_TOOL_NAME in tool_names
    assert agent.skill_read_tool_name == READ_SKILL_TOOL_NAME

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_acp_filesystem_runtime_injection_augments_local_shell_edit_tools() -> None:
    class ACPFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="ACP read tool",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                ),
                Tool(
                    name="write_text_file",
                    description="ACP write tool",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                ),
            ]

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(content=[TextContent(type="text", text="acp")], is_error=False)

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del tool_use_id
            assert arguments is not None
            return CallToolResult(
                content=[TextContent(type="text", text=f"acp-write:{arguments['path']}")],
                is_error=False,
            )

        def metadata(self) -> dict[str, object]:
            return {
                "variant": "acp_filesystem",
                "tools": ["read_text_file", "write_text_file"],
            }

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context())

    initial_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in initial_tool_names
    assert "write_text_file" in initial_tool_names
    assert "edit_file" in initial_tool_names

    acp_runtime = ACPFilesystemRuntime()
    agent.set_filesystem_runtime(cast("Any", acp_runtime))

    replaced_tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in replaced_tool_names
    assert "write_text_file" in replaced_tool_names
    assert "edit_file" in replaced_tool_names

    result = await agent.call_tool("read_text_file", {"path": "/tmp/anything"})
    assert result.is_error is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "acp"

    write_result = await agent.call_tool(
        "write_text_file",
        {"path": "/tmp/output.txt", "content": "ignored by acp stub"},
    )
    assert write_result.is_error is False
    assert write_result.content is not None
    assert isinstance(write_result.content[0], TextContent)
    assert write_result.content[0].text == "acp-write:/tmp/output.txt"

    edit_target = Path(tempfile.gettempdir()) / "fast-agent-edit-file-acp-test.txt"
    edit_target.write_text("hello world\n", encoding="utf-8")
    try:
        edit_result = await agent.call_tool(
            "edit_file",
            {
                "path": str(edit_target),
                "old_string": "world",
                "new_string": "there",
            },
        )
        assert edit_result.is_error is False
        assert edit_target.read_text(encoding="utf-8") == "hello there\n"
    finally:
        edit_target.unlink(missing_ok=True)

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_acp_filesystem_runtime_injection_preserves_local_apply_patch_for_codex_models(
    tmp_path: Path,
) -> None:
    class ACPReadWriteRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="ACP read tool",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                ),
                Tool(
                    name="write_text_file",
                    description="ACP write tool",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                ),
            ]

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="acp-read")], is_error=False
            )

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="acp-write")], is_error=False
            )

        def metadata(self) -> dict[str, object]:
            return {
                "variant": "acp_filesystem",
                "tools": ["read_text_file", "write_text_file"],
            }

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

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
    agent.set_filesystem_runtime(cast("Any", ACPReadWriteRuntime()))

    tool_names = {tool.name for tool in (await agent.list_tools()).tools}
    assert "read_text_file" in tool_names
    assert "write_text_file" in tool_names
    assert "apply_patch" in tool_names
    assert "edit_file" not in tool_names

    patch_text = (
        "*** Begin Patch\n*** Update File: notes.txt\n@@\n-one\n+ONE\n two\n*** End Patch\n"
    )
    result = await agent.call_tool("apply_patch", {"input": patch_text})

    assert result.is_error is False
    assert target_file.read_text(encoding="utf-8") == "ONE\ntwo\n"

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_unprefixed_read_text_file_routes_to_namespaced_mcp_when_local_fs_available() -> None:
    class RecordingFilesystemRuntime:
        def __init__(self) -> None:
            self.tools = [
                Tool(
                    name="read_text_file",
                    description="Local read tool",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ]
            self.read_calls: list[dict[str, object] | None] = []

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del tool_use_id
            self.read_calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="local")], is_error=False)

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="write unsupported")],
                is_error=True,
            )

        def metadata(self) -> dict[str, object]:
            return {"variant": "local_filesystem"}

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())

    mcp_tool = Tool(
        name="read_text_file",
        description="MCP read tool",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
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
        arguments: dict[str, object] | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: object | None = None,
    ) -> CallToolResult:
        del arguments, tool_use_id, request_tool_handler
        mcp_calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="mcp")], is_error=False)

    async def fake_get_app_integration_config(server_name: str) -> None:
        del server_name
        return None

    agent._aggregator.list_tools = cast("Any", fake_list_tools)
    agent._aggregator.call_tool = cast("Any", fake_call_tool)
    agent._aggregator.get_app_integration_config = cast("Any", fake_get_app_integration_config)

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
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                )
            ]
            self.write_calls: list[dict[str, object] | None] = []

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="read unsupported")],
                is_error=True,
            )

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del tool_use_id
            self.write_calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="local")], is_error=False)

        def metadata(self) -> dict[str, object]:
            return {"variant": "local_filesystem"}

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=False)
    agent = McpAgent(config=config, context=Context())
    agent._filesystem_runtime = cast("Any", RecordingFilesystemRuntime())

    mcp_tool = Tool(
        name="write_text_file",
        description="MCP write tool",
        input_schema={
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
        arguments: dict[str, object] | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: object | None = None,
    ) -> CallToolResult:
        del arguments, tool_use_id, request_tool_handler
        mcp_calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="mcp")], is_error=False)

    async def fake_get_app_integration_config(server_name: str) -> None:
        del server_name
        return None

    agent._aggregator.list_tools = cast("Any", fake_list_tools)
    agent._aggregator.call_tool = cast("Any", fake_call_tool)
    agent._aggregator.get_app_integration_config = cast("Any", fake_get_app_integration_config)

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
                name="bash",
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
    assert call["highlight_indexes"] == []
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
    assert call["highlight_indexes"] == []
    assert call["additional_message"] is None

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_grok_catalog_shell_output_limit_applies_when_setting_is_omitted() -> None:
    settings = Settings(shell_execution=ShellSettings())
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="xai/grok-4.5?reasoning=high",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == 16_000

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "xai/grok-4.5?reasoning=high",
        "openrouter/x-ai/grok-4.5",
    ],
)
async def test_grok_uses_aligned_shell_default_and_preserves_native_override(
    model_name: str,
) -> None:
    auto_agent = McpAgent(
        config=AgentConfig(
            name="minimal",
            instruction="Instruction",
            servers=[],
            shell=True,
            model=model_name,
        ),
        context=Context(config=Settings(shell_execution=ShellSettings())),
    )
    auto_runtime = auto_agent.shell_runtime
    assert auto_runtime is not None
    assert {tool.name for tool in auto_runtime.tools} == {
        GROK_SHELL_TOOL_NAME,
        PROCESS_TOOL_NAME,
    }

    native_agent = McpAgent(
        config=AgentConfig(
            name="native",
            instruction="Instruction",
            servers=[],
            shell=True,
            model=model_name,
        ),
        context=Context(
            config=Settings(
                shell_execution=ShellSettings(tool_profile="native"),
            )
        ),
    )
    native_runtime = native_agent.shell_runtime
    assert native_runtime is not None
    assert native_runtime.owns_tool(EXECUTE_TOOL_NAME)
    assert not native_runtime.owns_tool(BASH_TOOL_NAME)
    assert not native_runtime.owns_tool(PROCESS_TOOL_NAME)

    await auto_agent._aggregator.close()
    await native_agent._aggregator.close()


@pytest.mark.asyncio
async def test_default_shell_output_limit_returns_after_switching_away_from_grok() -> None:
    settings = Settings(shell_execution=ShellSettings())
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="xai/grok-4.5",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == 16_000

    agent._on_llm_attached(cast("Any", StubLLM("claude-opus-4-6")))

    assert shell_runtime.output_byte_limit == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
    assert {tool.name for tool in shell_runtime.tools} == {
        BASH_TOOL_NAME,
        PROCESS_TOOL_NAME,
    }

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "shell_tool_name"),
    [
        ("xai/grok-4.5", GROK_SHELL_TOOL_NAME),
        ("codexresponses/gpt-5.6-luna", LUNA_EXEC_TOOL_NAME),
    ],
)
async def test_auto_shell_profile_switches_with_llm(
    model_name: str,
    shell_tool_name: str,
) -> None:
    settings = Settings(shell_execution=ShellSettings())
    agent = McpAgent(
        config=AgentConfig(
            name="test",
            instruction="Instruction",
            servers=[],
            shell=True,
            model="claude-opus-4-6",
        ),
        context=Context(config=settings),
    )
    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert {tool.name for tool in shell_runtime.tools} == {
        BASH_TOOL_NAME,
        PROCESS_TOOL_NAME,
    }

    agent._on_llm_attached(cast("Any", StubLLM(model_name)))

    assert {tool.name for tool in shell_runtime.tools} == {
        shell_tool_name,
        PROCESS_TOOL_NAME,
    }

    await agent._aggregator.close()


@pytest.mark.asyncio
async def test_explicit_null_shell_output_limit_uses_automatic_model_sizing() -> None:
    settings = Settings(shell_execution=ShellSettings(output_byte_limit=None))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="claude-opus-4-6",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == calculate_terminal_output_limit_for_model(
        "claude-opus-4-6"
    )

    await agent._aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_limit", [8192, 32_000])
async def test_explicit_shell_output_limit_overrides_grok_catalog(
    configured_limit: int,
) -> None:
    settings = Settings(shell_execution=ShellSettings(output_byte_limit=configured_limit))
    config = AgentConfig(
        name="test",
        instruction="Instruction",
        servers=[],
        shell=True,
        model="xai/grok-4.5",
    )
    agent = McpAgent(config=config, context=Context(config=settings))

    shell_runtime = agent.shell_runtime
    assert shell_runtime is not None
    assert shell_runtime.output_byte_limit == configured_limit

    agent._on_llm_attached(cast("Any", StubLLM("xai/grok-4.5")))

    assert shell_runtime.output_byte_limit == configured_limit

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
async def test_local_shell_result_is_not_retruncated_by_mcp_result_policy() -> None:
    settings = Settings(shell_execution=ShellSettings(output_byte_limit=9000))
    config = AgentConfig(name="test", instruction="Instruction", servers=[], shell=True)
    agent = McpAgent(config=config, context=Context(config=settings))
    output = "x" * 80

    async def fake_call_tool(
        name: str,
        arguments: dict[str, object] | None = None,
        tool_use_id: str | None = None,
        *,
        request_tool_handler: object | None = None,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        del name, arguments, tool_use_id, request_tool_handler, request_params
        return CallToolResult(content=[TextContent(type="text", text=output)], is_error=False)

    agent.call_tool = cast("Any", fake_call_tool)
    agent._model_tool_output_byte_limit = cast("Any", lambda _llm=None: 40)
    request = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "call-1": CallToolRequest(
                params=CallToolRequestParams(
                    name="bash",
                    arguments={"command": "emit output"},
                )
            )
        },
    )

    result = await agent.run_tools(request)

    assert result.tool_results is not None
    shell_result = result.tool_results["call-1"]
    assert shell_result.content == [TextContent(type="text", text=output)]

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
                input_schema={"type": "object", "properties": {}},
            )
            self.calls: list[dict[str, object]] = []
            self.tools = [self.tool]

        def owns_tool(self, name: str) -> bool:
            return name == self.tool.name

        def metadata(self, command: str | None) -> dict[str, object]:
            return {
                "variant": "shell",
                "command": command,
                "shell_name": "shell",
                "shell_path": "/bin/bash",
            }

        async def execute(
            self,
            arguments: dict[str, object] | None = None,
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
            return CallToolResult(content=[TextContent(type="text", text="ok")], is_error=False)

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            show_tool_call_id: bool = False,
            defer_display_to_tool_result: bool = False,
        ) -> CallToolResult:
            assert name == self.tool.name
            return await self.execute(
                arguments,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )

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
                input_schema={"type": "object", "properties": {}},
            )
            self.tools = [self.tool]

        def owns_tool(self, name: str) -> bool:
            return name == self.tool.name

        def metadata(self, command: str | None) -> dict[str, object]:
            return {
                "variant": "shell",
                "command": command,
                "shell_name": "shell",
                "shell_path": "/bin/bash",
            }

        async def execute(
            self,
            arguments: dict[str, object] | None = None,
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
                is_error=False,
            )
            update_tool_result_display_metadata(
                result,
                {"suppress_display": not defer_display_to_tool_result},
            )
            return result

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            show_tool_call_id: bool = False,
            defer_display_to_tool_result: bool = False,
        ) -> CallToolResult:
            assert name == self.tool.name
            return await self.execute(
                arguments,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )

    class RecordingDisplay:
        def __init__(self) -> None:
            self.call_ids: list[str | None] = []
            self.result_ids: list[str | None] = []
            self.result_text: list[str] = []

        def show_tool_call(self, *args: object, **kwargs: object) -> None:
            tool_call_id = kwargs.get("tool_call_id")
            assert tool_call_id is None or isinstance(tool_call_id, str)
            self.call_ids.append(tool_call_id)

        def show_tool_result(self, *args: object, **kwargs: object) -> None:
            tool_call_id = kwargs.get("tool_call_id")
            assert tool_call_id is None or isinstance(tool_call_id, str)
            self.result_ids.append(tool_call_id)
            result = kwargs.get("result")
            if isinstance(result, CallToolResult) and result.content:
                block = result.content[0]
                if isinstance(block, TextContent):
                    self.result_text.append(block.text)

        def show_parallel_tool_calls(self, requests: list[object]) -> None:
            for request in requests:
                assert isinstance(request, ToolCallDisplayRequest)
                self.show_tool_call(
                    request.tool_name,
                    request.tool_args,
                    tool_call_id=request.tool_call_id,
                )

        def show_parallel_tool_results(self, requests: list[object]) -> None:
            for request in requests:
                assert isinstance(request, ToolResultDisplayRequest)
                self.show_tool_result(
                    result=request.result,
                    tool_call_id=request.tool_call_id,
                )

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
    assert recording_display.call_ids == ["call-1", "call-2"]
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
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ]

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="line-1\nline-2")],
                is_error=False,
            )

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="unsupported")],
                is_error=True,
            )

        def metadata(self) -> dict[str, object]:
            return {"variant": "local_filesystem"}

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

    class RecordingDisplay:
        def __init__(self) -> None:
            self.tool_call_count = 0
            self.result_count = 0

        def show_tool_call(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.tool_call_count += 1

        def show_tool_result(self, *args: object, **kwargs: object) -> None:
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
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ]

        async def read_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="line-1\nline-2")],
                is_error=False,
            )

        async def write_text_file(
            self,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> CallToolResult:
            del arguments, tool_use_id
            return CallToolResult(
                content=[TextContent(type="text", text="unsupported")],
                is_error=True,
            )

        def metadata(self) -> dict[str, object]:
            return {"variant": "local_filesystem"}

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
            *,
            request_params: object | None = None,
        ) -> CallToolResult:
            del request_params
            if name == "read_text_file":
                return await self.read_text_file(arguments, tool_use_id)
            if name == "write_text_file":
                return await self.write_text_file(arguments, tool_use_id)
            return CallToolResult(
                content=[TextContent(type="text", text=f"unsupported: {name}")],
                is_error=True,
            )

    class RecordingDisplay:
        def __init__(self) -> None:
            self.result_tool_call_ids: list[str | None] = []
            self.result_type_labels: list[str | None] = []
            self.results: list[CallToolResult] = []

        def show_tool_call(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            return None

        def show_tool_result(self, *args: object, **kwargs: object) -> None:
            self.results.append(cast("CallToolResult", kwargs["result"]))
            self.result_tool_call_ids.append(cast("str | None", kwargs.get("tool_call_id")))
            self.result_type_labels.append(cast("str | None", kwargs.get("type_label")))

        def show_parallel_tool_calls(self, requests: list[object]) -> None:
            del requests

        def show_parallel_tool_results(self, requests: list[object]) -> None:
            for request in requests:
                assert isinstance(request, ToolResultDisplayRequest)
                self.show_tool_result(
                    result=request.result,
                    tool_call_id=request.tool_call_id,
                    type_label=request.type_label,
                )

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
    assert recording_display.result_tool_call_ids == ["call-1", "call-2"]
    assert [
        tool_result_display_metadata(result).get("read_text_file_line")
        for result in recording_display.results
    ] == [
        1,
        1,
    ]
    assert [
        tool_result_display_metadata(result).get("read_text_file_limit")
        for result in recording_display.results
    ] == [
        20,
        20,
    ]

    await agent._aggregator.close()
