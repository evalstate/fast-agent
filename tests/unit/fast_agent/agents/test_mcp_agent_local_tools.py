import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent, Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import Settings, ShellSettings
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.context import Context
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model
from fast_agent.skills.registry import SkillRegistry
from fast_agent.types import PromptMessageExtended
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
