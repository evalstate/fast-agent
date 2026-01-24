from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent
from rich.text import Text

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.registry import SkillRegistry
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.console_display import ConsoleDisplay

if TYPE_CHECKING:
    from mcp.types import CallToolResult


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
