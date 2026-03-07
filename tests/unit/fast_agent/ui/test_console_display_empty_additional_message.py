import asyncio

from mcp.types import CallToolResult, TextContent
from rich.text import Text

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.message_primitives import MessageType


class _CaptureContentDisplay(ConsoleDisplay):
    def __init__(self) -> None:
        super().__init__(config=None)
        self.displayed_content: list[object] = []

    def _display_content(
        self,
        content: object,
        truncate: bool = True,
        is_error: bool = False,
        message_type: MessageType | None = None,
        check_markdown_markers: bool = False,
        render_markdown: bool | None = None,
    ) -> None:
        self.displayed_content.append(content)


def test_display_message_skips_empty_string_when_additional_message_present() -> None:
    display = _CaptureContentDisplay()

    display.display_message(
        content="",
        message_type=MessageType.ASSISTANT,
        additional_message=Text("The assistant requested tool calls"),
    )

    assert display.displayed_content == []


def test_display_message_shows_non_empty_content_when_additional_message_present() -> None:
    display = _CaptureContentDisplay()

    display.display_message(
        content="hello",
        message_type=MessageType.ASSISTANT,
        additional_message=Text("extra"),
    )

    assert display.displayed_content == ["hello"]


def test_display_message_keeps_empty_content_without_additional_message() -> None:
    display = _CaptureContentDisplay()

    display.display_message(
        content="",
        message_type=MessageType.ASSISTANT,
    )

    assert display.displayed_content == [""]


def test_display_message_skips_empty_content_when_pre_content_present() -> None:
    display = _CaptureContentDisplay()

    display.display_message(
        content="",
        message_type=MessageType.ASSISTANT,
        pre_content=Text("Reviewing existing plan format", style="dim italic"),
    )

    assert display.displayed_content == []


def test_reasoning_only_turn_does_not_emit_extra_gap_before_tool_result() -> None:
    display = ConsoleDisplay(config=None)

    message = PromptMessageExtended(
        role="assistant",
        content=[],
        channels={
            "reasoning": [
                TextContent(type="text", text="Inspecting agent failure tracking"),
            ]
        },
        stop_reason=LlmStopReason.TOOL_USE,
    )
    tool_result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(message_text=message, name="dev", model="gpt-test")
            display.show_tool_result(tool_result, name="dev", tool_name="demo_tool")
        return capture.get()

    rendered = asyncio.run(_render())
    assert "Inspecting agent failure tracking\n\n▎▶ dev" in rendered
    assert "Inspecting agent failure tracking\n\n\n▎▶ dev" not in rendered


def test_reasoning_then_text_has_single_blank_separator() -> None:
    display = ConsoleDisplay(config=None)

    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        channels={
            "reasoning": [
                TextContent(type="text", text="Thinking"),
            ]
        },
        stop_reason=LlmStopReason.END_TURN,
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(message_text=message, name="dev", model="gpt-test")
        return capture.get()

    rendered = asyncio.run(_render())
    assert "Thinking\n\nFinal answer" in rendered
    assert "Thinking\n\n\nFinal answer" not in rendered
