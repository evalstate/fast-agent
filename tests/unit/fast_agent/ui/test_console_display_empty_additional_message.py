import asyncio
import json

from mcp.types import CallToolResult, TextContent
from rich.console import Group
from rich.text import Text

from fast_agent.config import LoggerSettings, Settings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS
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


def test_console_display_uses_logger_render_settings_by_default() -> None:
    display = ConsoleDisplay(
        config=Settings(
            logger=LoggerSettings(
                code_theme="monokai",
                render_fences_with_syntax=False,
                code_word_wrap=True,
            )
        )
    )

    assert display.code_style == "monokai"
    assert display.render_fences_with_syntax is False
    assert display.code_word_wrap is True


def test_console_display_wraps_code_by_default() -> None:
    display = ConsoleDisplay(config=None)

    assert display.code_word_wrap is True


def test_normalize_assistant_display_text_trims_trailing_blank_lines() -> None:
    assert ConsoleDisplay._normalize_assistant_display_text("hello\n\n") == "hello"

    text = Text("hello\n\n", style="green")
    normalized = ConsoleDisplay._normalize_assistant_display_text(text)

    assert isinstance(normalized, Text)
    assert normalized.plain == "hello"


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


def test_show_stream_reprint_banner_renders_three_lines() -> None:
    display = ConsoleDisplay(config=None)

    with console.console.capture() as capture:
        display.show_stream_reprint_banner()

    rendered_lines = capture.get().splitlines()
    assert len(rendered_lines) == 3
    assert "FINAL RESPONSE" in rendered_lines[1]


def test_show_stream_reprint_banner_respects_logger_setting() -> None:
    display = ConsoleDisplay(
        config=Settings(
            logger=LoggerSettings(
                show_chat=True,
                stream_reprint_banner=False,
            )
        )
    )

    with console.console.capture() as capture:
        display.show_stream_reprint_banner()

    assert capture.get() == ""


def test_assistant_reprint_banner_is_rendered_with_final_assistant_message() -> None:
    display = ConsoleDisplay(config=None)
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        stop_reason=LlmStopReason.END_TURN,
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=message,
                name="dev",
                model="gpt-test",
                show_reprint_banner=True,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "FINAL RESPONSE" in rendered
    assert rendered.index("FINAL RESPONSE") < rendered.index("▎◀ dev")
    assert rendered.index("▎◀ dev") < rendered.index("Final answer")


def test_assistant_reprint_banner_can_be_disabled_in_logger_settings() -> None:
    display = ConsoleDisplay(
        config=Settings(
            logger=LoggerSettings(
                show_chat=True,
                stream_reprint_banner=False,
            )
        )
    )
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        stop_reason=LlmStopReason.END_TURN,
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=message,
                name="dev",
                model="gpt-test",
                show_reprint_banner=True,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "FINAL RESPONSE" not in rendered
    assert "▎◀ dev" in rendered
    assert "Final answer" in rendered


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


def test_assistant_pre_content_renders_between_reasoning_and_final_answer() -> None:
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
    sources = Text("Sources\n [1] Example — https://example.com\n")

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=message,
                name="dev",
                model="gpt-test",
                pre_content=sources,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert rendered.index("Thinking") < rendered.index("Sources") < rendered.index("Final answer")


def test_openai_phase_blocks_render_with_friendly_labels_in_assistant_output() -> None:
    display = ConsoleDisplay(config=None)

    message = PromptMessageExtended(
        role="assistant",
        content=[
            TextContent(type="text", text="Let me inspect that first.\n\nFinal answer"),
        ],
        channels={
            OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "message",
                            "phase": "commentary",
                            "content": [
                                {"type": "output_text", "text": "Let me inspect that first."}
                            ],
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "message",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "Final answer"}],
                        }
                    ),
                ),
            ]
        },
        stop_reason=LlmStopReason.END_TURN,
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(message_text=message, name="dev", model="gpt-test")
        return capture.get()

    rendered = asyncio.run(_render())
    assert "Commentary" in rendered
    assert "Let me inspect that first." in rendered
    assert "Final Answer:" in rendered
    assert "Final answer" in rendered


def test_openai_phase_blocks_use_renderable_group_for_dim_labels() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        channels={
            OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "message",
                            "phase": "final_answer",
                            "content": [{"type": "output_text", "text": "Final answer"}],
                        }
                    ),
                ),
            ]
        },
        stop_reason=LlmStopReason.END_TURN,
    )

    extracted = ConsoleDisplay._extract_openai_phase_content(message)

    assert isinstance(extracted, Group)


def test_user_message_header_uses_rule_fill_for_turn_info() -> None:
    display = ConsoleDisplay(config=None)

    with console.console.capture() as capture:
        display.show_user_message("Hello", name="dev", chat_turn=3, total_turns=12)

    rendered = capture.get()
    header = rendered.splitlines()[1]
    assert header.startswith("▎▶ dev ")
    assert "─" in header
    assert header.endswith(" turn 3 (12)")
