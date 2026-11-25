import asyncio
import json

from mcp_types import CallToolResult, TextContent
from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.command_actions import MarkdownTextStyle
from fast_agent.config import LoggerSettings, Settings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS, REASONING
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.message_primitives import MessageType


def _contains_renderable_type(renderable: object, renderable_type: type[object]) -> bool:
    if isinstance(renderable, renderable_type):
        return True
    if isinstance(renderable, Group):
        return any(
            _contains_renderable_type(child, renderable_type) for child in renderable.renderables
        )
    return False


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
        markdown_styles: tuple[MarkdownTextStyle, ...] = (),
    ) -> None:
        del markdown_styles
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
        channels={REASONING: [TextContent(type="text", text="Considering the answer")]},
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
    assert rendered.index("▎◀ dev") < rendered.index("Considering the answer")
    assert rendered.index("Considering the answer") < rendered.index("FINAL RESPONSE")
    assert rendered.index("FINAL RESPONSE") < rendered.index("Final answer")


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
    tool_result = CallToolResult(content=[TextContent(type="text", text="ok")], is_error=False)

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(message_text=message, name="dev", model="gpt-test")
            display.show_tool_result(tool_result, name="dev", tool_name="demo_tool")
        return capture.get()

    rendered = asyncio.run(_render())
    assert "Inspecting agent failure tracking\n▎▶ dev" in rendered
    assert "Inspecting agent failure tracking\n\n▎▶ dev" not in rendered


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
    display = ConsoleDisplay(config=None)
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

    extracted = display._extract_openai_phase_content(message)

    assert isinstance(extracted, Group)


def test_openai_phase_blocks_render_code_fences_with_syntax() -> None:
    display = ConsoleDisplay(config=None)
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="```python\nprint('hi')\n```")],
        channels={
            OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "message",
                            "phase": "final_answer",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "```python\nprint('hi')\n```",
                                }
                            ],
                        }
                    ),
                ),
            ]
        },
        stop_reason=LlmStopReason.END_TURN,
    )

    extracted = display._extract_openai_phase_content(message)

    assert isinstance(extracted, Group)
    assert _contains_renderable_type(extracted, Syntax)


def test_user_message_header_uses_rule_fill_for_turn_info() -> None:
    display = ConsoleDisplay(config=None)

    with console.console.capture() as capture:
        display.show_user_message("Hello", name="dev", chat_turn=3, total_turns=12)

    rendered = capture.get()
    header = rendered.splitlines()[1]
    assert header.startswith("▎▶ dev ")
    assert "─" in header
    assert header.endswith(" turn 3 (12)")


def test_hc_delivered_input_gets_a_cyan_panel_with_undimmed_text() -> None:
    display = ConsoleDisplay(config=None)

    with console.console.capture() as capture:
        display.show_user_message("Review ok", name="hc · t7", from_hc=True)

    header = capture.get().splitlines()[1]
    assert header.startswith("▎▶ hc · t7")
    assert display._message_header_left(
        message_type=MessageType.HC, name="hc · t7", is_error=False, show_hook_indicator=False
    ).startswith("[cyan]▎[/cyan][dim cyan]▶[/dim cyan]")
    assert (
        ConsoleDisplay._content_display_style(is_error=False, message_type=MessageType.HC) is None
    )


def test_hc_delivered_input_preserves_multiline_terminal_output() -> None:
    display = ConsoleDisplay(config=None)
    notice = (
        "t1p3 worker exited (exit 0). Last lines:\n"
        "┏━━━━━━━━┳━━━━━━━┓\n"
        "┃ Trials ┃  Mean ┃\n"
        "┡━━━━━━━━╇━━━━━━━┩\n"
        "│      1 │ 1.000 │\n"
        "└────────┴───────┘\n"
        "Read: hc read t1p3"
    )

    with console.console.capture() as capture:
        display.show_user_message(notice, name="hc · t1p3 exited", from_hc=True)

    lines = [line.strip() for line in capture.get().splitlines()]
    assert "┏━━━━━━━━┳━━━━━━━┓" in lines
    assert "┃ Trials ┃  Mean ┃" in lines
    assert "│      1 │ 1.000 │" in lines
    assert "Read: hc read t1p3" in lines


def test_show_assistant_message_auto_detect_renders_markdown_with_backticked_tokens() -> None:
    display = ConsoleDisplay(config=None)
    markdown_text = (
        "**Special tokens**: "
        "`<|im_end|>`, `<a>`, `<b>`, `<c>`, `<d>`, `<e>`, `<f>`, `` `<g>` ``, and `<think>`."
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=markdown_text,
                name="dev",
                model="gpt-test",
                render_markdown=None,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "Special tokens" in rendered
    assert "**Special tokens**" not in rendered
    assert "`<|im_end|>`" not in rendered
    assert "<|im_end|>" in rendered
    assert "`<a>`" not in rendered
    assert "<a>" in rendered


def test_show_assistant_message_auto_detect_renders_markdown_with_leading_autolink_and_backticked_tags() -> (
    None
):
    display = ConsoleDisplay(config=None)
    markdown_text = (
        "<https://example.com>\n\n**Tags**: `<a>`, `<b>`, `<c>`, `<d>`, `<e>`, `<f>`, and `<g>`."
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=markdown_text,
                name="dev",
                model="gpt-test",
                render_markdown=None,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "<https://example.com>" in rendered.split()
    assert "Tags" in rendered
    assert "**Tags**" not in rendered
    assert "`<a>`" not in rendered
    assert "<a>" in rendered


def test_show_assistant_message_auto_detect_renders_markdown_with_fenced_xml() -> None:
    display = ConsoleDisplay(config=None)
    markdown_text = (
        "**XML example**:\n\n"
        "```xml\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>demo</name>\n"
        "    <description>desc</description>\n"
        "  </skill>\n"
        "</available_skills>\n"
        "```"
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=markdown_text,
                name="dev",
                model="gpt-test",
                render_markdown=None,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "XML example" in rendered
    assert "**XML example**" not in rendered
    assert "```xml" not in rendered
    assert "<available_skills>" in rendered


def test_show_assistant_message_auto_detect_skips_markdown_for_raw_xml_outside_code() -> None:
    display = ConsoleDisplay(config=None)
    mixed_text = (
        "**Skills**:\n\n"
        "<available_skills>\n"
        "<skill>\n"
        "<name>demo</name>\n"
        "<description>desc</description>\n"
        "</skill>\n"
        "</available_skills>"
    )

    async def _render() -> str:
        with console.console.capture() as capture:
            await display.show_assistant_message(
                message_text=mixed_text,
                name="dev",
                model="gpt-test",
                render_markdown=None,
            )
        return capture.get()

    rendered = asyncio.run(_render())
    assert "**Skills**:" in rendered
    assert "<available_skills>\n<skill>\n<name>demo</name>" in rendered


def test_has_substantial_xml_ignores_non_xml_angles_autolinks_and_code() -> None:
    assert not ConsoleDisplay._has_substantial_xml(
        "Compare a < b, c < d, e < f, g < h, i < j, k < l -> done -> ok -> yes -> no -> end -> !"
    )
    assert not ConsoleDisplay._has_substantial_xml(
        "Links: <https://a.example>, <https://b.example>, <https://c.example>, "
        "<https://d.example>, <https://e.example>, <mailto:user@example.com>."
    )
    assert not ConsoleDisplay._has_substantial_xml(
        "Tokens: <|im_start|> <|im_end|> <|endoftext|> <|pad|> <|bos|> <|eos|>"
    )
    assert not ConsoleDisplay._has_substantial_xml(
        "Indented code:\n\n    <a><b><c><d><e><f></f></e></d></c></b></a>\n"
    )
    assert not ConsoleDisplay._has_substantial_xml(
        "Emphasized angles: <*a*> <*b*> <*c*> <*d*> <*e*> <*f*> <*g*>"
    )
    assert not ConsoleDisplay._has_substantial_xml(
        "Commented tags: <!-- <a><b><c><d><e><f></f></e></d></c></b></a> -->"
    )
    assert not ConsoleDisplay._looks_like_xml_content(
        "<https://example.com>\n\n`<a>` `<b>` `<c>` `<d>` `<e>` `<f>`"
    )
    assert ConsoleDisplay._has_substantial_xml(
        "Prose before\n<root><a>1</a><b>2</b><c>3</c></root>\nProse after"
    )
