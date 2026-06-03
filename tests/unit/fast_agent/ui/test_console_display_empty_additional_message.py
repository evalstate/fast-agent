import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.config import LoggerSettings, Settings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay, ParallelAgentDisplayResult
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.streaming_preferences import resolve_streaming_preferences


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
    ) -> None:
        self.displayed_content.append(content)


class _ReadOnlyPartialConfig:
    @property
    def logger(self) -> SimpleNamespace:
        return SimpleNamespace(show_chat=True)


class _FailingLoggerConfig:
    @property
    def logger(self) -> None:
        return None

    @logger.setter
    def logger(self, value: LoggerSettings) -> None:
        raise RuntimeError("logger setter failed")


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


def test_console_display_defaults_missing_logger_on_lightweight_config() -> None:
    config = SimpleNamespace()

    display = ConsoleDisplay(config=config)

    assert display.code_word_wrap is True
    assert isinstance(config.logger, LoggerSettings)


def test_console_display_syncs_logger_settings_to_mapping_config() -> None:
    config: dict[str, object] = {}

    display = ConsoleDisplay(config=config)

    assert display.code_word_wrap is True
    assert isinstance(config["logger"], LoggerSettings)


def test_console_display_defaults_missing_logger_fields_on_lightweight_config() -> None:
    config = SimpleNamespace(logger=SimpleNamespace(show_chat=False))

    display = ConsoleDisplay(config=config)

    assert display.code_word_wrap is True
    assert display.render_fences_with_syntax is True
    assert display.resolve_streaming_preferences().enabled is False


def test_console_display_does_not_swallow_logger_setter_failures() -> None:
    with pytest.raises(RuntimeError, match="logger setter failed"):
        ConsoleDisplay(config=_FailingLoggerConfig())


def test_tool_display_uses_normalized_logger_settings_for_read_only_config() -> None:
    display = ConsoleDisplay(config=_ReadOnlyPartialConfig())

    with console.console.capture() as capture:
        display.show_tool_call("lookup", {"query": "docs"})

    assert "lookup" in capture.get()


def test_assistant_images_use_normalized_settings_for_read_only_config() -> None:
    display = ConsoleDisplay(config=_ReadOnlyPartialConfig())
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
            )
        return capture.get()

    assert "Final answer" in asyncio.run(_render())


def test_console_display_resolves_streaming_preferences_from_logger_settings() -> None:
    display = ConsoleDisplay(
        config=Settings(
            logger=LoggerSettings(
                show_chat=False,
                streaming="plain",
            )
        )
    )

    preferences = display.resolve_streaming_preferences()

    assert preferences.mode == "plain"
    assert preferences.enabled is False


def test_streaming_preferences_normalizes_legacy_and_invalid_modes() -> None:
    plain_preferences = resolve_streaming_preferences(
        SimpleNamespace(
            show_chat=True,
            streaming="markdown",
            streaming_plain_text=True,
            streaming_display=True,
        )
    )
    invalid_preferences = resolve_streaming_preferences(
        SimpleNamespace(show_chat=True, streaming="bogus", streaming_display=True)
    )

    assert plain_preferences.mode == "plain"
    assert plain_preferences.enabled is True
    assert invalid_preferences.mode == "markdown"
    assert invalid_preferences.enabled is True


def test_parallel_result_header_prints_bracketed_model_literally() -> None:
    display = ConsoleDisplay(config=None)

    with console.console.capture() as capture:
        display._show_parallel_result_header(
            ParallelAgentDisplayResult(
                name="worker",
                model="[draft]",
                content="ok",
                tokens=0,
                tool_calls=0,
            )
        )

    assert "[draft]" in capture.get()


def test_parallel_summary_prints_with_styled_text() -> None:
    display = ConsoleDisplay(config=None)
    summary = display._parallel_summary_text(
        [
            ParallelAgentDisplayResult(
                name="worker",
                model="[draft]",
                content="ok",
                tokens=12,
                tool_calls=1,
            )
        ]
    )

    assert summary == "1 model • 12 tokens • 1 tool"


def test_parallel_result_usage_label_pluralizes_tools() -> None:
    result = ParallelAgentDisplayResult(
        name="worker",
        model="gpt",
        content="ok",
        tokens=0,
        tool_calls=1,
    )

    assert ConsoleDisplay._parallel_result_usage_label(result) == "1 tool"


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


def test_openai_phase_blocks_skip_non_text_items() -> None:
    display = ConsoleDisplay(config=None)

    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        channels={
            OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                ImageContent(type="image", data="abc", mimeType="image/png"),
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
