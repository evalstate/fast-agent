import json
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from mcp.types import CallToolResult, ContentBlock
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.protocol import is_renderable
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.config import LoggerSettings, TerminalImageSettings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS, REASONING
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_display_name import resolve_llm_display_name, resolve_model_display_name
from fast_agent.types.assistant_message_phase import coerce_assistant_message_phase
from fast_agent.ui import console
from fast_agent.ui.display_suppression import (
    display_chat_enabled,
    display_status_enabled,
    display_tools_enabled,
)
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.markdown_renderables import build_markdown_renderable
from fast_agent.ui.mcp_ui_utils import UILink
from fast_agent.ui.mermaid_utils import (
    MermaidDiagram,
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
)
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.message_styles import A3MessageStyle
from fast_agent.ui.shell_output_truncation import format_shell_output_line_count
from fast_agent.ui.streaming import (
    NullStreamingHandle as _NullStreamingHandle,
)
from fast_agent.ui.streaming import (
    StreamingHandle,
)
from fast_agent.ui.streaming import (
    StreamingMessageHandle as _StreamingMessageHandle,
)
from fast_agent.ui.streaming_preferences import (
    StreamingPreferences,
    resolve_streaming_preferences,
)
from fast_agent.ui.tool_call_ids import format_tool_call_id
from fast_agent.ui.tool_display import ToolDisplay
from fast_agent.utils.count_display import format_count
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.mcp.skybridge import SkybridgeServerConfig
    from fast_agent.ui.terminal_images import ImageRenderItem

logger = get_logger(__name__)

# Glyph to indicate tool hooks are active
HOOK_INDICATOR_GLYPH = "◆"

PHASE_LABELS = {"final_answer": "Final Answer:", "commentary": "Commentary"}


@runtime_checkable
class _LoggerConfig(Protocol):
    logger: object | None


@runtime_checkable
class _TextPayloadBlock(Protocol):
    text: object


@dataclass(frozen=True, slots=True)
class ParallelAgentDisplayResult:
    name: str
    model: str
    content: str
    tokens: int
    tool_calls: int


@dataclass(frozen=True, slots=True)
class OpenAIPhaseSection:
    phase: str | None
    text: str


class ConsoleDisplay:
    """
    Handles displaying formatted messages, tool calls, and results to the console.
    This centralizes the UI display logic used by LLM implementations.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        code_word_wrap: bool | None = None,
        render_fences_with_syntax: bool | None = None,
        code_theme: str | None = None,
    ) -> None:
        """
        Initialize the console display handler.

        Args:
            config: Configuration object containing display preferences
        """
        self.config = config
        self._logger_settings = self._resolve_logger_settings(config)
        self._sync_logger_settings()
        self._markup = self._logger_settings.enable_markup
        self._escape_xml = True
        self._code_word_wrap = (
            self._logger_settings.code_word_wrap if code_word_wrap is None else code_word_wrap
        )
        self._render_fences_with_syntax = (
            self._logger_settings.render_fences_with_syntax
            if render_fences_with_syntax is None
            else render_fences_with_syntax
        )
        self._code_style = self._logger_settings.code_theme if code_theme is None else code_theme
        self._apply_console_theme()
        self._style = A3MessageStyle()
        self._tool_display = ToolDisplay(self)

    @staticmethod
    def _resolve_logger_settings(config: Any | None) -> LoggerSettings:
        """Provide a logger settings object even when callers omit it."""
        if config is None:
            return LoggerSettings()
        if isinstance(config, Mapping):
            logger_settings = config.get("logger")
        elif isinstance(config, _LoggerConfig):
            logger_settings = config.logger
        else:
            return LoggerSettings()
        if logger_settings is None:
            return LoggerSettings()
        if isinstance(logger_settings, LoggerSettings):
            return logger_settings
        if isinstance(logger_settings, Mapping):
            return LoggerSettings.model_validate(logger_settings)
        return LoggerSettings.model_validate(logger_settings, from_attributes=True)

    @property
    def logger_settings(self) -> LoggerSettings:
        return self._logger_settings

    def update_logger_settings(self, logger_settings: LoggerSettings) -> None:
        """Replace display logger settings and keep the backing config in sync."""
        self._logger_settings = logger_settings
        self._markup = logger_settings.enable_markup
        self._sync_logger_settings()

    def _sync_logger_settings(self) -> None:
        if self.config is None:
            return
        if isinstance(self.config, MutableMapping):
            self.config["logger"] = self._logger_settings
            return
        try:
            self.config.logger = self._logger_settings
        except AttributeError:
            return

    def _apply_console_theme(self) -> None:
        theme_file = self._logger_settings.theme_file
        if theme_file is None and self.config is None:
            return

        base_dir: Path | None = None

        theme_config_file = self._logger_settings._theme_file_config_path
        config_file = theme_config_file or (
            getattr(self.config, "_config_file", None) if self.config else None
        )
        if config_file:
            base_dir = Path(config_file).expanduser().resolve().parent

        try:
            console.configure_console_theme(theme_file, base_dir=base_dir)
        except Exception as exc:
            console.configure_console_theme(None)
            logger.warning(
                "Failed to load Rich theme file; using default console theme",
                data={"theme_file": theme_file, "error": str(exc)},
            )

    def _truncate_text(self, text: str, *, truncate: bool) -> str:
        if truncate and self._logger_settings.truncate_tools and len(text) > 360:
            return text[:360] + "..."
        return text

    def _print_with_style(self, content: object, *, style: str | None) -> None:
        if style:
            console.console.print(content, style=style, markup=self._markup)
        else:
            console.console.print(content, markup=self._markup)

    def _print_plain_text(self, text: str, *, truncate: bool, style: str | None) -> None:
        safe_text = self._truncate_text(text, truncate=truncate)
        if self._markup:
            safe_text = escape_markup(safe_text)
        self._print_with_style(safe_text, style=style)

    def _print_pretty(self, content: object, *, truncate: bool, style: str | None) -> None:
        from rich.pretty import Pretty

        if truncate and self._logger_settings.truncate_tools:
            pretty_obj = Pretty(content, max_length=10, max_string=50)
        else:
            pretty_obj = Pretty(content)
        self._print_with_style(pretty_obj, style=style)

    @property
    def code_style(self) -> str:
        return self._code_style

    @property
    def markup_enabled(self) -> bool:
        return self._markup

    @property
    def code_word_wrap(self) -> bool:
        return self._code_word_wrap

    @property
    def render_fences_with_syntax(self) -> bool:
        return self._render_fences_with_syntax

    @property
    def apply_patch_preview_max_lines(self) -> int | None:
        return self._logger_settings.apply_patch_preview_max_lines

    @property
    def show_tools_enabled(self) -> bool:
        return self._logger_settings.show_tools

    @property
    def terminal_image_settings(self) -> TerminalImageSettings:
        return self._logger_settings.terminal_images

    @property
    def style(self) -> A3MessageStyle:
        return self._style

    def _chat_output_enabled(self) -> bool:
        return display_chat_enabled() and self._logger_settings.show_chat

    def show_status_message(self, content: Text) -> None:
        """Display a status message without a header."""
        if not display_status_enabled():
            return
        console.ensure_blocking_console()
        console.console.print(content, markup=self._markup)

    def show_stream_reprint_banner(self, *, label: str | None = None) -> None:
        """Display a bright banner before reprinting a streamed final response."""
        if not self._chat_output_enabled():
            return
        if not self._logger_settings.stream_reprint_banner:
            return
        console.ensure_blocking_console()
        for line in self._style.stream_reprint_banner(
            console.console.size.width,
            label=label,
        ):
            console.console.print(line, markup=self._markup)

    def resolve_streaming_preferences(self) -> StreamingPreferences:
        """Return whether streaming is enabled plus the active mode."""
        return resolve_streaming_preferences(self._logger_settings)

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        """
        Heuristic to detect markdown-ish content.

        We keep this lightweight: focus on common structures that benefit from markdown
        rendering without requiring strict syntax validation.
        """
        import re

        if not text or len(text) < 3:
            return False

        if "```" in text:
            return True

        # Simple markers for common cases that the regex might miss
        # Note: single "*" excluded to avoid false positives
        simple_markers = ["##", "**", "---", "###"]
        if any(marker in text for marker in simple_markers):
            return True

        markdown_patterns = [
            r"^#{1,6}\s+\S",  # headings
            r"^\s*[-*+]\s+\S",  # unordered list
            r"^\s*\d+\.\s+\S",  # ordered list
            r"`[^`]+`",  # inline code
            r"\*\*[^*]+\*\*",
            r"__[^_]+__",
            r"^\s*>\s+\S",  # blockquote
            r"\[.+?\]\(.+?\)",  # links
            r"!\[.*?\]\(.+?\)",  # images
            r"^\s*\|.+\|\s*$",  # simple tables
            r"^\s*[-*_]{3,}\s*$",  # horizontal rules
        ]

        return any(re.search(pattern, text, re.MULTILINE) for pattern in markdown_patterns)

    @staticmethod
    def _format_elapsed(elapsed: float) -> str:
        """Format elapsed seconds for display."""
        if elapsed < 0:
            elapsed = 0.0
        if elapsed < 0.001:
            return "<1ms"
        if elapsed < 1:
            return f"{elapsed * 1000:.0f}ms"
        if elapsed < 10:
            return f"{elapsed:.2f}s"
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        return format_duration(elapsed)

    def show_shell_exit_code(
        self,
        exit_code: int,
        *,
        no_output: bool = False,
        output_line_count: int | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Display a shell-style exit code banner."""
        detail_parts: list[str] = []
        if output_line_count is not None and output_line_count > 0:
            detail_parts.append(format_shell_output_line_count(output_line_count))

        if no_output:
            detail_parts.append("(no output)")

        formatted_id = format_tool_call_id(tool_call_id)
        if formatted_id:
            detail_parts.append(f"id: {formatted_id}")

        detail = f" {' '.join(detail_parts)}" if detail_parts else None

        line = self._style.shell_exit_line(
            exit_code,
            detail,
        )
        console.console.print()
        console.console.print(line)
        for _ in range(self._style.shell_exit_spacing_after):
            console.console.print()

    def show_managed_process_status(
        self,
        *,
        process_id: str,
        status: str,
        reason: str | None,
        elapsed_seconds: float,
        os_process_id: int | None,
    ) -> None:
        """Display a compact, nonduplicating managed-process lifecycle line."""
        detail_parts = [status]
        if reason == "background":
            detail_parts.append("background")
        elif reason == "idle":
            detail_parts.append("idle yield")
        elif reason == "foreground":
            detail_parts.append("foreground yield")
        detail_parts.append(self._format_elapsed(elapsed_seconds))
        if os_process_id is not None:
            detail_parts.append(f"pid {os_process_id}")

        line = Text("▎", style="dim")
        line.append(" ▶ ", style="yellow")
        line.append(process_id, style="bold")
        line.append(" ")
        line.append(" • ".join(detail_parts), style="dim")
        console.console.print()
        console.console.print(line)
        for _ in range(self._style.shell_exit_spacing_after):
            console.console.print()

    def _format_header_line(
        self,
        left_content: str,
        right_info: str = "",
        *,
        rule_fill: bool = False,
    ) -> Text:
        width = console.console.size.width
        return self._style.header_line(left_content, right_info, width, rule_fill=rule_fill)

    @staticmethod
    def build_header_left(
        block_color: str,
        arrow: str,
        arrow_style: str,
        name: str | None = None,
        is_error: bool = False,
        show_hook_indicator: bool = False,
    ) -> str:
        """
        Build the left side of a message header.

        Args:
            block_color: Color for the block indicator and name
            arrow: Arrow character for the message type
            arrow_style: Style for the arrow
            name: Optional name to display (agent name, user name, etc.)
            is_error: Whether this is an error message (uses red for name)
            show_hook_indicator: Whether to show the hook indicator glyph

        Returns:
            Rich markup string for the left side of the header
        """
        left = f"[{block_color}]▎[/{block_color}]"
        if arrow:
            left += f"[{arrow_style}]{arrow}[/{arrow_style}]"
        if show_hook_indicator:
            left += f" [{block_color}]{HOOK_INDICATOR_GLYPH}[/{block_color}]"
        if name:
            name_color = block_color if not is_error else "red"
            left += f" [{name_color}]{escape_markup(name)}[/{name_color}]"
        return left

    def display_message(
        self,
        content: Any,
        message_type: MessageType,
        name: str | None = None,
        right_info: str = "",
        bottom_metadata: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        is_error: bool = False,
        truncate_content: bool = True,
        additional_message: Text | None = None,
        pre_content: Text | Group | None = None,
        post_content: RenderableType | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
        header_rule_fill: bool = False,
    ) -> None:
        """
        Unified method to display formatted messages to the console.

        Args:
            content: The main content to display (str, Text, JSON, etc.)
            message_type: Type of message (USER, ASSISTANT, TOOL_CALL, TOOL_RESULT)
            name: Optional name to display (agent name, user name, etc.)
            right_info: Information to display on the right side of the header
            bottom_metadata: Optional list of items for bottom separator
            highlight_indexes: Indexes of items to highlight in bottom metadata
            max_item_length: Optional max length for bottom metadata items (with ellipsis)
            is_error: For tool results, whether this is an error (uses red color)
            truncate_content: Whether to truncate long content
            additional_message: Optional Rich Text appended after the main content
            pre_content: Optional Rich Text shown before the main content
            post_content: Optional Rich renderable shown after the main content
            render_markdown: Force markdown rendering (True) or plain rendering (False)
            show_hook_indicator: Whether to show the hook indicator glyph (◆)
            header_rule_fill: Whether to extend the header with a dim rule to the right edge
        """
        console.ensure_blocking_console()

        left_header = self._message_header_left(
            message_type=message_type,
            name=name,
            is_error=is_error,
            show_hook_indicator=show_hook_indicator,
        )
        self._create_combined_separator_status(
            left_header,
            right_info,
            rule_fill=header_rule_fill,
        )
        skip_empty_content = self._is_empty_message_content(content) and (
            additional_message is not None or pre_content is not None
        )

        self._print_pre_content(pre_content, skip_empty_content=skip_empty_content)

        if not skip_empty_content:
            self._display_content(
                content,
                truncate_content,
                is_error,
                message_type,
                check_markdown_markers=False,
                render_markdown=render_markdown,
            )
        if additional_message:
            console.console.print(additional_message, markup=self._markup)
        if post_content:
            console.console.print()
            console.console.print(post_content, markup=self._markup)

        # Handle bottom separator with optional metadata
        self._render_bottom_metadata(
            message_type=message_type,
            bottom_metadata=bottom_metadata,
            highlight_indexes=highlight_indexes,
            max_item_length=max_item_length,
        )

    @staticmethod
    def _message_block_color(message_type: MessageType, *, is_error: bool) -> str:
        if is_error and message_type == MessageType.TOOL_RESULT:
            return "red"
        return MESSAGE_CONFIGS[message_type]["block_color"]

    def _message_header_left(
        self,
        *,
        message_type: MessageType,
        name: str | None,
        is_error: bool,
        show_hook_indicator: bool,
    ) -> str:
        config = MESSAGE_CONFIGS[message_type]
        return self.build_header_left(
            block_color=self._message_block_color(message_type, is_error=is_error),
            arrow=config["arrow"],
            arrow_style=config["arrow_style"],
            name=name,
            is_error=is_error,
            show_hook_indicator=show_hook_indicator,
        )

    @staticmethod
    def _is_empty_message_content(content: Any) -> bool:
        if isinstance(content, str):
            return content == ""
        if isinstance(content, Text):
            return content.plain == ""
        return False

    def _print_pre_content(
        self,
        pre_content: Text | Group | None,
        *,
        skip_empty_content: bool,
    ) -> None:
        if not pre_content:
            return
        if not isinstance(pre_content, Text) or pre_content.plain:
            console.console.print(pre_content, markup=self._markup)
        if not skip_empty_content:
            console.console.print()

    @staticmethod
    def _user_message_turn_info(
        *,
        chat_turn: int,
        total_turns: int | None,
        turn_range: tuple[int, int] | None,
        part_count: int | None,
    ) -> str:
        if part_count and part_count > 1:
            turn_number = turn_range[0] if turn_range else chat_turn
            return f"turn {turn_number}" if turn_number > 0 else ""

        if turn_range:
            turn_start, turn_end = turn_range
            if turn_start == turn_end:
                turn_info = f"turn {turn_start}"
            else:
                turn_info = f"turn {turn_start}-{turn_end}"
            return f"{turn_info} ({total_turns})" if total_turns else turn_info

        if chat_turn <= 0:
            return ""
        return f"turn {chat_turn} ({total_turns})" if total_turns else f"turn {chat_turn}"

    @classmethod
    def _user_message_right_info(
        cls,
        *,
        chat_turn: int,
        total_turns: int | None,
        turn_range: tuple[int, int] | None,
        part_count: int | None,
    ) -> str:
        right_parts: list[str] = []
        turn_info = cls._user_message_turn_info(
            chat_turn=chat_turn,
            total_turns=total_turns,
            turn_range=turn_range,
            part_count=part_count,
        )
        if turn_info:
            right_parts.append(turn_info)
        if part_count and part_count > 1:
            right_parts.append(f"({part_count} parts)")
        return f"[dim]{' '.join(right_parts)}[/dim]" if right_parts else ""

    @staticmethod
    def _attachment_pre_content(attachments: list[str] | None) -> Text | None:
        if not attachments:
            return None
        attachment_text = Text()
        attachment_text.append("🔗 ", style="dim")
        attachment_text.append(", ".join(attachments), style="dim blue")
        return attachment_text

    def _image_preview_pre_content(
        self,
        image_previews: list["ImageRenderItem"] | None,
    ) -> Group | None:
        if not image_previews:
            return None
        terminal_images = self._logger_settings.terminal_images
        if not terminal_images.enabled or terminal_images.backend == "none":
            return None

        from fast_agent.ui.terminal_images import render_image_items

        rendered_previews = render_image_items(terminal_images, image_previews)
        if rendered_previews is None:
            return None
        return Group(rendered_previews)

    def _user_message_pre_content(
        self,
        *,
        attachments: list[str] | None,
        image_previews: list["ImageRenderItem"] | None,
    ) -> Text | Group | None:
        pre_content_parts: list[Text | Group] = []
        attachment_content = self._attachment_pre_content(attachments)
        if attachment_content is not None:
            pre_content_parts.append(attachment_content)

        image_content = self._image_preview_pre_content(image_previews)
        if image_content is not None:
            pre_content_parts.append(image_content)

        if len(pre_content_parts) == 1:
            return pre_content_parts[0]
        if pre_content_parts:
            return Group(*pre_content_parts)
        return None

    @staticmethod
    def _content_display_style(
        *,
        is_error: bool,
        message_type: MessageType | None,
    ) -> str | None:
        if is_error:
            return "dim red"
        if message_type in (MessageType.USER, MessageType.ASSISTANT, MessageType.SYSTEM):
            return None
        return "dim"

    def _print_markdown_text(self, content: str) -> None:
        console.console.print(
            build_markdown_renderable(
                content,
                code_theme=self.code_style,
                escape_xml=self._escape_xml,
                render_fences_with_syntax=self.render_fences_with_syntax,
                code_word_wrap=self.code_word_wrap,
            ),
            markup=self._markup,
        )

    @staticmethod
    def _looks_like_xml_content(content: str) -> bool:
        import re

        xml_pattern = r"^<[a-zA-Z_][a-zA-Z0-9_-]*[^>]*>"
        return bool(re.match(xml_pattern, content.strip())) and content.count("<") > 5

    @staticmethod
    def _has_substantial_xml(content: str) -> bool:
        import re

        xml_probe = re.sub(r"<(?:https?://|mailto:)[^>]+>", "", content)
        return xml_probe.count("<") > 5 and xml_probe.count(">") > 5

    def _display_forced_markdown_string(
        self,
        content: str,
        *,
        truncate: bool,
        style: str | None,
        render_markdown: bool,
    ) -> None:
        try:
            json_obj = json.loads(content)
            self._print_pretty(json_obj, truncate=truncate, style=style)
        except (JSONDecodeError, TypeError, ValueError):
            if render_markdown:
                self._print_markdown_text(content)
            else:
                self._print_plain_text(content, truncate=truncate, style=style)

    def _display_string_content(
        self,
        content: str,
        *,
        truncate: bool,
        style: str | None,
        check_markdown_markers: bool,
        render_markdown: bool | None,
    ) -> None:
        if render_markdown is not None:
            self._display_forced_markdown_string(
                content,
                truncate=truncate,
                style=style,
                render_markdown=render_markdown,
            )
            return

        try:
            json_obj = json.loads(content)
            self._print_pretty(json_obj, truncate=truncate, style=style)
            return
        except (JSONDecodeError, TypeError, ValueError):
            pass

        if self._looks_like_xml_content(content):
            console.console.print(
                Syntax(
                    content,
                    "xml",
                    theme=self.code_style,
                    line_numbers=False,
                    word_wrap=self.code_word_wrap,
                ),
                markup=self._markup,
            )
            return

        if check_markdown_markers:
            if self._looks_like_markdown(content):
                self._print_markdown_text(content)
            else:
                self._print_plain_text(content, truncate=truncate, style=style)
            return

        if self._looks_like_markdown(content) and not self._has_substantial_xml(content):
            self._print_markdown_text(content)
            return

        self._print_plain_text(content, truncate=truncate, style=style)

    def _display_forced_markdown_text(
        self,
        content: Text,
        *,
        truncate: bool,
        style: str | None,
        render_markdown: bool,
    ) -> None:
        plain_text = content.plain
        try:
            json_obj = json.loads(plain_text)
            self._print_pretty(json_obj, truncate=truncate, style=style)
        except (JSONDecodeError, TypeError, ValueError):
            if render_markdown:
                prepared_content = prepare_markdown_content(plain_text, self._escape_xml)
                console.console.print(
                    Markdown(prepared_content, code_theme=self.code_style),
                    markup=self._markup,
                )
            else:
                console.console.print(content, markup=self._markup)

    def _display_text_markdown_with_spans(self, content: Text) -> None:
        plain_text = content.plain
        markdown_end = content._spans[0].end if content._spans else len(plain_text)
        markdown_part = plain_text[:markdown_end]
        if not self._looks_like_markdown(markdown_part):
            console.console.print(content, markup=self._markup)
            return

        self._print_markdown_text(markdown_part)
        if markdown_end >= len(plain_text):
            return

        remaining_text = Text()
        for span in content._spans:
            if span.start >= markdown_end:
                segment_text = plain_text[span.start : span.end]
                remaining_text.append(segment_text, style=span.style)
        if remaining_text.plain:
            console.console.print(remaining_text, markup=self._markup)

    def _display_text_content(
        self,
        content: Text,
        *,
        truncate: bool,
        style: str | None,
        render_markdown: bool | None,
    ) -> None:
        if render_markdown is not None:
            self._display_forced_markdown_text(
                content,
                truncate=truncate,
                style=style,
                render_markdown=render_markdown,
            )
            return

        plain_text = content.plain
        if not self._looks_like_markdown(plain_text):
            console.console.print(content, markup=self._markup)
            return

        if len(content._spans) > 1:
            self._display_text_markdown_with_spans(content)
        else:
            self._print_markdown_text(plain_text)

    def _display_content_blocks(
        self,
        content: list[Any],
        *,
        truncate: bool,
        style: str | None,
    ) -> None:
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        if len(content) == 1 and is_text_content(content[0]):
            text_content = get_text(content[0])
            if text_content:
                self._print_plain_text(text_content, truncate=truncate, style=style)
            elif style:
                console.console.print("(empty text)", style=style, markup=self._markup)
            else:
                console.console.print("(empty text)", markup=self._markup)
            return

        from fast_agent.mcp.prompt_render import render_content_blocks

        self._print_plain_text(
            render_content_blocks(cast("list[ContentBlock]", content)),
            truncate=truncate,
            style=style,
        )

    def _display_content(
        self,
        content: Any,
        truncate: bool = True,
        is_error: bool = False,
        message_type: MessageType | None = None,
        check_markdown_markers: bool = False,
        render_markdown: bool | None = None,
    ) -> None:
        """
        Display content in the appropriate format.

        Args:
            content: Content to display
            truncate: Whether to truncate long content
            is_error: Whether this is error content (affects styling)
            message_type: Type of message to determine appropriate styling
            check_markdown_markers: If True, only use markdown rendering when markers are present
            render_markdown: If set, force markdown rendering (True) or plain rendering (False)
        """
        style = self._content_display_style(is_error=is_error, message_type=message_type)

        if isinstance(content, str):
            self._display_string_content(
                content,
                truncate=truncate,
                style=style,
                check_markdown_markers=check_markdown_markers,
                render_markdown=render_markdown,
            )
        elif isinstance(content, Text):
            self._display_text_content(
                content,
                truncate=truncate,
                style=style,
                render_markdown=render_markdown,
            )
        elif isinstance(content, Group) or is_renderable(content):
            console.console.print(content, markup=self._markup)
        elif isinstance(content, list):
            self._display_content_blocks(content, truncate=truncate, style=style)
        else:
            self._print_pretty(content, truncate=truncate, style=style)

    def _render_bottom_metadata(
        self,
        *,
        message_type: MessageType,
        bottom_metadata: list[str] | None,
        highlight_indexes: list[int] | None,
        max_item_length: int | None,
    ) -> None:
        """
        Render the bottom separator line with optional metadata.

        Args:
            message_type: The type of message being displayed
            bottom_metadata: Optional list of items to show in the separator
            highlight_indexes: Optional indexes of items to highlight
            max_item_length: Optional maximum length for individual items
        """
        if not bottom_metadata or not highlight_indexes:
            return
        valid_highlights = [
            index for index in highlight_indexes if 0 <= index < len(bottom_metadata)
        ]
        if not valid_highlights:
            return

        line = self._style.bottom_metadata_line(
            bottom_metadata,
            valid_highlights,
            MESSAGE_CONFIGS[message_type]["highlight_color"],
            max_item_length,
            console.console.size.width,
        )
        if line is None:
            return

        console.console.print()
        console.console.print(line, markup=self._markup)
        console.console.print()

    def show_tool_result(
        self,
        result: CallToolResult,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config: "SkybridgeServerConfig | None" = None,
        timing_ms: float | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        truncate_content: bool = True,
        show_hook_indicator: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {
            "name": name,
            "tool_name": tool_name,
            "skybridge_config": skybridge_config,
            "timing_ms": timing_ms,
            "tool_call_id": tool_call_id,
            "truncate_content": truncate_content,
            "show_hook_indicator": show_hook_indicator,
        }
        if type_label is not None:
            kwargs["type_label"] = type_label

        if not display_tools_enabled():
            return
        self._tool_display.show_tool_result(result, **kwargs)

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {
            "bottom_items": bottom_items,
            "highlight_indexes": highlight_indexes,
            "max_item_length": max_item_length,
            "name": name,
            "metadata": metadata,
            "tool_call_id": tool_call_id,
            "show_hook_indicator": show_hook_indicator,
        }
        if type_label is not None:
            kwargs["type_label"] = type_label

        if not display_tools_enabled():
            return
        self._tool_display.show_tool_call(tool_name, tool_args, **kwargs)

    async def show_tool_update(self, updated_server: str, agent_name: str | None = None) -> None:
        if not display_tools_enabled():
            return
        await self._tool_display.show_tool_update(updated_server, agent_name=agent_name)

    def _create_combined_separator_status(
        self,
        left_content: str,
        right_info: str = "",
        *,
        rule_fill: bool = False,
    ) -> None:
        """
        Create a combined separator and status line.

        Args:
            left_content: The main content (block, arrow, name) - left justified with color
            right_info: Supplementary information to show in brackets - right aligned
            rule_fill: Whether to fill remaining header space with a dim rule
        """
        combined = self._format_header_line(left_content, right_info, rule_fill=rule_fill)

        console.console.print()
        console.console.print(combined, markup=self._markup)
        for _ in range(self._style.header_spacing_after):
            console.console.print()

    @staticmethod
    def summarize_skybridge_configs(
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return ToolDisplay.summarize_skybridge_configs(configs)

    def show_skybridge_summary(
        self,
        agent_name: str,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> None:
        self._tool_display.show_skybridge_summary(agent_name, configs)

    def _extract_reasoning_content(self, message: "PromptMessageExtended") -> Text | Group | None:
        """Extract reasoning channel content as dim text."""
        channels = message.channels or {}
        reasoning_blocks = channels.get(REASONING) or []
        if not reasoning_blocks:
            return None

        from fast_agent.mcp.helpers.content_helpers import get_text

        reasoning_segments = []
        for block in reasoning_blocks:
            text = get_text(block)
            if text:
                reasoning_segments.append(text)

        if not reasoning_segments:
            return None

        joined = "\n".join(reasoning_segments)
        if not joined.strip():
            return None

        # Render reasoning in dim italic. Spacing between reasoning and main
        # content is handled by display_message() so reasoning-only turns don't
        # emit extra blank lines before the next header.
        joined = joined.rstrip("\n")
        if not joined.strip():
            return None

        if self._looks_like_markdown(joined):
            try:
                prepared = prepare_markdown_content(joined, self._escape_xml)
                markdown = Markdown(
                    prepared,
                    code_theme=self.code_style,
                    style="dim italic",
                )
                return Group(markdown)
            except Exception as exc:
                logger.exception(
                    "Failed to render reasoning markdown",
                    data={"error": str(exc)},
                )

        return Text(joined, style="dim italic")

    def _render_markdown_body(
        self,
        text: str,
        *,
        cursor_suffix: str = "",
        close_incomplete_fences: bool = False,
    ) -> RenderableType:
        from fast_agent.ui.markdown_renderables import build_markdown_renderable

        return build_markdown_renderable(
            text,
            code_theme=self.code_style,
            escape_xml=self._escape_xml,
            cursor_suffix=cursor_suffix,
            close_incomplete_fences=close_incomplete_fences,
            render_fences_with_syntax=self.render_fences_with_syntax,
            code_word_wrap=self.code_word_wrap,
        )

    @staticmethod
    def _openai_phase_raw_blocks(message: "PromptMessageExtended") -> list[Any]:
        channels = message.channels or {}
        if not isinstance(channels, Mapping):
            return []
        raw_blocks = channels.get(OPENAI_ASSISTANT_MESSAGE_ITEMS)
        return raw_blocks if isinstance(raw_blocks, list) else []

    @staticmethod
    def _openai_phase_payload(raw_text: object) -> Mapping[str, Any] | None:
        if not isinstance(raw_text, str) or not raw_text:
            return None
        try:
            payload = json.loads(raw_text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping) or payload.get("type") != "message":
            return None
        return payload

    @staticmethod
    def _openai_phase_text(content: object) -> str:
        if not isinstance(content, list):
            return ""

        text_segments: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            mapped_part = cast("Mapping[str, Any]", part)
            part_type = mapped_part.get("type")
            part_text = mapped_part.get("text")
            if part_type in {"output_text", "text"} and isinstance(part_text, str):
                text_segments.append(part_text)
        return "".join(text_segments).strip()

    @classmethod
    def _openai_phase_section_from_block(cls, block: Any) -> OpenAIPhaseSection | None:
        if not isinstance(block, _TextPayloadBlock):
            return None
        payload = cls._openai_phase_payload(block.text)
        if payload is None:
            return None

        section_text = cls._openai_phase_text(payload.get("content"))
        if not section_text:
            return None

        return OpenAIPhaseSection(
            phase=coerce_assistant_message_phase(payload.get("phase")),
            text=section_text,
        )

    @classmethod
    def _openai_phase_sections(
        cls,
        message: "PromptMessageExtended",
    ) -> list[OpenAIPhaseSection]:
        raw_blocks = cls._openai_phase_raw_blocks(message)
        sections: list[OpenAIPhaseSection] = []
        for block in raw_blocks:
            section = cls._openai_phase_section_from_block(block)
            if section is not None:
                sections.append(section)
        return sections

    def _render_openai_phase_section(
        self,
        section: OpenAIPhaseSection,
    ) -> str | Group | Text:
        if section.phase is None:
            return section.text

        phase_label = PHASE_LABELS.get(section.phase, section.phase)
        label = Text()
        label.append("▎", style="green")
        label.append(phase_label, style="green")
        if self._looks_like_markdown(section.text):
            return Group(label, self._render_markdown_body(section.text))

        label.append(" ")
        label.append(section.text)
        return label

    @staticmethod
    def _combine_openai_phase_renderables(
        sections: list[str | Group | Text],
    ) -> str | Group:
        if all(isinstance(section, str) for section in sections):
            return "\n\n".join(section for section in sections if isinstance(section, str))

        renderables: list[str | Group | Text] = []
        for index, section in enumerate(sections):
            if index:
                renderables.append(Text("\n"))
            renderables.append(section)
        return Group(*renderables)

    def _extract_openai_phase_content(
        self,
        message: "PromptMessageExtended",
    ) -> str | Group | None:
        sections = self._openai_phase_sections(message)
        if not sections or not any(section.phase is not None for section in sections):
            return None

        rendered_sections = [self._render_openai_phase_section(section) for section in sections]
        return self._combine_openai_phase_renderables(rendered_sections)

    async def show_assistant_message(
        self,
        message_text: "str | Text | PromptMessageExtended",
        bottom_items: list[str] | None = None,
        highlight_indexes: list[int] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        pre_content: Text | Group | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
        show_reprint_banner: bool = False,
    ) -> None:
        """Display an assistant message in a formatted panel.

        Args:
            message_text: The message content to display (str, Text, or PromptMessageExtended)
            bottom_items: Optional list of items for bottom separator (e.g., servers, destinations)
            highlight_indexes: Indexes of items to highlight in the bottom separator
            max_item_length: Optional max length for bottom items (with ellipsis)
            title: Title for the message (default "ASSISTANT")
            name: Optional agent name
            model: Optional model name for right info
            additional_message: Optional additional styled message to append
            pre_content: Optional additional styled message to prepend before body text
            render_markdown: Force markdown rendering (True) or plain rendering (False)
            show_hook_indicator: Whether to show the hook indicator glyph (◆)
            show_reprint_banner: Whether to emit the bright reprint banner for this message
        """
        if not self._chat_output_enabled():
            return

        # Extract text from PromptMessageExtended if needed
        from fast_agent.types import PromptMessageExtended

        resolved_pre_content = pre_content
        post_content: RenderableType | None = None

        if isinstance(message_text, PromptMessageExtended):
            # Prefer full assistant text so streamed/finalized multi-block responses
            # (e.g., provider-side web tool turns) are preserved after live refresh.
            display_text = (
                self._extract_openai_phase_content(message_text)
                or message_text.all_text()
                or message_text.last_text()
                or ""
            )
            resolved_pre_content = self._merge_pre_content(
                self._extract_reasoning_content(message_text),
                pre_content,
            )
            from fast_agent.ui.terminal_images import render_assistant_images_for_settings

            post_content = render_assistant_images_for_settings(
                self.terminal_image_settings,
                message_text,
            )
        else:
            display_text = message_text

        display_text = self._normalize_assistant_display_text(display_text)

        # Build right info
        display_model = resolve_model_display_name(model)
        right_info = f"[dim]{display_model}[/dim]" if display_model else ""

        if show_reprint_banner:
            self.show_stream_reprint_banner()

        # Display main message using unified method
        self.display_message(
            content=display_text,
            message_type=MessageType.ASSISTANT,
            name=name,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_indexes=highlight_indexes,
            max_item_length=max_item_length,
            truncate_content=False,  # Assistant messages shouldn't be truncated
            additional_message=additional_message,
            pre_content=resolved_pre_content,
            post_content=post_content,
            render_markdown=render_markdown,
            show_hook_indicator=show_hook_indicator,
        )

        # Handle mermaid diagrams separately (after the main message)
        self.show_mermaid_diagrams_from_message_text(message_text)

    @staticmethod
    def _normalize_assistant_display_text(content: object) -> object:
        if isinstance(content, str):
            return content.rstrip()
        if isinstance(content, Text):
            normalized = content.copy()
            normalized.rstrip()
            return normalized
        return content

    @staticmethod
    def _merge_pre_content(
        primary: Text | Group | None,
        secondary: Text | Group | None,
    ) -> Text | Group | None:
        if primary is None:
            return secondary
        if secondary is None:
            return primary
        return Group(primary, Text("\n\n"), secondary)

    def show_mermaid_diagrams_from_message_text(
        self,
        message_text: "str | Text | PromptMessageExtended",
    ) -> None:
        """Display mermaid links extracted from assistant text payload."""
        if not self._chat_output_enabled():
            return
        from fast_agent.types import PromptMessageExtended

        plain_text = ""
        if isinstance(message_text, PromptMessageExtended):
            plain_text = message_text.all_text() or message_text.last_text() or ""
        elif isinstance(message_text, Text):
            plain_text = message_text.plain
        elif isinstance(message_text, str):
            plain_text = message_text

        diagrams = extract_mermaid_diagrams(plain_text)
        if diagrams:
            self._display_mermaid_diagrams(diagrams)

    @contextmanager
    def streaming_assistant_message(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        show_hook_indicator: bool = False,
        tool_metadata_resolver: Callable[[str], Mapping[str, Any] | None] | None = None,
    ) -> Iterator[StreamingHandle]:
        """Create a streaming context for assistant messages."""
        if not self._chat_output_enabled():
            yield _NullStreamingHandle()
            return
        streaming_preferences = self.resolve_streaming_preferences()

        if not streaming_preferences.enabled:
            yield _NullStreamingHandle()
            return

        from fast_agent.ui.progress_display import progress_display

        config = MESSAGE_CONFIGS[MessageType.ASSISTANT]
        block_color = config["block_color"]
        arrow = config["arrow"]
        arrow_style = config["arrow_style"]

        left = self.build_header_left(
            block_color=block_color,
            arrow=arrow,
            arrow_style=arrow_style,
            name=name,
            is_error=False,
            show_hook_indicator=show_hook_indicator,
        )

        display_model = resolve_model_display_name(model)
        right_info = f"[dim]{display_model}[/dim]" if display_model else ""

        # Determine renderer based on streaming mode
        use_plain_text = streaming_preferences.mode == "plain"

        handle = _StreamingMessageHandle(
            display=self,
            use_plain_text=use_plain_text,
            header_left=left,
            header_right=right_info,
            tool_header_name=name,
            tool_metadata_resolver=tool_metadata_resolver,
            progress_display=progress_display,
        )
        try:
            yield handle
        finally:
            handle.close()

    def _display_mermaid_diagrams(self, diagrams: list[MermaidDiagram]) -> None:
        """Display mermaid diagram links."""
        diagram_content = Text()
        # Add bullet at the beginning
        diagram_content.append("● ", style="dim")

        for i, diagram in enumerate(diagrams, 1):
            if i > 1:
                diagram_content.append(" • ", style="dim")

            # Generate URL
            url = create_mermaid_live_link(diagram.content)

            # Format: "1 - Title" or "1 - Flowchart" or "Diagram 1"
            if diagram.title:
                diagram_content.append(f"{i} - {diagram.title}", style=f"bright_blue link {url}")
            else:
                # Try to detect diagram type, fallback to "Diagram N"
                diagram_type = detect_diagram_type(diagram.content)
                if diagram_type != "Diagram":
                    diagram_content.append(f"{i} - {diagram_type}", style=f"bright_blue link {url}")
                else:
                    diagram_content.append(f"Diagram {i}", style=f"bright_blue link {url}")

        # Display diagrams on a simple new line (more space efficient)
        console.console.print()
        console.console.print(diagram_content, markup=self._markup)

    async def show_mcp_ui_links(self, links: list[UILink]) -> None:
        """Display MCP-UI links beneath the chat like mermaid links."""
        if not self._chat_output_enabled():
            return

        if not links:
            return

        content = Text()
        content.append("● mcp-ui ", style="dim")
        for i, link in enumerate(links, 1):
            if i > 1:
                content.append(" • ", style="dim")
            # Prefer a web-friendly URL (http(s) or data:) if available; fallback to local file
            url = link.web_url if getattr(link, "web_url", None) else f"file://{link.file_path}"
            label = f"{i} - {link.title}"
            content.append(label, style=f"bright_blue link {url}")

        console.console.print()
        console.console.print(content, markup=self._markup)

    def show_url_elicitation(
        self,
        message: str,
        url: str,
        server_name: str,
        agent_name: str | None = None,
        elicitation_id: str | None = None,
    ) -> None:
        """Display URL elicitation request with clickable link.

        Compact format similar to mermaid diagram links, while maintaining
        security visibility (server name, domain, full URL).

        Args:
            message: The server's message explaining why navigation is needed
            url: The URL the server wants the user to navigate to
            server_name: Name of the MCP server making the request
            agent_name: Optional name of the agent
            elicitation_id: Optional URL elicitation ID
        """
        del agent_name
        if not self._chat_output_enabled():
            return

        from urllib.parse import urlparse

        console.configure_console_stream("stdout")

        # Extract domain for security display
        parsed = urlparse(url)
        domain = parsed.netloc or url  # Fallback to full URL if no domain

        console.console.print()

        # Line 1: prominent requirement indicator
        header = Text()
        header.append("● ", style="bright_yellow bold")
        header.append("URL elicitation required", style="bright_yellow bold")
        console.console.print(header, markup=self._markup)

        # Line 2: server + message
        message_line = Text()
        message_line.append("  ", style="dim")
        message_line.append(f"[{server_name}] ", style="cyan bold")
        message_line.append(message, style="bold")
        console.console.print(message_line, markup=self._markup)

        # Line 3: elicitation ID (if present)
        if elicitation_id:
            id_line = Text()
            id_line.append("  ", style="dim")
            id_line.append(f"elicitationId: {elicitation_id}", style="dim")
            console.console.print(id_line, markup=self._markup)

        # Line 4: domain (highlighted) + full URL (dim)
        url_line = Text()
        url_line.append("  ", style="dim")
        url_line.append(domain, style="yellow bold")
        url_line.append(" → ", style="dim")
        url_line.append(url, style="dim")
        console.console.print(url_line, markup=self._markup)

        # Line 5: clickable link
        link_line = Text()
        link_line.append("  ", style="dim")
        link_line.append("Open URL", style=f"bright_blue link {url}")
        console.console.print(link_line, markup=self._markup)

    def show_user_message(
        self,
        message: str | Text,
        model: str | None = None,
        chat_turn: int = 0,
        total_turns: int | None = None,
        turn_range: tuple[int, int] | None = None,
        name: str | None = None,
        attachments: list[str] | None = None,
        image_previews: list["ImageRenderItem"] | None = None,
        part_count: int | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        """Display a user message in the new visual style."""
        if not self._chat_output_enabled():
            return

        _ = model

        self.display_message(
            content=message,
            message_type=MessageType.USER,
            name=name,
            right_info=self._user_message_right_info(
                chat_turn=chat_turn,
                total_turns=total_turns,
                turn_range=turn_range,
                part_count=part_count,
            ),
            truncate_content=False,  # User messages typically shouldn't be truncated
            pre_content=self._user_message_pre_content(
                attachments=attachments,
                image_previews=image_previews,
            ),
            show_hook_indicator=show_hook_indicator,
            header_rule_fill=True,
        )

    def show_system_message(
        self,
        system_prompt: str,
        agent_name: str | None = None,
        server_count: int = 0,
    ) -> None:
        """Display the system prompt in a formatted panel."""
        if not self._chat_output_enabled():
            return

        # Build right side info
        right_parts = []
        if server_count > 0:
            right_parts.append(f"{format_count(server_count, 'MCP server')}")

        right_info = f"[dim]{' '.join(right_parts)}[/dim]" if right_parts else ""

        self.display_message(
            content=system_prompt,
            message_type=MessageType.SYSTEM,
            name=agent_name,
            right_info=right_info,
            truncate_content=False,  # Don't truncate system prompts
        )

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: str | None = None,
        message_count: int = 0,
        agent_name: str | None = None,
        server_list: list[str] | None = None,
        highlight_server: str | None = None,
        arguments: dict[str, str] | None = None,
    ) -> None:
        """
        Display information about a loaded prompt template.

        Args:
            prompt_name: The name of the prompt that was loaded
            description: Optional description of the prompt
            message_count: Number of messages added to the conversation history
            agent_name: Name of the agent using the prompt
            server_list: Optional list of servers to display
            highlight_server: Optional server name to highlight
            arguments: Optional dictionary of arguments passed to the prompt template
        """
        if not self._logger_settings.show_tools:
            return

        # Build the server list with highlighting
        display_server_list = Text()
        if server_list:
            for server_name in server_list:
                style = "green" if server_name == highlight_server else "dim white"
                display_server_list.append(f"[{server_name}] ", style)

        # Create content text
        content = Text()
        messages_phrase = f"Loaded {format_count(message_count, 'message')}"
        content.append(f"{messages_phrase} from template ", style="cyan italic")
        content.append(f"'{prompt_name}'", style="cyan bold italic")

        if agent_name:
            content.append(f" for {agent_name}", style="cyan italic")

        # Add template arguments if provided
        if arguments:
            content.append("\n\nArguments:", style="cyan")
            for key, value in arguments.items():
                content.append(f"\n  {key}: ", style="cyan bold")
                content.append(value, style="white")

        if description:
            content.append("\n\n", style="default")
            content.append(description, style="dim white")

        # Create panel
        panel = Panel(
            content,
            title="[PROMPT LOADED]",
            title_align="right",
            style="cyan",
            border_style="white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    @staticmethod
    def _parallel_fan_out_agents(parallel_agent: Any) -> list[Any]:
        if parallel_agent is None:
            return []
        try:
            return list(parallel_agent.fan_out_agents)
        except AttributeError:
            return []

    @staticmethod
    def _parallel_agent_usage(agent: Any) -> tuple[int, int]:
        accumulator = agent.usage_accumulator
        if not accumulator:
            return 0, 0
        summary = accumulator.summary
        return summary.total or 0, summary.tool_calls

    @classmethod
    def _parallel_agent_result(cls, agent: Any) -> ParallelAgentDisplayResult | None:
        message_history = agent.message_history
        if not message_history:
            return None

        model = "unknown"
        if agent.llm:
            model = resolve_llm_display_name(agent.llm) or "unknown"

        tokens, tool_calls = cls._parallel_agent_usage(agent)
        return ParallelAgentDisplayResult(
            name=agent.name,
            model=model,
            content=message_history[-1].last_text(),
            tokens=tokens,
            tool_calls=tool_calls,
        )

    @classmethod
    def _parallel_agent_results(cls, parallel_agent: Any) -> list[ParallelAgentDisplayResult]:
        results: list[ParallelAgentDisplayResult] = []
        for agent in cls._parallel_fan_out_agents(parallel_agent):
            result = cls._parallel_agent_result(agent)
            if result is not None:
                results.append(result)
        return results

    @staticmethod
    def _parallel_result_usage_label(result: ParallelAgentDisplayResult) -> str:
        right_parts: list[str] = []
        if result.tokens > 0:
            right_parts.append(f"{result.tokens:,} tokens")
        if result.tool_calls > 0:
            right_parts.append(format_count(result.tool_calls, "tool"))
        return " • ".join(right_parts) if right_parts else "no usage data"

    def _show_parallel_result_header(self, result: ParallelAgentDisplayResult) -> None:
        left_text = Text()
        left_text.append("▎", style="green")
        left_text.append(" ")
        left_text.append(result.model, style="bold green")
        right_text = Text(self._parallel_result_usage_label(result), style="dim")
        padding = max(1, console.console.size.width - left_text.cell_len - right_text.cell_len)
        console.console.print(Text.assemble(left_text, " " * padding, right_text))
        console.console.print()

    def _show_parallel_result(self, result: ParallelAgentDisplayResult, *, index: int) -> None:
        if index > 0:
            console.console.print()
            console.console.print("─" * console.console.size.width, style="dim")
            console.console.print()

        self._show_parallel_result_header(result)
        self._display_content(
            result.content,
            truncate=False,
            is_error=False,
            message_type=MessageType.ASSISTANT,
            check_markdown_markers=True,
        )

    @staticmethod
    def _parallel_summary_text(agent_results: list[ParallelAgentDisplayResult]) -> str:
        total_tokens = sum(result.tokens for result in agent_results)
        total_tools = sum(result.tool_calls for result in agent_results)

        summary_parts = [format_count(len(agent_results), "model")]
        if total_tokens > 0:
            summary_parts.append(f"{total_tokens:,} tokens")
        if total_tools > 0:
            summary_parts.append(format_count(total_tools, "tool"))
        return " • ".join(summary_parts)

    def show_parallel_results(self, parallel_agent: Any) -> None:
        """Display parallel agent results in a clean, organized format."""
        if not self._chat_output_enabled():
            return

        agent_results = self._parallel_agent_results(parallel_agent)
        if not agent_results:
            return

        console.console.print()
        console.console.print("[dim]Parallel execution complete[/dim]")
        console.console.print()

        for index, result in enumerate(agent_results):
            self._show_parallel_result(result, index=index)

        console.console.print()
        console.console.print("─" * console.console.size.width, style="dim")
        console.console.print(Text(self._parallel_summary_text(agent_results), style="dim"))
        console.console.print()
