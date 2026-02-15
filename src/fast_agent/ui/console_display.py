from contextlib import contextmanager
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Union

from mcp.types import CallToolResult
from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.text import Text

from fast_agent.config import LoggerSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.mcp_ui_utils import UILink
from fast_agent.ui.mermaid_utils import (
    MermaidDiagram,
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
)
from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType
from fast_agent.ui.message_styles import MessageStyle, resolve_message_style
from fast_agent.ui.model_display import format_model_display
from fast_agent.ui.streaming import (
    NullStreamingHandle as _NullStreamingHandle,
)
from fast_agent.ui.streaming import (
    StreamingHandle,
)
from fast_agent.ui.streaming import (
    StreamingMessageHandle as _StreamingMessageHandle,
)
from fast_agent.ui.tool_display import ToolDisplay
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.mcp.skybridge import SkybridgeServerConfig

logger = get_logger(__name__)

CODE_STYLE = "native"

# Glyph to indicate tool hooks are active
HOOK_INDICATOR_GLYPH = "◆"


class ConsoleDisplay:
    """
    Handles displaying formatted messages, tool calls, and results to the console.
    This centralizes the UI display logic used by LLM implementations.
    """

    CODE_STYLE = CODE_STYLE

    def __init__(self, config: Settings | None = None) -> None:
        """
        Initialize the console display handler.

        Args:
            config: Configuration object containing display preferences
        """
        self.config = config
        self._logger_settings = self._resolve_logger_settings(config)
        if self.config and not getattr(self.config, "logger", None):
            # Ensure callers passing in a bare namespace still get sane defaults
            try:
                setattr(self.config, "logger", self._logger_settings)
            except Exception:
                pass
        self._markup = getattr(self._logger_settings, "enable_markup", True)
        self._escape_xml = True
        self._style = resolve_message_style(
            getattr(self._logger_settings, "message_style", "a3")
        )
        self._tool_display = ToolDisplay(self)

    @staticmethod
    def _resolve_logger_settings(config: Settings | None) -> LoggerSettings:
        """Provide a logger settings object even when callers omit it."""
        logger_settings = getattr(config, "logger", None) if config else None
        return logger_settings if logger_settings is not None else LoggerSettings()

    def _truncate_text(self, text: str, *, truncate: bool) -> str:
        if (
            truncate
            and self.config
            and self.config.logger.truncate_tools
            and len(text) > 360
        ):
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

        if truncate and self.config and self.config.logger.truncate_tools:
            pretty_obj = Pretty(content, max_length=10, max_string=50)
        else:
            pretty_obj = Pretty(content)
        self._print_with_style(pretty_obj, style=style)

    @property
    def code_style(self) -> str:
        return CODE_STYLE

    @property
    def style(self) -> MessageStyle:
        return self._style

    def show_status_message(self, content: Text) -> None:
        """Display a status message without a header."""
        console.ensure_blocking_console()
        console.console.print(content, markup=self._markup)

    def resolve_streaming_preferences(self) -> tuple[bool, str]:
        """Return whether streaming is enabled plus the active mode."""
        if not self.config:
            return True, "markdown"

        logger_settings = getattr(self.config, "logger", None)
        if not logger_settings:
            return True, "markdown"

        streaming_mode = getattr(logger_settings, "streaming", "markdown")
        if streaming_mode not in {"markdown", "plain", "none"}:
            streaming_mode = "markdown"

        # Legacy compatibility: allow streaming_plain_text override
        if streaming_mode == "markdown" and getattr(
            logger_settings, "streaming_plain_text", False
        ):
            streaming_mode = "plain"

        show_chat = bool(getattr(logger_settings, "show_chat", True))
        streaming_display = bool(getattr(logger_settings, "streaming_display", True))

        enabled = show_chat and streaming_display and streaming_mode != "none"
        return enabled, streaming_mode

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

    def show_shell_exit_code(self, exit_code: int) -> None:
        """Display a shell-style exit code banner."""
        line = self._style.shell_exit_line(exit_code, console.console.size.width)
        console.console.print()
        console.console.print(line)
        for _ in range(self._style.shell_exit_spacing_after):
            console.console.print()

    def _format_header_line(self, left_content: str, right_info: str = "") -> Text:
        width = console.console.size.width
        return self._style.header_line(left_content, right_info, width)

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
        left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}]"
        if show_hook_indicator:
            left += f" [{block_color}]{HOOK_INDICATOR_GLYPH}[/{block_color}]"
        if name:
            name_color = block_color if not is_error else "red"
            left += f" [{name_color}]{name}[/{name_color}]"
        return left

    def display_message(
        self,
        content: Any,
        message_type: MessageType,
        name: str | None = None,
        right_info: str = "",
        bottom_metadata: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        is_error: bool = False,
        truncate_content: bool = True,
        additional_message: Text | None = None,
        pre_content: Text | Group | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        """
        Unified method to display formatted messages to the console.

        Args:
            content: The main content to display (str, Text, JSON, etc.)
            message_type: Type of message (USER, ASSISTANT, TOOL_CALL, TOOL_RESULT)
            name: Optional name to display (agent name, user name, etc.)
            right_info: Information to display on the right side of the header
            bottom_metadata: Optional list of items for bottom separator
            highlight_index: Index of item to highlight in bottom metadata (0-based), or None
            max_item_length: Optional max length for bottom metadata items (with ellipsis)
            is_error: For tool results, whether this is an error (uses red color)
            truncate_content: Whether to truncate long content
            additional_message: Optional Rich Text appended after the main content
            pre_content: Optional Rich Text shown before the main content
            render_markdown: Force markdown rendering (True) or plain rendering (False)
            show_hook_indicator: Whether to show the hook indicator glyph (◆)
        """
        # Ensure Rich writes to a blocking TTY when stdout/stderr was
        # flipped to non-blocking by the event loop (e.g. uvloop).
        console.ensure_blocking_console()

        # Get configuration for this message type
        config = MESSAGE_CONFIGS[message_type]

        # Override colors for error states
        if is_error and message_type == MessageType.TOOL_RESULT:
            block_color = "red"
        else:
            block_color = config["block_color"]

        # Build the left side of the header
        arrow = config["arrow"]
        arrow_style = config["arrow_style"]
        left = self.build_header_left(
            block_color=block_color,
            arrow=arrow,
            arrow_style=arrow_style,
            name=name,
            is_error=is_error,
            show_hook_indicator=show_hook_indicator,
        )

        # Create combined separator and status line
        self._create_combined_separator_status(left, right_info)

        # Display the content
        if pre_content:
            if isinstance(pre_content, Text):
                if pre_content.plain:
                    console.console.print(pre_content, markup=self._markup)
            else:
                console.console.print(pre_content, markup=self._markup)
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

        # Handle bottom separator with optional metadata
        self._render_bottom_metadata(
            message_type=message_type,
            bottom_metadata=bottom_metadata,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
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
        import json
        import re

        from rich.syntax import Syntax

        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        # Determine the style based on message type
        # USER, ASSISTANT, and SYSTEM messages should display in normal style
        # TOOL_CALL and TOOL_RESULT should be dimmed
        if is_error:
            style = "dim red"
        elif message_type in [MessageType.USER, MessageType.ASSISTANT, MessageType.SYSTEM]:
            style = None  # No style means default/normal white
        else:
            style = "dim"

        # Handle different content types
        if isinstance(content, str):
            if render_markdown is not None:
                try:
                    json_obj = json.loads(content)
                    self._print_pretty(json_obj, truncate=truncate, style=style)
                    return
                except (JSONDecodeError, TypeError, ValueError):
                    if render_markdown:
                        prepared_content = prepare_markdown_content(content, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        console.console.print(md, markup=self._markup)
                    else:
                        self._print_plain_text(content, truncate=truncate, style=style)
                    return
            # Try to detect and handle different string formats
            try:
                # Try as JSON first
                json_obj = json.loads(content)
                self._print_pretty(json_obj, truncate=truncate, style=style)
            except (JSONDecodeError, TypeError, ValueError):
                # Check if content appears to be primarily XML
                xml_pattern = r"^<[a-zA-Z_][a-zA-Z0-9_-]*[^>]*>"
                is_xml_content = (
                    bool(re.match(xml_pattern, content.strip())) and content.count("<") > 5
                )

                if is_xml_content:
                    # Display XML content with syntax highlighting for better readability
                    syntax = Syntax(content, "xml", theme=CODE_STYLE, line_numbers=False)
                    console.console.print(syntax, markup=self._markup)
                elif check_markdown_markers:
                    # Check for markdown markers before deciding to use markdown rendering
                    if self._looks_like_markdown(content):
                        # Has markdown markers - render as markdown with escaping
                        prepared_content = prepare_markdown_content(content, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        console.console.print(md, markup=self._markup)
                    else:
                        # Plain text - display as-is
                        self._print_plain_text(content, truncate=truncate, style=style)
                else:
                    # Check if content has substantial XML (mixed content)
                    # If so, skip markdown rendering as it turns XML into an unreadable blob.
                    # Ignore markdown autolinks like <https://...>.
                    xml_probe = re.sub(r"<(?:https?://|mailto:)[^>]+>", "", content)
                    has_substantial_xml = xml_probe.count("<") > 5 and xml_probe.count(">") > 5

                    # Check if it looks like markdown
                    if self._looks_like_markdown(content) and not has_substantial_xml:
                        # Escape HTML/XML tags while preserving code blocks
                        prepared_content = prepare_markdown_content(content, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        # Markdown handles its own styling, don't apply style
                        console.console.print(md, markup=self._markup)
                    else:
                        # Plain text (or mixed markdown+XML content)
                        self._print_plain_text(content, truncate=truncate, style=style)
        elif isinstance(content, Text):
            if render_markdown is not None:
                plain_text = content.plain
                try:
                    json_obj = json.loads(plain_text)
                    self._print_pretty(json_obj, truncate=truncate, style=style)
                    return
                except (JSONDecodeError, TypeError, ValueError):
                    if render_markdown:
                        prepared_content = prepare_markdown_content(plain_text, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        console.console.print(md, markup=self._markup)
                    else:
                        console.console.print(content, markup=self._markup)
                    return
            # Rich Text object - check if it contains markdown
            plain_text = content.plain

            # Check if the plain text contains markdown markers
            if self._looks_like_markdown(plain_text):
                # Split the Text object into segments
                # We need to handle the main content (which may have markdown)
                # and any styled segments that were appended

                # If the Text object has multiple spans with different styles,
                # we need to be careful about how we render them
                if len(content._spans) > 1:
                    # Complex case: Text has multiple styled segments
                    # We'll render the first part as markdown if it contains markers
                    # and append other styled parts separately

                    # Find where the markdown content ends (usually the first span)
                    markdown_end = content._spans[0].end if content._spans else len(plain_text)
                    markdown_part = plain_text[:markdown_end]

                    # Check if the first part has markdown
                    if self._looks_like_markdown(markdown_part):
                        # Render markdown part
                        prepared_content = prepare_markdown_content(markdown_part, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        console.console.print(md, markup=self._markup)

                        # Then render any additional styled segments
                        if markdown_end < len(plain_text):
                            remaining_text = Text()
                            for span in content._spans:
                                if span.start >= markdown_end:
                                    segment_text = plain_text[span.start : span.end]
                                    remaining_text.append(segment_text, style=span.style)
                            if remaining_text.plain:
                                console.console.print(remaining_text, markup=self._markup)
                    else:
                        # No markdown in first part, just print the whole Text object
                        console.console.print(content, markup=self._markup)
                else:
                    # Simple case: entire text should be rendered as markdown
                    prepared_content = prepare_markdown_content(plain_text, self._escape_xml)
                    md = Markdown(prepared_content, code_theme=CODE_STYLE)
                    console.console.print(md, markup=self._markup)
            else:
                # No markdown markers, print as regular Rich Text
                console.console.print(content, markup=self._markup)
        elif isinstance(content, list):
            # Handle content blocks (for tool results)
            if len(content) == 1 and is_text_content(content[0]):
                # Single text block - display directly
                text_content = get_text(content[0])
                if text_content:
                    self._print_plain_text(text_content, truncate=truncate, style=style)
                else:
                    # Apply style only if specified
                    if style:
                        console.console.print("(empty text)", style=style, markup=self._markup)
                    else:
                        console.console.print("(empty text)", markup=self._markup)
            else:
                # Multiple blocks or non-text content
                self._print_pretty(content, truncate=truncate, style=style)
        else:
            # Any other type - use Pretty
            self._print_pretty(content, truncate=truncate, style=style)

    def _render_bottom_metadata(
        self,
        *,
        message_type: MessageType,
        bottom_metadata: list[str] | None,
        highlight_index: int | None,
        max_item_length: int | None,
    ) -> None:
        """
        Render the bottom separator line with optional metadata.

        Args:
            message_type: The type of message being displayed
            bottom_metadata: Optional list of items to show in the separator
            highlight_index: Optional index of the item to highlight
            max_item_length: Optional maximum length for individual items
        """
        if self._style.bottom_metadata_requires_highlight:
            if not bottom_metadata or highlight_index is None:
                return
            if highlight_index < 0 or highlight_index >= len(bottom_metadata):
                return

        line = self._style.bottom_metadata_line(
            bottom_metadata,
            highlight_index,
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

        self._tool_display.show_tool_result(result, **kwargs)

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        type_label: str | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {
            "bottom_items": bottom_items,
            "highlight_index": highlight_index,
            "max_item_length": max_item_length,
            "name": name,
            "metadata": metadata,
            "tool_call_id": tool_call_id,
            "show_hook_indicator": show_hook_indicator,
        }
        if type_label is not None:
            kwargs["type_label"] = type_label

        self._tool_display.show_tool_call(tool_name, tool_args, **kwargs)

    async def show_tool_update(self, updated_server: str, agent_name: str | None = None) -> None:
        await self._tool_display.show_tool_update(updated_server, agent_name=agent_name)

    def _create_combined_separator_status(self, left_content: str, right_info: str = "") -> None:
        """
        Create a combined separator and status line.

        Args:
            left_content: The main content (block, arrow, name) - left justified with color
            right_info: Supplementary information to show in brackets - right aligned
        """
        combined = self._format_header_line(left_content, right_info)

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

        # Render reasoning in dim italic and leave a blank line before main content
        if self._looks_like_markdown(joined):
            try:
                prepared = prepare_markdown_content(joined, self._escape_xml)
                markdown = Markdown(
                    prepared,
                    code_theme=self.code_style,
                    style="dim italic",
                )
                return Group(markdown, Text("\n"))
            except Exception as exc:
                logger.exception(
                    "Failed to render reasoning markdown",
                    data={"error": str(exc)},
                )

        text = joined
        if not text.endswith("\n"):
            text += "\n"
        return Text(text, style="dim italic")

    async def show_assistant_message(
        self,
        message_text: Union[str, Text, "PromptMessageExtended"],
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Text | None = None,
        render_markdown: bool | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        """Display an assistant message in a formatted panel.

        Args:
            message_text: The message content to display (str, Text, or PromptMessageExtended)
            bottom_items: Optional list of items for bottom separator (e.g., servers, destinations)
            highlight_index: Index of item to highlight in the bottom separator (0-based), or None
            max_item_length: Optional max length for bottom items (with ellipsis)
            title: Title for the message (default "ASSISTANT")
            name: Optional agent name
            model: Optional model name for right info
            additional_message: Optional additional styled message to append
            render_markdown: Force markdown rendering (True) or plain rendering (False)
            show_hook_indicator: Whether to show the hook indicator glyph (◆)
        """
        if self.config and not self.config.logger.show_chat:
            return

        # Extract text from PromptMessageExtended if needed
        from fast_agent.types import PromptMessageExtended

        pre_content: Text | Group | None = None

        if isinstance(message_text, PromptMessageExtended):
            display_text = message_text.last_text() or ""
            pre_content = self._extract_reasoning_content(message_text)
        else:
            display_text = message_text

        # Build right info
        display_model = format_model_display(model)
        right_info = f"[dim]{display_model}[/dim]" if display_model else ""

        # Display main message using unified method
        self.display_message(
            content=display_text,
            message_type=MessageType.ASSISTANT,
            name=name,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            truncate_content=False,  # Assistant messages shouldn't be truncated
            additional_message=additional_message,
            pre_content=pre_content,
            render_markdown=render_markdown,
            show_hook_indicator=show_hook_indicator,
        )

        # Handle mermaid diagrams separately (after the main message)
        # Extract plain text for mermaid detection
        plain_text = display_text
        if isinstance(display_text, Text):
            plain_text = display_text.plain

        if isinstance(plain_text, str):
            diagrams = extract_mermaid_diagrams(plain_text)
            if diagrams:
                self._display_mermaid_diagrams(diagrams)

    @contextmanager
    def streaming_assistant_message(
        self,
        *,
        bottom_items: list[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        show_hook_indicator: bool = False,
    ) -> Iterator[StreamingHandle]:
        """Create a streaming context for assistant messages."""
        streaming_enabled, streaming_mode = self.resolve_streaming_preferences()

        if not streaming_enabled:
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

        display_model = format_model_display(model)
        right_info = f"[dim]{display_model}[/dim]" if display_model else ""

        # Determine renderer based on streaming mode
        use_plain_text = streaming_mode == "plain"

        handle = _StreamingMessageHandle(
            display=self,
            bottom_items=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            use_plain_text=use_plain_text,
            header_left=left,
            header_right=right_info,
            tool_header_name=name,
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
        if self.config and not self.config.logger.show_chat:
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
            agent_name: Optional name of the agent (for future use)
            elicitation_id: Optional URL elicitation ID
        """
        if self.config and not self.config.logger.show_chat:
            return

        from urllib.parse import urlparse

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
        message: Union[str, Text],
        model: str | None = None,
        chat_turn: int = 0,
        total_turns: int | None = None,
        turn_range: tuple[int, int] | None = None,
        name: str | None = None,
        attachments: list[str] | None = None,
        part_count: int | None = None,
        show_hook_indicator: bool = False,
    ) -> None:
        """Display a user message in the new visual style."""
        if self.config and not self.config.logger.show_chat:
            return

        _ = model

        # Build right side with turn info and parts
        right_parts: list[str] = []

        turn_info = ""
        if part_count and part_count > 1:
            turn_number = 0
            if turn_range:
                turn_number = turn_range[0]
            elif chat_turn > 0:
                turn_number = chat_turn
            if turn_number > 0:
                turn_info = f"turn {turn_number}"
        elif turn_range:
            turn_start, turn_end = turn_range
            if total_turns:
                if turn_start == turn_end:
                    turn_info = f"turn {turn_start} ({total_turns})"
                else:
                    turn_info = f"turn {turn_start}-{turn_end} ({total_turns})"
            elif turn_start == turn_end:
                turn_info = f"turn {turn_start}"
            else:
                turn_info = f"turn {turn_start}-{turn_end}"
        elif chat_turn > 0:
            if total_turns:
                turn_info = f"turn {chat_turn} ({total_turns})"
            else:
                turn_info = f"turn {chat_turn}"

        if turn_info:
            right_parts.append(turn_info)

        if part_count and part_count > 1:
            right_parts.append(f"({part_count} parts)")

        right_info = f"[dim]{' '.join(right_parts)}[/dim]" if right_parts else ""

        # Build attachment indicator as pre_content
        pre_content: Text | Group | None = None
        if attachments:
            pre_content = Text()
            pre_content.append("🔗 ", style="dim")
            pre_content.append(", ".join(attachments), style="dim blue")

        self.display_message(
            content=message,
            message_type=MessageType.USER,
            name=name,
            right_info=right_info,
            truncate_content=False,  # User messages typically shouldn't be truncated
            pre_content=pre_content,
            show_hook_indicator=show_hook_indicator,
        )

    def show_system_message(
        self,
        system_prompt: str,
        agent_name: str | None = None,
        server_count: int = 0,
    ) -> None:
        """Display the system prompt in a formatted panel."""
        if self.config and not self.config.logger.show_chat:
            return

        # Build right side info
        right_parts = []
        if server_count > 0:
            server_word = "server" if server_count == 1 else "servers"
            right_parts.append(f"{server_count} MCP {server_word}")

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
        if self.config and not self.config.logger.show_tools:
            return

        # Build the server list with highlighting
        display_server_list = Text()
        if server_list:
            for server_name in server_list:
                style = "green" if server_name == highlight_server else "dim white"
                display_server_list.append(f"[{server_name}] ", style)

        # Create content text
        content = Text()
        messages_phrase = f"Loaded {message_count} message{'s' if message_count != 1 else ''}"
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

    def show_parallel_results(self, parallel_agent) -> None:
        """Display parallel agent results in a clean, organized format.

        Args:
            parallel_agent: The parallel agent containing fan_out_agents with results
        """

        from rich.text import Text

        if self.config and not self.config.logger.show_chat:
            return

        if not parallel_agent or not hasattr(parallel_agent, "fan_out_agents"):
            return

        # Collect results and agent information
        agent_results = []

        for agent in parallel_agent.fan_out_agents:
            # Get the last response text from this agent
            message_history = agent.message_history
            if not message_history:
                continue

            last_message = message_history[-1]
            content = last_message.last_text()

            # Get model name
            model = "unknown"
            if agent.llm:
                model = format_model_display(agent.llm.model_name) or "unknown"

            # Get usage information
            tokens = 0
            tool_calls = 0
            if hasattr(agent, "usage_accumulator") and agent.usage_accumulator:
                summary = agent.usage_accumulator.get_summary()
                tokens = summary.get("cumulative_input_tokens", 0) + summary.get(
                    "cumulative_output_tokens", 0
                )
                tool_calls = summary.get("cumulative_tool_calls", 0)

            agent_results.append(
                {
                    "name": agent.name,
                    "model": model,
                    "content": content,
                    "tokens": tokens,
                    "tool_calls": tool_calls,
                }
            )

        if not agent_results:
            return

        # Display header
        console.console.print()
        console.console.print("[dim]Parallel execution complete[/dim]")
        console.console.print()

        # Display results for each agent
        for i, result in enumerate(agent_results):
            if i > 0:
                # Simple full-width separator
                console.console.print()
                console.console.print("─" * console.console.size.width, style="dim")
                console.console.print()

            # Two column header: model name (green) + usage info (dim)
            left = f"[green]▎[/green] [bold green]{result['model']}[/bold green]"

            # Build right side with tokens and tool calls if available
            right_parts = []
            if result["tokens"] > 0:
                right_parts.append(f"{result['tokens']:,} tokens")
            if result["tool_calls"] > 0:
                right_parts.append(f"{result['tool_calls']} tools")

            right = f"[dim]{' • '.join(right_parts) if right_parts else 'no usage data'}[/dim]"

            # Calculate padding to right-align usage info
            width = console.console.size.width
            left_text = Text.from_markup(left)
            right_text = Text.from_markup(right)
            padding = max(1, width - left_text.cell_len - right_text.cell_len)

            console.console.print(left + " " * padding + right, markup=self._markup)
            console.console.print()

            # Display content based on its type (check for markdown markers in parallel results)
            content = result["content"]
            # Use _display_content with assistant message type so content isn't dimmed
            self._display_content(
                content,
                truncate=False,
                is_error=False,
                message_type=MessageType.ASSISTANT,
                check_markdown_markers=True,
            )

        # Summary
        console.console.print()
        console.console.print("─" * console.console.size.width, style="dim")

        total_tokens = sum(result["tokens"] for result in agent_results)
        total_tools = sum(result["tool_calls"] for result in agent_results)

        summary_parts = [f"{len(agent_results)} models"]
        if total_tokens > 0:
            summary_parts.append(f"{total_tokens:,} tokens")
        if total_tools > 0:
            summary_parts.append(f"{total_tools} tools")

        summary_text = " • ".join(summary_parts)
        console.console.print(f"[dim]{summary_text}[/dim]")
        console.console.print()
