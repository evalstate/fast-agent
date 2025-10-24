import asyncio
import math
import time
from contextlib import contextmanager
from enum import Enum
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

from mcp.types import CallToolResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from fast_agent.config import Settings
from fast_agent.constants import REASONING
from fast_agent.ui import console
from fast_agent.ui.markdown_truncator import MarkdownTruncator
from fast_agent.ui.mcp_ui_utils import UILink
from fast_agent.ui.mermaid_utils import (
    MermaidDiagram,
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
)
from fast_agent.ui.plain_text_truncator import PlainTextTruncator

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.mcp.skybridge import SkybridgeServerConfig

CODE_STYLE = "native"

MARKDOWN_STREAM_TARGET_RATIO = 0.7
MARKDOWN_STREAM_REFRESH_PER_SECOND = 4
MARKDOWN_STREAM_HEIGHT_FUDGE = 1
PLAIN_STREAM_TARGET_RATIO = 0.9
PLAIN_STREAM_REFRESH_PER_SECOND = 20
PLAIN_STREAM_HEIGHT_FUDGE = 1


class MessageType(Enum):
    """Types of messages that can be displayed."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# Configuration for each message type
MESSAGE_CONFIGS = {
    MessageType.USER: {
        "block_color": "blue",
        "arrow": "▶",
        "arrow_style": "dim blue",
        "highlight_color": "blue",
    },
    MessageType.ASSISTANT: {
        "block_color": "green",
        "arrow": "◀",
        "arrow_style": "dim green",
        "highlight_color": "bright_green",
    },
    MessageType.SYSTEM: {
        "block_color": "yellow",
        "arrow": "●",
        "arrow_style": "dim yellow",
        "highlight_color": "bright_yellow",
    },
    MessageType.TOOL_CALL: {
        "block_color": "magenta",
        "arrow": "◀",
        "arrow_style": "dim magenta",
        "highlight_color": "magenta",
    },
    MessageType.TOOL_RESULT: {
        "block_color": "magenta",  # Can be overridden to red if error
        "arrow": "▶",
        "arrow_style": "dim magenta",
        "highlight_color": "magenta",
    },
}

HTML_ESCAPE_CHARS = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
}


def _prepare_markdown_content(content: str, escape_xml: bool = True) -> str:
    """Prepare content for markdown rendering by escaping HTML/XML tags
    while preserving code blocks and inline code.

    This ensures XML/HTML tags are displayed as visible text rather than
    being interpreted as markup by the markdown renderer.

    Uses markdown-it parser to properly identify code regions, avoiding
    the issues with regex-based approaches (e.g., backticks inside fenced
    code blocks being misidentified as inline code).
    """
    if not escape_xml or not isinstance(content, str):
        return content

    # Import markdown-it for proper parsing
    from markdown_it import MarkdownIt

    # Parse the markdown to identify code regions
    parser = MarkdownIt()
    try:
        tokens = parser.parse(content)
    except Exception:
        # If parsing fails, fall back to escaping everything
        # (better safe than corrupting content)
        result = content
        for char, replacement in HTML_ESCAPE_CHARS.items():
            result = result.replace(char, replacement)
        return result

    # Collect protected ranges from tokens
    protected_ranges = []
    lines = content.split("\n")

    def _flatten_tokens(tokens):
        """Recursively flatten token tree."""
        for token in tokens:
            yield token
            if token.children:
                yield from _flatten_tokens(token.children)

    # Process all tokens to find code blocks and inline code
    for token in _flatten_tokens(tokens):
        if token.map is not None:
            # Block-level tokens with line mapping (fence, code_block)
            if token.type in ("fence", "code_block"):
                start_line = token.map[0]
                end_line = token.map[1]
                start_pos = sum(len(line) + 1 for line in lines[:start_line])
                end_pos = sum(len(line) + 1 for line in lines[:end_line])
                protected_ranges.append((start_pos, end_pos))

        # Inline code tokens don't have map, but have content
        if token.type == "code_inline":
            # For inline code, we need to find its position in the source
            # The token has the content, but we need to search for it
            # We'll look for the pattern `content` in the content string
            code_content = token.content
            if code_content:
                # Search for this inline code in the content
                # We need to account for the backticks: `content`
                pattern = f"`{code_content}`"
                start = 0
                while True:
                    pos = content.find(pattern, start)
                    if pos == -1:
                        break
                    # Check if this position is already in a protected range
                    in_protected = any(s <= pos < e for s, e in protected_ranges)
                    if not in_protected:
                        protected_ranges.append((pos, pos + len(pattern)))
                    start = pos + len(pattern)

    # Check for incomplete code blocks (streaming scenario)
    # Count opening vs closing fences
    import re

    fence_pattern = r"^```"
    fences = list(re.finditer(fence_pattern, content, re.MULTILINE))

    # If we have an odd number of fences, the last one is incomplete
    if len(fences) % 2 == 1:
        # Protect from the last fence to the end
        last_fence_pos = fences[-1].start()
        # Only add if not already protected
        in_protected = any(s <= last_fence_pos < e for s, e in protected_ranges)
        if not in_protected:
            protected_ranges.append((last_fence_pos, len(content)))

    # Sort and merge overlapping ranges
    protected_ranges.sort(key=lambda x: x[0])

    # Merge overlapping ranges
    merged_ranges = []
    for start, end in protected_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            # Overlapping or adjacent - merge
            merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
        else:
            merged_ranges.append((start, end))

    # Build the escaped content
    result = []
    last_end = 0

    for start, end in merged_ranges:
        # Escape everything outside protected ranges
        unprotected_text = content[last_end:start]
        for char, replacement in HTML_ESCAPE_CHARS.items():
            unprotected_text = unprotected_text.replace(char, replacement)
        result.append(unprotected_text)

        # Keep protected ranges (code blocks) as-is
        result.append(content[start:end])
        last_end = end

    # Escape any remaining content after the last protected range
    remainder_text = content[last_end:]
    for char, replacement in HTML_ESCAPE_CHARS.items():
        remainder_text = remainder_text.replace(char, replacement)
    result.append(remainder_text)

    return "".join(result)


class ConsoleDisplay:
    """
    Handles displaying formatted messages, tool calls, and results to the console.
    This centralizes the UI display logic used by LLM implementations.
    """

    def __init__(self, config: Settings | None = None) -> None:
        """
        Initialize the console display handler.

        Args:
            config: Configuration object containing display preferences
        """
        self.config = config
        self._markup = config.logger.enable_markup if config else True
        self._escape_xml = True

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
        if streaming_mode == "markdown" and getattr(logger_settings, "streaming_plain_text", False):
            streaming_mode = "plain"

        show_chat = bool(getattr(logger_settings, "show_chat", True))
        streaming_display = bool(getattr(logger_settings, "streaming_display", True))

        enabled = show_chat and streaming_display and streaming_mode != "none"
        return enabled, streaming_mode

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
        minutes, seconds = divmod(elapsed, 60)
        if minutes < 60:
            return f"{int(minutes)}m {seconds:02.0f}s"
        hours, minutes = divmod(int(minutes), 60)
        return f"{hours}h {minutes:02d}m"

    def display_message(
        self,
        content: Any,
        message_type: MessageType,
        name: str | None = None,
        right_info: str = "",
        bottom_metadata: List[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        is_error: bool = False,
        truncate_content: bool = True,
        additional_message: Text | None = None,
        pre_content: Text | None = None,
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
        """
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
        left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}]"
        if name:
            left += f" [{block_color if not is_error else 'red'}]{name}[/{block_color if not is_error else 'red'}]"

        # Create combined separator and status line
        self._create_combined_separator_status(left, right_info)

        # Display the content
        if pre_content and pre_content.plain:
            console.console.print(pre_content, markup=self._markup)
        self._display_content(
            content, truncate_content, is_error, message_type, check_markdown_markers=False
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
        message_type: Optional[MessageType] = None,
        check_markdown_markers: bool = False,
    ) -> None:
        """
        Display content in the appropriate format.

        Args:
            content: Content to display
            truncate: Whether to truncate long content
            is_error: Whether this is error content (affects styling)
            message_type: Type of message to determine appropriate styling
            check_markdown_markers: If True, only use markdown rendering when markers are present
        """
        import json
        import re

        from rich.markdown import Markdown
        from rich.pretty import Pretty
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
            # Try to detect and handle different string formats
            try:
                # Try as JSON first
                json_obj = json.loads(content)
                if truncate and self.config and self.config.logger.truncate_tools:
                    pretty_obj = Pretty(json_obj, max_length=10, max_string=50)
                else:
                    pretty_obj = Pretty(json_obj)
                # Apply style only if specified
                if style:
                    console.console.print(pretty_obj, style=style, markup=self._markup)
                else:
                    console.console.print(pretty_obj, markup=self._markup)
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
                    if any(marker in content for marker in ["##", "**", "*", "`", "---", "###"]):
                        # Has markdown markers - render as markdown with escaping
                        prepared_content = _prepare_markdown_content(content, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        console.console.print(md, markup=self._markup)
                    else:
                        # Plain text - display as-is
                        if (
                            truncate
                            and self.config
                            and self.config.logger.truncate_tools
                            and len(content) > 360
                        ):
                            content = content[:360] + "..."
                        if style:
                            console.console.print(content, style=style, markup=self._markup)
                        else:
                            console.console.print(content, markup=self._markup)
                else:
                    # Check if it looks like markdown
                    if any(marker in content for marker in ["##", "**", "*", "`", "---", "###"]):
                        # Escape HTML/XML tags while preserving code blocks
                        prepared_content = _prepare_markdown_content(content, self._escape_xml)
                        md = Markdown(prepared_content, code_theme=CODE_STYLE)
                        # Markdown handles its own styling, don't apply style
                        console.console.print(md, markup=self._markup)
                    else:
                        # Plain text
                        if (
                            truncate
                            and self.config
                            and self.config.logger.truncate_tools
                            and len(content) > 360
                        ):
                            content = content[:360] + "..."
                        # Apply style only if specified (None means default white)
                        if style:
                            console.console.print(content, style=style, markup=self._markup)
                        else:
                            console.console.print(content, markup=self._markup)
        elif isinstance(content, Text):
            # Rich Text object - check if it contains markdown
            plain_text = content.plain

            # Check if the plain text contains markdown markers
            if any(marker in plain_text for marker in ["##", "**", "*", "`", "---", "###"]):
                # Split the Text object into segments
                # We need to handle the main content (which may have markdown)
                # and any styled segments that were appended

                # For now, we'll render the entire content with markdown support
                # This means extracting each span and handling it appropriately
                from rich.markdown import Markdown

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
                    if any(
                        marker in markdown_part for marker in ["##", "**", "*", "`", "---", "###"]
                    ):
                        # Render markdown part
                        prepared_content = _prepare_markdown_content(
                            markdown_part, self._escape_xml
                        )
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
                    prepared_content = _prepare_markdown_content(plain_text, self._escape_xml)
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
                    if (
                        truncate
                        and self.config
                        and self.config.logger.truncate_tools
                        and len(text_content) > 360
                    ):
                        text_content = text_content[:360] + "..."
                    # Apply style only if specified
                    if style:
                        console.console.print(text_content, style=style, markup=self._markup)
                    else:
                        console.console.print(text_content, markup=self._markup)
                else:
                    # Apply style only if specified
                    if style:
                        console.console.print("(empty text)", style=style, markup=self._markup)
                    else:
                        console.console.print("(empty text)", markup=self._markup)
            else:
                # Multiple blocks or non-text content
                if truncate and self.config and self.config.logger.truncate_tools:
                    pretty_obj = Pretty(content, max_length=10, max_string=50)
                else:
                    pretty_obj = Pretty(content)
                # Apply style only if specified
                if style:
                    console.console.print(pretty_obj, style=style, markup=self._markup)
                else:
                    console.console.print(pretty_obj, markup=self._markup)
        else:
            # Any other type - use Pretty
            if truncate and self.config and self.config.logger.truncate_tools:
                pretty_obj = Pretty(content, max_length=10, max_string=50)
            else:
                pretty_obj = Pretty(content)
            # Apply style only if specified
            if style:
                console.console.print(pretty_obj, style=style, markup=self._markup)
            else:
                console.console.print(pretty_obj, markup=self._markup)

    def _shorten_items(self, items: List[str], max_length: int) -> List[str]:
        """
        Shorten items to max_length with ellipsis if needed.

        Args:
            items: List of strings to potentially shorten
            max_length: Maximum length for each item

        Returns:
            List of shortened strings
        """
        return [item[: max_length - 1] + "…" if len(item) > max_length else item for item in items]

    def _render_bottom_metadata(
        self,
        *,
        message_type: MessageType,
        bottom_metadata: List[str] | None,
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
        console.console.print()

        if bottom_metadata:
            display_items = bottom_metadata
            if max_item_length:
                display_items = self._shorten_items(bottom_metadata, max_item_length)

            total_width = console.console.size.width
            prefix = Text("─| ")
            prefix.stylize("dim")
            suffix = Text(" |")
            suffix.stylize("dim")
            available = max(0, total_width - prefix.cell_len - suffix.cell_len)

            highlight_color = MESSAGE_CONFIGS[message_type]["highlight_color"]
            metadata_text = self._format_bottom_metadata(
                display_items,
                highlight_index,
                highlight_color,
                max_width=available,
            )

            line = Text()
            line.append_text(prefix)
            line.append_text(metadata_text)
            line.append_text(suffix)
            remaining = total_width - line.cell_len
            if remaining > 0:
                line.append("─" * remaining, style="dim")
            console.console.print(line, markup=self._markup)
        else:
            console.console.print("─" * console.console.size.width, style="dim")

        console.console.print()

    def _format_bottom_metadata(
        self,
        items: List[str],
        highlight_index: int | None,
        highlight_color: str,
        max_width: int | None = None,
    ) -> Text:
        """
        Format a list of items with pipe separators and highlighting.

        Args:
            items: List of items to display
            highlight_index: Index of item to highlight (0-based), or None for no highlighting
            highlight_color: Color to use for highlighting
            max_width: Maximum width for the formatted text

        Returns:
            Formatted Text object with proper separators and highlighting
        """
        formatted = Text()

        def will_fit(next_segment: Text) -> bool:
            if max_width is None:
                return True
            # projected length if we append next_segment
            return formatted.cell_len + next_segment.cell_len <= max_width

        for i, item in enumerate(items):
            sep = Text(" | ", style="dim") if i > 0 else Text("")

            # Prepare item text with potential highlighting
            should_highlight = highlight_index is not None and i == highlight_index

            item_text = Text(item, style=(highlight_color if should_highlight else "dim"))

            # Check if separator + item fits in available width
            if not will_fit(sep + item_text):
                # If nothing has been added yet and the item itself is too long,
                # leave space for an ellipsis and stop.
                if formatted.cell_len == 0 and max_width is not None and max_width > 1:
                    # show truncated indicator only
                    formatted.append("…", style="dim")
                else:
                    # Indicate there are more items but avoid wrapping
                    if max_width is None or formatted.cell_len < max_width:
                        formatted.append(" …", style="dim")
                break

            # Append separator and item
            if sep.plain:
                formatted.append_text(sep)
            formatted.append_text(item_text)

        return formatted

    def show_tool_result(
        self,
        result: CallToolResult,
        name: str | None = None,
        tool_name: str | None = None,
        skybridge_config: "SkybridgeServerConfig | None" = None,
    ) -> None:
        """Display a tool result in the new visual style.

        Args:
            result: The tool result to display
            name: Optional agent name
            tool_name: Optional tool name for skybridge detection
            skybridge_config: Optional skybridge configuration for the server
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Import content helpers
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        # Analyze content to determine display format and status
        content = result.content
        structured_content = getattr(result, "structuredContent", None)
        has_structured = structured_content is not None

        # Determine if this is a skybridge tool
        is_skybridge_tool = False
        skybridge_resource_uri = None
        if has_structured and tool_name and skybridge_config:
            # Check if this tool is a valid skybridge tool
            for tool_cfg in skybridge_config.tools:
                if tool_cfg.tool_name == tool_name and tool_cfg.is_valid:
                    is_skybridge_tool = True
                    skybridge_resource_uri = tool_cfg.resource_uri
                    break

        if result.isError:
            status = "ERROR"
        else:
            # Check if it's a list with content blocks
            if len(content) == 0:
                status = "No Content"
            elif len(content) == 1 and is_text_content(content[0]):
                text_content = get_text(content[0])
                char_count = len(text_content) if text_content else 0
                status = f"Text Only {char_count} chars"
            else:
                text_count = sum(1 for item in content if is_text_content(item))
                if text_count == len(content):
                    status = f"{len(content)} Text Blocks" if len(content) > 1 else "1 Text Block"
                else:
                    status = (
                        f"{len(content)} Content Blocks" if len(content) > 1 else "1 Content Block"
                    )

        # Build transport channel info for bottom bar
        channel = getattr(result, "transport_channel", None)
        bottom_metadata_items: List[str] = []
        if channel:
            # Format channel info for bottom bar
            if channel == "post-json":
                transport_info = "HTTP (JSON-RPC)"
            elif channel == "post-sse":
                transport_info = "HTTP (SSE)"
            elif channel == "get":
                transport_info = "Legacy SSE"
            elif channel == "resumption":
                transport_info = "Resumption"
            elif channel == "stdio":
                transport_info = "STDIO"
            else:
                transport_info = channel.upper()

            bottom_metadata_items.append(transport_info)

        elapsed = getattr(result, "transport_elapsed", None)
        if isinstance(elapsed, (int, float)):
            bottom_metadata_items.append(self._format_elapsed(float(elapsed)))

        # Add structured content indicator if present
        if has_structured:
            bottom_metadata_items.append("Structured ■")

        bottom_metadata = bottom_metadata_items or None

        # Build right info (without channel info)
        right_info = f"[dim]tool result - {status}[/dim]"

        if has_structured:
            # Handle structured content display manually to insert it before bottom separator
            # Display main content without bottom separator
            config = MESSAGE_CONFIGS[MessageType.TOOL_RESULT]
            block_color = "red" if result.isError else config["block_color"]
            arrow = config["arrow"]
            arrow_style = config["arrow_style"]
            left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}]"
            if name:
                left += f" [{block_color if not result.isError else 'red'}]{name}[/{block_color if not result.isError else 'red'}]"

            # Top separator
            self._create_combined_separator_status(left, right_info)

            # Main content
            self._display_content(
                content, True, result.isError, MessageType.TOOL_RESULT, check_markdown_markers=False
            )

            # Structured content separator and display
            console.console.print()
            total_width = console.console.size.width

            if is_skybridge_tool:
                # Skybridge: magenta separator with resource URI
                resource_label = (
                    f"skybridge resource: {skybridge_resource_uri}"
                    if skybridge_resource_uri
                    else "skybridge resource"
                )
                prefix = Text("─| ")
                prefix.stylize("dim")
                resource_text = Text(resource_label, style="magenta")
                suffix = Text(" |")
                suffix.stylize("dim")

                separator_line = Text()
                separator_line.append_text(prefix)
                separator_line.append_text(resource_text)
                separator_line.append_text(suffix)
                remaining = total_width - separator_line.cell_len
                if remaining > 0:
                    separator_line.append("─" * remaining, style="dim")
                console.console.print(separator_line, markup=self._markup)
                console.console.print()

                # Display with bright syntax highlighting
                import json

                from rich.syntax import Syntax

                json_str = json.dumps(structured_content, indent=2)
                syntax_obj = Syntax(json_str, "json", theme=CODE_STYLE, background_color="default")
                console.console.print(syntax_obj, markup=self._markup)
            else:
                # Regular tool: dim separator
                prefix = Text("─| ")
                prefix.stylize("dim")
                label_text = Text("Structured Content", style="dim")
                suffix = Text(" |")
                suffix.stylize("dim")

                separator_line = Text()
                separator_line.append_text(prefix)
                separator_line.append_text(label_text)
                separator_line.append_text(suffix)
                remaining = total_width - separator_line.cell_len
                if remaining > 0:
                    separator_line.append("─" * remaining, style="dim")
                console.console.print(separator_line, markup=self._markup)
                console.console.print()

                # Display truncated content in dim
                from rich.pretty import Pretty

                if self.config and self.config.logger.truncate_tools:
                    pretty_obj = Pretty(structured_content, max_length=10, max_string=50)
                else:
                    pretty_obj = Pretty(structured_content)
                console.console.print(pretty_obj, style="dim", markup=self._markup)

            # Bottom separator with metadata
            console.console.print()
            if bottom_metadata:
                display_items = (
                    self._shorten_items(bottom_metadata, 12) if True else bottom_metadata
                )
                prefix = Text("─| ")
                prefix.stylize("dim")
                suffix = Text(" |")
                suffix.stylize("dim")
                available = max(0, total_width - prefix.cell_len - suffix.cell_len)

                metadata_text = self._format_bottom_metadata(
                    display_items,
                    None,
                    config["highlight_color"],
                    max_width=available,
                )

                line = Text()
                line.append_text(prefix)
                line.append_text(metadata_text)
                line.append_text(suffix)
                remaining = total_width - line.cell_len
                if remaining > 0:
                    line.append("─" * remaining, style="dim")
                console.console.print(line, markup=self._markup)
            else:
                console.console.print("─" * total_width, style="dim")
            console.console.print()

        else:
            # No structured content - use standard display
            self.display_message(
                content=content,
                message_type=MessageType.TOOL_RESULT,
                name=name,
                right_info=right_info,
                bottom_metadata=bottom_metadata,
                is_error=result.isError,
                truncate_content=True,
            )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any] | None,
        bottom_items: List[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Display a tool call in the new visual style.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments being passed to the tool
            bottom_items: Optional list of items for bottom separator (e.g., available tools)
            highlight_index: Index of item to highlight in the bottom separator (0-based), or None
            max_item_length: Optional max length for bottom items (with ellipsis)
            name: Optional agent name
            metadata: Optional dictionary of metadata about the tool call
        """
        if not self.config or not self.config.logger.show_tools:
            return

        tool_args = tool_args or {}
        metadata = metadata or {}
        # Build right info and specialised content for known variants
        right_info = f"[dim]tool request - {tool_name}[/dim]"
        content: Any = tool_args
        pre_content: Text | None = None
        truncate_content = True

        if metadata.get("variant") == "shell":
            bottom_items = None
            command = metadata.get("command") or tool_args.get("command")

            command_text = Text()
            if command and isinstance(command, str):
                # Only prepend $ to the first line, not continuation lines
                command_text.append("$ ", style="magenta")
                command_text.append(command, style="white")
            else:
                command_text.append("$ ", style="magenta")
                command_text.append("(no shell command provided)", style="dim")

            content = command_text

            # Include shell name and path in the header, with timeout
            shell_name = metadata.get("shell_name") or "shell"
            shell_path = metadata.get("shell_path")

            # Build header right info with shell and timeout
            right_parts = []
            if shell_path and shell_path != shell_name:
                right_parts.append(f"{shell_name} ({shell_path})")
            elif shell_name:
                right_parts.append(shell_name)

            right_info = f"[dim]{' | '.join(right_parts)}[/dim]" if right_parts else ""
            truncate_content = False

            # Build compact metadata summary - just working directory now
            metadata_text = Text()
            working_dir_display = metadata.get("working_dir_display") or metadata.get("working_dir")
            if working_dir_display:
                metadata_text.append(f"cwd: {working_dir_display} ", style="dim")

            timeout_seconds = metadata.get("timeout_seconds")
            warning_interval = metadata.get("warning_interval_seconds")

            if timeout_seconds and warning_interval:
                metadata_text.append(
                    f"timeout: {timeout_seconds}s, warning every {warning_interval}s\n", style="dim"
                )

            pre_content = metadata_text

        # Display using unified method
        self.display_message(
            content=content,
            message_type=MessageType.TOOL_CALL,
            name=name,
            pre_content=pre_content,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_index=highlight_index,
            max_item_length=max_item_length,
            truncate_content=truncate_content,
        )

    async def show_tool_update(self, updated_server: str, agent_name: str | None = None) -> None:
        """Show a tool update for a server in the new visual style.

        Args:
            updated_server: Name of the server being updated
            agent_name: Optional agent name to display
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Check if prompt_toolkit is active
        try:
            from prompt_toolkit.application.current import get_app

            app = get_app()
            # We're in interactive mode - add to notification tracker
            from fast_agent.ui import notification_tracker

            notification_tracker.add_tool_update(updated_server)
            app.invalidate()  # Force toolbar redraw

        except:  # noqa: E722
            # No active prompt_toolkit session - display with rich as before
            # Combined separator and status line
            if agent_name:
                left = f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{agent_name}[/magenta]"
            else:
                left = "[magenta]▎[/magenta][dim magenta]▶[/dim magenta]"

            right = f"[dim]{updated_server}[/dim]"
            self._create_combined_separator_status(left, right)

            # Display update message
            message = f"Updating tools for server {updated_server}"
            console.console.print(message, style="dim", markup=self._markup)

            # Bottom separator
            console.console.print()
            console.console.print("─" * console.console.size.width, style="dim")
            console.console.print()

    def _create_combined_separator_status(self, left_content: str, right_info: str = "") -> None:
        """
        Create a combined separator and status line.

        Args:
            left_content: The main content (block, arrow, name) - left justified with color
            right_info: Supplementary information to show in brackets - right aligned
        """
        width = console.console.size.width

        # Create left text
        left_text = Text.from_markup(left_content)

        # Create right text if we have info
        if right_info and right_info.strip():
            # Add dim brackets around the right info
            right_text = Text()
            right_text.append("[", style="dim")
            right_text.append_text(Text.from_markup(right_info))
            right_text.append("]", style="dim")
            # Calculate separator count
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1  # Always at least 1 separator
        else:
            right_text = Text("")
            separator_count = width - left_text.cell_len

        # Build the combined line
        combined = Text()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * (separator_count - 1), style="dim")
        combined.append_text(right_text)

        # Print with empty line before
        console.console.print()
        console.console.print(combined, markup=self._markup)
        console.console.print()

    @staticmethod
    def summarize_skybridge_configs(
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Convert raw Skybridge configs into display-friendly summary data."""
        server_rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        warning_seen: Set[str] = set()

        if not configs:
            return server_rows, warnings

        def add_warning(message: str) -> None:
            formatted = message.strip()
            if not formatted:
                return
            if formatted not in warning_seen:
                warnings.append(formatted)
                warning_seen.add(formatted)

        for server_name in sorted(configs.keys()):
            config = configs.get(server_name)
            if not config:
                continue
            resources = list(config.ui_resources or [])
            has_skybridge_signal = bool(
                config.enabled or resources or config.tools or config.warnings
            )
            if not has_skybridge_signal:
                continue

            valid_resource_count = sum(1 for resource in resources if resource.is_skybridge)

            server_rows.append(
                {
                    "server_name": server_name,
                    "config": config,
                    "resources": resources,
                    "valid_resource_count": valid_resource_count,
                    "total_resource_count": len(resources),
                    "active_tools": [
                        {
                            "name": tool.display_name,
                            "template": str(tool.template_uri) if tool.template_uri else None,
                        }
                        for tool in config.tools
                        if tool.is_valid
                    ],
                    "enabled": config.enabled,
                }
            )

            for warning in config.warnings:
                message = warning.strip()
                if not message:
                    continue
                if not message.startswith(server_name):
                    message = f"{server_name} {message}"
                add_warning(message)

        return server_rows, warnings

    def show_skybridge_summary(
        self,
        agent_name: str,
        configs: Mapping[str, "SkybridgeServerConfig"] | None,
    ) -> None:
        """Display Skybridge availability and warnings."""
        server_rows, warnings = self.summarize_skybridge_configs(configs)

        if not server_rows and not warnings:
            return

        heading = "[dim]OpenAI Apps SDK ([/dim][cyan]skybridge[/cyan][dim]) detected:[/dim]"
        console.console.print()
        console.console.print(heading, markup=self._markup)

        if not server_rows:
            console.console.print("[dim]  ● none detected[/dim]", markup=self._markup)
        else:
            for row in server_rows:
                server_name = row["server_name"]
                resource_count = row["valid_resource_count"]
                total_resource_count = row["total_resource_count"]
                tool_infos = row["active_tools"]
                enabled = row["enabled"]

                tool_count = len(tool_infos)
                tool_word = "tool" if tool_count == 1 else "tools"
                resource_word = (
                    "skybridge resource" if resource_count == 1 else "skybridge resources"
                )
                tool_segment = f"[cyan]{tool_count}[/cyan][dim] {tool_word}[/dim]"
                resource_segment = f"[cyan]{resource_count}[/cyan][dim] {resource_word}[/dim]"
                name_style = "cyan" if enabled else "yellow"
                status_suffix = "" if enabled else "[dim] (issues detected)[/dim]"

                console.console.print(
                    f"[dim]  ● [/dim][{name_style}]{server_name}[/{name_style}]{status_suffix}"
                    f"[dim] — [/dim]{tool_segment}[dim], [/dim]{resource_segment}",
                    markup=self._markup,
                )

                for tool_info in tool_infos:
                    template_text = (
                        f"[dim] ({tool_info['template']})[/dim]" if tool_info["template"] else ""
                    )
                    console.console.print(
                        f"[dim]    ▶ [/dim][white]{tool_info['name']}[/white]{template_text}",
                        markup=self._markup,
                    )

                if tool_count == 0 and resource_count > 0:
                    console.console.print(
                        "[dim]     ▶ tools not linked[/dim]",
                        markup=self._markup,
                    )
                if not enabled and total_resource_count > resource_count:
                    invalid_count = total_resource_count - resource_count
                    invalid_word = "resource" if invalid_count == 1 else "resources"
                    console.console.print(
                        (
                            "[dim]     ▶ "
                            f"[/dim][cyan]{invalid_count}[/cyan][dim] {invalid_word} detected with non-skybridge MIME type[/dim]"
                        ),
                        markup=self._markup,
                    )

        for warning_entry in warnings:
            console.console.print(
                f"[dim red]  ▶ [/dim red][red]warning[/red] [dim]{warning_entry}[/dim]",
                markup=self._markup,
            )

    def _extract_reasoning_content(self, message: "PromptMessageExtended") -> Text | None:
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

        return Text(joined, style="dim default")

    async def show_assistant_message(
        self,
        message_text: Union[str, Text, "PromptMessageExtended"],
        bottom_items: List[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional[Text] = None,
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
        """
        if not self.config or not self.config.logger.show_chat:
            return

        # Extract text from PromptMessageExtended if needed
        from fast_agent.types import PromptMessageExtended

        pre_content: Text | None = None

        if isinstance(message_text, PromptMessageExtended):
            display_text = message_text.last_text() or ""
            pre_content = self._extract_reasoning_content(message_text)
        else:
            display_text = message_text

        # Build right info
        right_info = f"[dim]{model}[/dim]" if model else ""

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
        bottom_items: List[str] | None = None,
        highlight_index: int | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
    ) -> Iterator["_StreamingMessageHandle"]:
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

        left = f"[{block_color}]▎[/{block_color}][{arrow_style}]{arrow}[/{arrow_style}] "
        if name:
            left += f"[{block_color}]{name}[/{block_color}]"

        right_info = f"[dim]{model}[/dim]" if model else ""

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
            progress_display=progress_display,
        )
        try:
            yield handle
        finally:
            handle.close()

    def _display_mermaid_diagrams(self, diagrams: List[MermaidDiagram]) -> None:
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

    async def show_mcp_ui_links(self, links: List[UILink]) -> None:
        """Display MCP-UI links beneath the chat like mermaid links."""
        if not self.config or not self.config.logger.show_chat:
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

    def show_user_message(
        self,
        message: Union[str, Text],
        model: str | None = None,
        chat_turn: int = 0,
        name: str | None = None,
    ) -> None:
        """Display a user message in the new visual style."""
        if not self.config or not self.config.logger.show_chat:
            return

        # Build right side with model and turn
        right_parts = []
        if model:
            right_parts.append(model)
        if chat_turn > 0:
            right_parts.append(f"turn {chat_turn}")

        right_info = f"[dim]{' '.join(right_parts)}[/dim]" if right_parts else ""

        self.display_message(
            content=message,
            message_type=MessageType.USER,
            name=name,
            right_info=right_info,
            truncate_content=False,  # User messages typically shouldn't be truncated
        )

    def show_system_message(
        self,
        system_prompt: str,
        agent_name: str | None = None,
        server_count: int = 0,
    ) -> None:
        """Display the system prompt in a formatted panel."""
        if not self.config or not self.config.logger.show_chat:
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
        description: Optional[str] = None,
        message_count: int = 0,
        agent_name: Optional[str] = None,
        server_list: List[str] | None = None,
        highlight_server: str | None = None,
        arguments: Optional[dict[str, str]] = None,
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
        if not self.config or not self.config.logger.show_tools:
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
            if (
                hasattr(agent, "_llm")
                and agent._llm
                and hasattr(agent._llm, "default_request_params")
            ):
                model = getattr(agent._llm.default_request_params, "model", "unknown")

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


class _NullStreamingHandle:
    """No-op streaming handle used when streaming is disabled."""

    def update(self, _chunk: str) -> None:
        return

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        return

    def close(self) -> None:
        return


class _StreamingMessageHandle:
    """Helper that manages live rendering for streaming assistant responses."""

    def __init__(
        self,
        *,
        display: ConsoleDisplay,
        bottom_items: List[str] | None,
        highlight_index: int | None,
        max_item_length: int | None,
        use_plain_text: bool = False,
        header_left: str = "",
        header_right: str = "",
        progress_display=None,
    ) -> None:
        self._display = display
        self._bottom_items = bottom_items
        self._highlight_index = highlight_index
        self._max_item_length = max_item_length
        self._use_plain_text = use_plain_text
        self._header_left = header_left
        self._header_right = header_right
        self._progress_display = progress_display
        self._progress_paused = False
        self._buffer: List[str] = []
        initial_renderable = Text("") if self._use_plain_text else Markdown("")
        refresh_rate = (
            PLAIN_STREAM_REFRESH_PER_SECOND
            if self._use_plain_text
            else MARKDOWN_STREAM_REFRESH_PER_SECOND
        )
        self._min_render_interval = 1.0 / refresh_rate if refresh_rate else None
        self._last_render_time = 0.0
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._async_mode = self._loop is not None
        self._queue: asyncio.Queue[object] | None = asyncio.Queue() if self._async_mode else None
        self._stop_sentinel: object = object()
        self._worker_task: asyncio.Task[None] | None = None
        self._live: Live | None = Live(
            initial_renderable,
            console=console.console,
            vertical_overflow="ellipsis",
            refresh_per_second=refresh_rate,
            transient=True,
        )
        self._live_started = False
        self._active = True
        self._finalized = False
        # Track whether we're in a table to batch updates
        self._in_table = False
        self._pending_table_row = ""
        # Smart markdown truncator for creating display window (doesn't mutate buffer)
        self._truncator = MarkdownTruncator(target_height_ratio=MARKDOWN_STREAM_TARGET_RATIO)
        self._plain_truncator = (
            PlainTextTruncator(target_height_ratio=PLAIN_STREAM_TARGET_RATIO)
            if self._use_plain_text
            else None
        )
        self._max_render_height = 0

        if self._async_mode and self._loop and self._queue is not None:
            self._worker_task = self._loop.create_task(self._render_worker())

    def update(self, chunk: str) -> None:
        if not self._active or not chunk:
            return

        if self._async_mode and self._queue is not None:
            self._enqueue_chunk(chunk)
            return

        if self._handle_chunk(chunk):
            self._render_current_buffer()

    def _build_header(self) -> Text:
        """Build the header bar as a Text renderable.

        Returns:
            Text object representing the header bar.
        """
        width = console.console.size.width

        # Create left text
        left_text = Text.from_markup(self._header_left)

        # Create right text if we have info
        if self._header_right and self._header_right.strip():
            # Add dim brackets around the right info
            right_text = Text()
            right_text.append("[", style="dim")
            right_text.append_text(Text.from_markup(self._header_right))
            right_text.append("]", style="dim")
            # Calculate separator count
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1  # Always at least 1 separator
        else:
            right_text = Text("")
            separator_count = width - left_text.cell_len

        # Build the combined line
        combined = Text()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * (separator_count - 1), style="dim")
        combined.append_text(right_text)

        return combined

    def _pause_progress_display(self) -> None:
        if self._progress_display and not self._progress_paused:
            try:
                self._progress_display.pause()
                self._progress_paused = True
            except Exception:
                self._progress_paused = False

    def _resume_progress_display(self) -> None:
        if self._progress_display and self._progress_paused:
            try:
                self._progress_display.resume()
            except Exception:
                pass
            finally:
                self._progress_paused = False

    def _ensure_started(self) -> None:
        """Start live rendering and pause progress display if needed."""
        if not self._live:
            return

        if self._live_started:
            return

        self._pause_progress_display()

        if self._live and not self._live_started:
            self._live.__enter__()
            self._live_started = True

    def _close_incomplete_code_blocks(self, text: str) -> str:
        """Add temporary closing fence to incomplete code blocks for display.

        During streaming, incomplete code blocks (opening fence without closing)
        are rendered as literal text by Rich's Markdown renderer. This method
        adds a temporary closing fence so the code can be syntax-highlighted
        during streaming display.

        When the real closing fence arrives in a subsequent chunk, this method
        will detect the now-complete block and stop adding the temporary fence.

        Args:
            text: The markdown text that may contain incomplete code blocks.

        Returns:
            Text with temporary closing fences added for incomplete code blocks.
        """
        import re

        # Count opening and closing fences
        opening_fences = len(re.findall(r"^```", text, re.MULTILINE))
        closing_fences = len(re.findall(r"^```\s*$", text, re.MULTILINE))

        # If we have more opening fences than closing fences, and the text
        # doesn't end with a closing fence, we have an incomplete code block
        if opening_fences > closing_fences:
            # Check if text ends with a closing fence (might be partial line)
            if not re.search(r"```\s*$", text):
                # Add temporary closing fence for display only
                return text + "\n```\n"

        return text

    def _trim_to_displayable(self, text: str) -> str:
        """Trim text to keep only displayable content plus small buffer.

        Keeps ~1.5x terminal height worth of recent content.
        Uses the optimized streaming truncator for better performance.

        Args:
            text: Full text to trim

        Returns:
            Trimmed text (most recent content)
        """
        if not text:
            return text

        terminal_height = console.console.size.height - 1

        if self._use_plain_text and self._plain_truncator:
            terminal_width = console.console.size.width
            return self._plain_truncator.truncate(
                text,
                terminal_height=terminal_height,
                terminal_width=terminal_width,
            )

        # Use the optimized streaming truncator (16x faster!) for markdown
        return self._truncator.truncate(
            text,
            terminal_height=terminal_height,
            console=console.console,
            code_theme=CODE_STYLE,
            prefer_recent=True,  # Streaming mode
        )

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        if not self._active or self._finalized:
            return

        self._finalized = True
        self.close()

    def close(self) -> None:
        if not self._active:
            return

        self._active = False
        if self._async_mode:
            if self._queue and self._loop:
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None
                enqueue = (
                    self._queue.put_nowait
                    if current_loop is self._loop
                    else lambda item: self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
                )
                try:
                    enqueue(self._stop_sentinel)
                except RuntimeError:
                    pass
            if self._worker_task:
                self._worker_task.cancel()
                self._worker_task = None
        self._shutdown_live_resources()
        self._max_render_height = 0

    def _extract_trailing_paragraph(self, text: str) -> str:
        """Return text since the last blank line, used to detect in-progress paragraphs."""
        if not text:
            return ""
        double_break = text.rfind("\n\n")
        if double_break != -1:
            candidate = text[double_break + 2 :]
        else:
            candidate = text
        if "\n" in candidate:
            candidate = candidate.split("\n")[-1]
        return candidate

    def _wrap_plain_chunk(self, chunk: str) -> str:
        """Insert soft line breaks into long plain text segments."""
        width = max(1, console.console.size.width)
        if not chunk or width <= 1:
            return chunk

        result_segments: List[str] = []
        start = 0
        length = len(chunk)

        while start < length:
            newline_pos = chunk.find("\n", start)
            if newline_pos == -1:
                line = chunk[start:]
                delimiter = ""
                start = length
            else:
                line = chunk[start:newline_pos]
                delimiter = "\n"
                start = newline_pos + 1

            if len(line.expandtabs()) > width:
                wrapped = self._wrap_plain_line(line, width)
                result_segments.append("\n".join(wrapped))
            else:
                result_segments.append(line)

            result_segments.append(delimiter)

        return "".join(result_segments)

    @staticmethod
    def _wrap_plain_line(line: str, width: int) -> List[str]:
        """Wrap a single line to the terminal width."""
        if not line:
            return [""]

        segments: List[str] = []
        remaining = line

        while len(remaining) > width:
            break_at = remaining.rfind(" ", 0, width)
            if break_at == -1 or break_at < width // 2:
                break_at = width
                segments.append(remaining[:break_at])
                remaining = remaining[break_at:]
            else:
                segments.append(remaining[:break_at])
                remaining = remaining[break_at + 1 :]
        segments.append(remaining)
        return segments

    def _estimate_plain_render_height(self, text: str) -> int:
        """Estimate rendered height for plain text taking terminal width into account."""
        if not text:
            return 0

        width = max(1, console.console.size.width)
        lines = text.split("\n")
        total = 0
        for line in lines:
            expanded_len = len(line.expandtabs())
            total += max(1, math.ceil(expanded_len / width)) if expanded_len else 1
        return total

    def _enqueue_chunk(self, chunk: str) -> None:
        if not self._queue or not self._loop:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._loop:
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Shouldn't happen with default unlimited queue, but fail safe
                pass
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

    def _handle_chunk(self, chunk: str) -> bool:
        """
        Process an incoming chunk and determine whether rendering is needed.

        Returns:
            True if the display should be updated, False otherwise.
        """
        if not chunk:
            return False

        if self._use_plain_text:
            chunk = self._wrap_plain_chunk(chunk)
            if self._pending_table_row:
                self._buffer.append(self._pending_table_row)
                self._pending_table_row = ""
        else:
            text_so_far = "".join(self._buffer)
            lines = text_so_far.strip().split("\n")
            last_line = lines[-1] if lines else ""
            currently_in_table = last_line.strip().startswith("|")

            if currently_in_table and "\n" not in chunk:
                self._pending_table_row += chunk
                return False

            if self._pending_table_row:
                self._buffer.append(self._pending_table_row)
                self._pending_table_row = ""

        self._buffer.append(chunk)
        return True

    def _render_current_buffer(self) -> None:
        if not self._buffer:
            return

        self._ensure_started()

        if not self._live:
            return

        text = "".join(self._buffer)

        if self._use_plain_text:
            trimmed = self._trim_to_displayable(text)
            if trimmed != text:
                text = trimmed
                self._buffer = [trimmed]
        trailing_paragraph = self._extract_trailing_paragraph(text)
        if trailing_paragraph and "\n" not in trailing_paragraph:
            width = max(1, console.console.size.width)
            target_ratio = (
                PLAIN_STREAM_TARGET_RATIO if self._use_plain_text else MARKDOWN_STREAM_TARGET_RATIO
            )
            target_rows = max(1, int(console.console.size.height * target_ratio) - 1)
            estimated_rows = math.ceil(len(trailing_paragraph.expandtabs()) / width)
            if estimated_rows > target_rows:
                trimmed_text = self._trim_to_displayable(text)
                if trimmed_text != text:
                    text = trimmed_text
                    self._buffer = [trimmed_text]

        if len(self._buffer) > 10:
            text = self._trim_to_displayable(text)
            self._buffer = [text]

        # Build the header bar
        header = self._build_header()

        # Build the content renderable
        max_allowed_height = max(1, console.console.size.height - 2)
        self._max_render_height = min(self._max_render_height, max_allowed_height)

        if self._use_plain_text:
            content_height = self._estimate_plain_render_height(text)
            budget_height = min(content_height + PLAIN_STREAM_HEIGHT_FUDGE, max_allowed_height)

            if budget_height > self._max_render_height:
                self._max_render_height = budget_height

            padding_lines = max(0, self._max_render_height - content_height)
            display_text = text + ("\n" * padding_lines if padding_lines else "")
            content = Text(display_text)
        else:
            prepared = _prepare_markdown_content(text, self._display._escape_xml)
            prepared_for_display = self._close_incomplete_code_blocks(prepared)

            content_height = self._truncator.measure_rendered_height(
                prepared_for_display, console.console, CODE_STYLE
            )
            budget_height = min(content_height + MARKDOWN_STREAM_HEIGHT_FUDGE, max_allowed_height)

            if budget_height > self._max_render_height:
                self._max_render_height = budget_height

            padding_lines = max(0, self._max_render_height - content_height)
            if padding_lines:
                prepared_for_display = prepared_for_display + ("\n" * padding_lines)

            content = Markdown(prepared_for_display, code_theme=CODE_STYLE)

        from rich.console import Group

        header_with_spacing = header.copy()
        header_with_spacing.append("\n", style="default")

        combined = Group(header_with_spacing, content)
        try:
            self._live.update(combined)
            self._last_render_time = time.monotonic()
        except Exception:
            # Avoid crashing streaming on renderer errors
            pass

    async def _render_worker(self) -> None:
        assert self._queue is not None
        try:
            while True:
                try:
                    item = await self._queue.get()
                except asyncio.CancelledError:
                    break

                if item is self._stop_sentinel:
                    break

                stop_requested = False
                chunks = [item]
                while True:
                    try:
                        next_item = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if next_item is self._stop_sentinel:
                        stop_requested = True
                        break
                    chunks.append(next_item)

                should_render = False
                for chunk in chunks:
                    if isinstance(chunk, str):
                        should_render = self._handle_chunk(chunk) or should_render

                if should_render:
                    self._render_current_buffer()
                    if self._min_render_interval:
                        try:
                            await asyncio.sleep(self._min_render_interval)
                        except asyncio.CancelledError:
                            break

                if stop_requested:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown_live_resources()

    def _shutdown_live_resources(self) -> None:
        if self._live and self._live_started:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
            self._live_started = False

        self._resume_progress_display()
        self._active = False

    def handle_tool_event(self, event_type: str, info: Dict[str, Any] | None = None) -> None:
        if not self._active:
            return

        if event_type == "start":
            self.close()
            return
        elif event_type == "text":
            self._pause_progress_display()
        elif event_type == "stop":
            self._resume_progress_display()
