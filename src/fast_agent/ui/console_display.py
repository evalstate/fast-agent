from enum import Enum
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from mcp.types import CallToolResult
from rich.panel import Panel
from rich.text import Text

from fast_agent.ui import console
from fast_agent.ui.mermaid_utils import (
    MermaidDiagram,
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
)

CODE_STYLE = "native"


class MessageType(Enum):
    """Types of messages that can be displayed."""

    USER = "user"
    ASSISTANT = "assistant"
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

    Note: This method does not handle overlapping code blocks (e.g., if inline
    code appears within a fenced code block range). In practice, this is not
    an issue since markdown syntax doesn't support such overlapping.
    """
    if not escape_xml or not isinstance(content, str):
        return content

    protected_ranges = []
    import re

    # Protect fenced code blocks (don't escape anything inside these)
    code_block_pattern = r"```[\s\S]*?```"
    for match in re.finditer(code_block_pattern, content):
        protected_ranges.append((match.start(), match.end()))

    # Protect inline code (don't escape anything inside these)
    inline_code_pattern = r"(?<!`)`(?!``)[^`\n]+`(?!`)"
    for match in re.finditer(inline_code_pattern, content):
        protected_ranges.append((match.start(), match.end()))

    protected_ranges.sort(key=lambda x: x[0])

    # Build the escaped content
    result = []
    last_end = 0

    for start, end in protected_ranges:
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

    def __init__(self, config=None) -> None:
        """
        Initialize the console display handler.

        Args:
            config: Configuration object containing display preferences
        """
        self.config = config
        self._markup = config.logger.enable_markup if config else True
        self._escape_xml = True

    def display_message(
        self,
        content: Any,
        message_type: MessageType,
        name: str | None = None,
        right_info: str = "",
        bottom_metadata: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        is_error: bool = False,
        truncate_content: bool = True,
        additional_message: Text | None = None,
    ) -> None:
        """
        Unified method to display formatted messages to the console.

        Args:
            content: The main content to display (str, Text, JSON, etc.)
            message_type: Type of message (USER, ASSISTANT, TOOL_CALL, TOOL_RESULT)
            name: Optional name to display (agent name, user name, etc.)
            right_info: Information to display on the right side of the header
            bottom_metadata: Optional list of items for bottom separator
            highlight_items: Item(s) to highlight in bottom metadata
            max_item_length: Optional max length for bottom metadata items (with ellipsis)
            is_error: For tool results, whether this is an error (uses red color)
            truncate_content: Whether to truncate long content
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
        self._display_content(
            content, truncate_content, is_error, message_type, check_markdown_markers=False
        )
        if additional_message:
            console.console.print(additional_message, markup=self._markup)

        # Handle bottom separator with optional metadata
        console.console.print()

        if bottom_metadata:
            # Apply shortening if requested
            display_items = bottom_metadata
            if max_item_length:
                display_items = self._shorten_items(bottom_metadata, max_item_length)

            # Normalize highlight_items
            if highlight_items is None:
                highlight_items = []
            elif isinstance(highlight_items, str):
                highlight_items = [highlight_items]

            # Shorten highlight items to match if we shortened display items
            if max_item_length:
                highlight_items = self._shorten_items(highlight_items, max_item_length)

            # Format the metadata with highlighting, clipped to available width
            # Compute available width for the metadata segment (excluding the fixed prefix/suffix)
            total_width = console.console.size.width
            prefix = Text("─| ")
            prefix.stylize("dim")
            suffix = Text(" |")
            suffix.stylize("dim")
            available = max(0, total_width - prefix.cell_len - suffix.cell_len)

            metadata_text = self._format_bottom_metadata(
                display_items,
                highlight_items,
                config["highlight_color"],
                max_width=available,
            )

            # Create the separator line with metadata
            line = Text()
            line.append_text(prefix)
            line.append_text(metadata_text)
            line.append_text(suffix)
            remaining = total_width - line.cell_len
            if remaining > 0:
                line.append("─" * remaining, style="dim")
            console.console.print(line, markup=self._markup)
        else:
            # No metadata - continuous bar
            console.console.print("─" * console.console.size.width, style="dim")

        console.console.print()

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
        # USER and ASSISTANT messages should display in normal style
        # TOOL_CALL and TOOL_RESULT should be dimmed
        if is_error:
            style = "dim red"
        elif message_type in [MessageType.USER, MessageType.ASSISTANT]:
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

    def _format_bottom_metadata(
        self,
        items: List[str],
        highlight_items: List[str],
        highlight_color: str,
        max_width: int | None = None,
    ) -> Text:
        """
        Format a list of items with pipe separators and highlighting.

        Args:
            items: List of items to display
            highlight_items: List of items to highlight
            highlight_color: Color to use for highlighting

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
            should_highlight = False
            if item in highlight_items:
                should_highlight = True
            else:
                for highlight in highlight_items:
                    if item.startswith(highlight) or highlight.endswith(item):
                        should_highlight = True
                        break

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

    def show_tool_result(self, result: CallToolResult, name: str | None = None) -> None:
        """Display a tool result in the new visual style."""
        if not self.config or not self.config.logger.show_tools:
            return

        # Import content helpers
        from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content

        # Analyze content to determine display format and status
        content = result.content
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

        # Build right info
        right_info = f"[dim]tool result - {status}[/dim]"

        # Display using unified method
        self.display_message(
            content=content,
            message_type=MessageType.TOOL_RESULT,
            name=name,
            right_info=right_info,
            is_error=result.isError,
            truncate_content=True,
        )

    def show_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any] | None,
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
    ) -> None:
        """Display a tool call in the new visual style.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments being passed to the tool
            bottom_items: Optional list of items for bottom separator (e.g., available tools)
            highlight_items: Item(s) to highlight in the bottom separator
            max_item_length: Optional max length for bottom items (with ellipsis)
            name: Optional agent name
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Build right info
        right_info = f"[dim]tool request - {tool_name}[/dim]"

        # Display using unified method
        self.display_message(
            content=tool_args,
            message_type=MessageType.TOOL_CALL,
            name=name,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_items=tool_name,
            max_item_length=max_item_length,
            truncate_content=True,
        )

    async def show_tool_update(self, updated_server: str, agent_name: str | None = None) -> None:
        """Show a tool update for a server in the new visual style.

        Args:
            updated_server: Name of the server being updated
            agent_name: Optional agent name to display
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Combined separator and status line
        if agent_name:
            left = (
                f"[magenta]▎[/magenta][dim magenta]▶[/dim magenta] [magenta]{agent_name}[/magenta]"
            )
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

        # Force prompt_toolkit redraw if active
        try:
            from prompt_toolkit.application.current import get_app

            get_app().invalidate()  # Forces prompt_toolkit to redraw
        except:  # noqa: E722
            pass  # No active prompt_toolkit session

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

    async def show_assistant_message(
        self,
        message_text: Union[str, Text],
        bottom_items: List[str] | None = None,
        highlight_items: str | List[str] | None = None,
        max_item_length: int | None = None,
        name: str | None = None,
        model: str | None = None,
        additional_message: Optional[Text] = None,
    ) -> None:
        """Display an assistant message in a formatted panel.

        Args:
            message_text: The message content to display
            bottom_items: Optional list of items for bottom separator (e.g., servers, destinations)
            highlight_items: Item(s) to highlight in the bottom separator
            max_item_length: Optional max length for bottom items (with ellipsis)
            title: Title for the message (default "ASSISTANT")
            name: Optional agent name
            model: Optional model name for right info
            additional_message: Optional additional styled message to append
        """
        if not self.config or not self.config.logger.show_chat:
            return

        # Build right info
        right_info = f"[dim]{model}[/dim]" if model else ""

        # Display main message using unified method
        self.display_message(
            content=message_text,
            message_type=MessageType.ASSISTANT,
            name=name,
            right_info=right_info,
            bottom_metadata=bottom_items,
            highlight_items=highlight_items,
            max_item_length=max_item_length,
            truncate_content=False,  # Assistant messages shouldn't be truncated
            additional_message=additional_message,
        )

        # Handle mermaid diagrams separately (after the main message)
        # Extract plain text if message_text is a Text object
        plain_text = message_text
        if isinstance(message_text, Text):
            plain_text = message_text.plain

        if isinstance(plain_text, str):
            diagrams = extract_mermaid_diagrams(plain_text)
            if diagrams:
                self._display_mermaid_diagrams(diagrams)

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
            # Use _display_content with check_markdown_markers flag for parallel results
            self._display_content(content, truncate=False, check_markdown_markers=True)

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
