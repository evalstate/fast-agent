from json import JSONDecodeError
from typing import Optional, Union

from mcp.types import CallToolResult
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text

from mcp_agent import console
from mcp_agent.core.mermaid_utils import (
    create_mermaid_live_link,
    detect_diagram_type,
    extract_mermaid_diagrams,
)
from mcp_agent.mcp.common import SEP
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

# Constants
HUMAN_INPUT_TOOL_NAME = "__human_input__"
CODE_STYLE = "native"

HTML_ESCAPE_CHARS = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


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

    def _render_content_smartly(self, content: str, check_markdown_markers: bool = False) -> None:
        """
        Helper method to intelligently render content based on its type.

        - Pure XML: Use syntax highlighting for readability
        - Markdown (with markers): Use markdown rendering with proper escaping
        - Plain text: Display as-is (when check_markdown_markers=True and no markers found)

        Args:
            content: The text content to render
            check_markdown_markers: If True, only use markdown rendering when markers are present
        """
        import re

        from rich.markdown import Markdown

        # Check if content appears to be primarily XML
        xml_pattern = r"^<[a-zA-Z_][a-zA-Z0-9_-]*[^>]*>"
        is_xml_content = bool(re.match(xml_pattern, content.strip())) and content.count("<") > 5

        if is_xml_content:
            # Display XML content with syntax highlighting for better readability
            from rich.syntax import Syntax

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
                console.console.print(content, markup=self._markup)
        else:
            # Always treat as markdown with proper escaping
            prepared_content = _prepare_markdown_content(content, self._escape_xml)
            md = Markdown(prepared_content, code_theme=CODE_STYLE)
            console.console.print(md, markup=self._markup)

    def show_tool_result(self, result: CallToolResult, name: str | None = None) -> None:
        """Display a tool result in the new visual style."""
        if not self.config or not self.config.logger.show_tools:
            return

        # Import content helpers
        from mcp_agent.mcp.helpers.content_helpers import get_text, is_text_content

        # Use red for errors, magenta for success (keep block bright)
        block_color = "red" if result.isError else "magenta"
        text_color = "dim red" if result.isError else "dim magenta"

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

        # Combined separator and status line
        left = f"[{block_color}]▎[/{block_color}][{text_color}]▶[/{text_color}]{f' [{block_color}]{name}[/{block_color}]' if name else ''}"
        right = f"[dim]tool result - {status}[/dim]"
        self._create_combined_separator_status(left, right)

        # Display tool result content
        content = result.content

        # Handle special case: single text content block
        if isinstance(content, list) and len(content) == 1 and is_text_content(content[0]):
            # Display just the text content directly
            text_content = get_text(content[0])
            if text_content:
                if self.config and self.config.logger.truncate_tools and len(text_content) > 360:
                    text_content = text_content[:360] + "..."
                console.console.print(text_content, style="dim", markup=self._markup)
            else:
                console.console.print("(empty text)", style="dim", markup=self._markup)
        else:
            # Use Rich pretty printing for everything else
            try:
                import json

                from rich.pretty import Pretty

                # Try to parse as JSON for pretty printing
                if isinstance(content, str):
                    json_obj = json.loads(content)
                else:
                    json_obj = content

                # Use Rich's built-in truncation with dimmed styling
                if self.config and self.config.logger.truncate_tools:
                    pretty_obj = Pretty(json_obj, max_length=10, max_string=50)
                else:
                    pretty_obj = Pretty(json_obj)

                # Print with dim styling
                console.console.print(pretty_obj, style="dim", markup=self._markup)

            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                # Fall back to string representation if not valid JSON
                content_str = str(content)
                if self.config and self.config.logger.truncate_tools and len(content_str) > 360:
                    content_str = content_str[:360] + "..."
                console.console.print(content_str, style="dim", markup=self._markup)

        # Bottom separator (no additional info for tool results)
        console.console.print()
        console.console.print("─" * console.console.size.width, style="dim")
        console.console.print()

    def show_tool_call(
        self, available_tools, tool_name, tool_args, name: str | None = None
    ) -> None:
        """Display a tool call in the new visual style."""
        if not self.config or not self.config.logger.show_tools:
            return

        display_tool_list = self._format_tool_list(available_tools, tool_name)

        # Combined separator and status line
        left = f"[magenta]▎[/magenta][dim magenta]◀[/dim magenta]{f' [magenta]{name}[/magenta]' if name else ''}"
        right = f"[dim]tool request - {tool_name}[/dim]"
        self._create_combined_separator_status(left, right)

        # Display tool arguments using Rich JSON pretty printing (dimmed)
        try:
            import json

            from rich.pretty import Pretty

            # Try to parse as JSON for pretty printing
            if isinstance(tool_args, str):
                json_obj = json.loads(tool_args)
            else:
                json_obj = tool_args

            # Use Rich's built-in truncation with dimmed styling
            if self.config and self.config.logger.truncate_tools:
                pretty_obj = Pretty(json_obj, max_length=10, max_string=50)
            else:
                pretty_obj = Pretty(json_obj)

            # Print with dim styling
            console.console.print(pretty_obj, style="dim", markup=self._markup)

        except (json.JSONDecodeError, TypeError, ValueError):
            # Fall back to string representation if not valid JSON
            content = str(tool_args)
            if self.config and self.config.logger.truncate_tools and len(content) > 360:
                content = content[:360] + "..."
            console.console.print(content, style="dim", markup=self._markup)

        # Bottom separator with tool list using pipe separators (matching server style)
        console.console.print()

        # Use existing tool list formatting with pipe separators
        if display_tool_list and len(display_tool_list) > 0:
            # Truncate tool list if needed (leave space for "─| " prefix and " |" suffix)
            max_tool_width = console.console.size.width - 10  # Reserve space for separators
            truncated_tool_list = self._truncate_list_if_needed(display_tool_list, max_tool_width)

            # Create the separator line: ─| [tools] |──────
            line1 = Text()
            line1.append("─| ", style="dim")
            line1.append_text(truncated_tool_list)
            line1.append(" |", style="dim")
            remaining = console.console.size.width - line1.cell_len
            if remaining > 0:
                line1.append("─" * remaining, style="dim")
        else:
            # No tools - continuous bar
            line1 = Text()
            line1.append("─" * console.console.size.width, style="dim")

        console.console.print(line1, markup=self._markup)
        console.console.print()

    async def show_tool_update(self, aggregator: MCPAggregator | None, updated_server: str) -> None:
        """Show a tool update for a server in the new visual style."""
        if not self.config or not self.config.logger.show_tools:
            return

        # Check if aggregator is actually an agent (has name attribute)
        agent_name = None
        if aggregator and hasattr(aggregator, "name") and aggregator.name:
            agent_name = aggregator.name

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

    def _format_tool_list(self, available_tools, selected_tool_name):
        """Format the list of available tools, highlighting the selected one."""
        display_tool_list = Text()
        matching_tools = []

        for display_tool in available_tools:
            # Handle both OpenAI and Anthropic tool formats
            if isinstance(display_tool, dict):
                if "function" in display_tool:
                    # OpenAI format
                    tool_call_name = display_tool["function"]["name"]
                else:
                    # Anthropic format
                    tool_call_name = display_tool["name"]
            else:
                # Handle potential object format (e.g., Pydantic models)
                tool_call_name = (
                    display_tool.function.name
                    if hasattr(display_tool, "function")
                    else display_tool.name
                )

            parts = (
                tool_call_name.split(SEP)
                if SEP in tool_call_name
                else [tool_call_name, tool_call_name]
            )

            if selected_tool_name.split(SEP)[0] == parts[0]:
                shortened_name = parts[1] if len(parts[1]) <= 12 else parts[1][:11] + "…"
                matching_tools.append((shortened_name, tool_call_name))

        # Format with pipe separators instead of brackets
        for i, (shortened_name, tool_call_name) in enumerate(matching_tools):
            if i > 0:
                display_tool_list.append(" | ", style="dim")
            style = "magenta" if tool_call_name == selected_tool_name else "dim"
            display_tool_list.append(shortened_name, style)

        return display_tool_list

    def _truncate_list_if_needed(self, text_list: Text, max_width: int) -> Text:
        """Truncate a Text list if it exceeds the maximum width."""
        if text_list.cell_len <= max_width:
            return text_list

        # Create a new truncated version
        truncated = Text()
        current_width = 0

        for span in text_list._spans:
            text_part = text_list.plain[span.start : span.end]
            if current_width + len(text_part) <= max_width - 1:  # -1 for ellipsis
                truncated.append(text_part, style=span.style)
                current_width += len(text_part)
            else:
                # Add what we can fit and ellipsis
                remaining = max_width - current_width - 1
                if remaining > 0:
                    truncated.append(text_part[:remaining], style=span.style)
                truncated.append("…", style="dim")
                break

        return truncated

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
        aggregator=None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
        name: str | None = None,
        model: str | None = None,
    ) -> None:
        """Display an assistant message in a formatted panel."""

        if not self.config or not self.config.logger.show_chat:
            return

        # Build server list for bottom separator (using same logic as legacy)
        display_server_list = Text()

        if aggregator:
            # Add human input tool if available
            tools = await aggregator.list_tools()
            if any(tool.name == HUMAN_INPUT_TOOL_NAME for tool in tools.tools):
                style = (
                    "green" if highlight_namespaced_tool == HUMAN_INPUT_TOOL_NAME else "dim white"
                )
                display_server_list.append("[human] ", style)

            # Add all available servers
            mcp_server_name = (
                highlight_namespaced_tool.split(SEP)[0]
                if SEP in highlight_namespaced_tool
                else highlight_namespaced_tool
            )

            for server_name in await aggregator.list_servers():
                style = "green" if server_name == mcp_server_name else "dim white"
                display_server_list.append(f"[{server_name}] ", style)

        # Combined separator and status line
        left = f"[green]▎[/green][dim green]◀[/dim green]{f' [bold green]{name}[/bold green]' if name else ''}"
        right = f"[dim]{model}[/dim]" if model else ""
        self._create_combined_separator_status(left, right)

        if isinstance(message_text, str):
            content = message_text

            # Try to detect and pretty print JSON
            try:
                import json

                json.loads(content)
                json = JSON(message_text)
                console.console.print(json, markup=self._markup)
            except (JSONDecodeError, TypeError, ValueError):
                # Use the smart rendering helper to handle XML vs Markdown
                self._render_content_smartly(content)
        else:
            # Handle Rich Text objects directly
            console.console.print(message_text, markup=self._markup)

        # Bottom separator with server list and diagrams
        console.console.print()

        # Check for mermaid diagrams in the message content
        diagrams = []
        if isinstance(message_text, str):
            diagrams = extract_mermaid_diagrams(message_text)

        # Create server list with pipe separators (no "mcp:" prefix)
        server_content = Text()
        if display_server_list and len(display_server_list) > 0:
            # Convert the existing server list to pipe-separated format
            servers = []
            if aggregator:
                for server_name in await aggregator.list_servers():
                    servers.append(server_name)

            # Create pipe-separated server list
            for i, server_name in enumerate(servers):
                if i > 0:
                    server_content.append(" | ", style="dim")
                # Highlight active server, dim inactive ones
                mcp_server_name = (
                    highlight_namespaced_tool.split(SEP)[0]
                    if SEP in highlight_namespaced_tool
                    else highlight_namespaced_tool
                )
                style = "bright_green" if server_name == mcp_server_name else "dim"
                server_content.append(server_name, style)

        # Create main separator line
        line1 = Text()
        if server_content.cell_len > 0:
            line1.append("─| ", style="dim")
            line1.append_text(server_content)
            line1.append(" |", style="dim")
            remaining = console.console.size.width - line1.cell_len
            if remaining > 0:
                line1.append("─" * remaining, style="dim")
        else:
            # No servers - continuous bar (no break)
            line1.append("─" * console.console.size.width, style="dim")

        console.console.print(line1, markup=self._markup)

        # Add diagram links in panel if any diagrams found
        if diagrams:
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
                    diagram_content.append(
                        f"{i} - {diagram.title}", style=f"bright_blue link {url}"
                    )
                else:
                    # Try to detect diagram type, fallback to "Diagram N"
                    diagram_type = detect_diagram_type(diagram.content)
                    if diagram_type != "Diagram":
                        diagram_content.append(
                            f"{i} - {diagram_type}", style=f"bright_blue link {url}"
                        )
                    else:
                        diagram_content.append(f"Diagram {i}", style=f"bright_blue link {url}")

            # Display diagrams on a simple new line (more space efficient)
            console.console.print()
            console.console.print(diagram_content, markup=self._markup)

        console.console.print()

    def show_user_message(
        self, message, model: str | None = None, chat_turn: int = 0, name: str | None = None
    ) -> None:
        """Display a user message in the new visual style."""

        if not self.config or not self.config.logger.show_chat:
            return

        # Combined separator and status line
        left = f"[blue]▎[/blue][dim blue]▶[/dim blue]{f' [bold blue]{name}[/bold blue]' if name else ''}"

        # Build right side with model and turn
        right_parts = []
        if model:
            right_parts.append(model)
        if chat_turn > 0:
            right_parts.append(f"turn {chat_turn}")

        right = f"[dim]{' '.join(right_parts)}[/dim]" if right_parts else ""
        self._create_combined_separator_status(left, right)

        # Display content as markdown if it looks like markdown, otherwise as text
        if isinstance(message, str):
            # Use the smart rendering helper to handle XML vs Markdown
            self._render_content_smartly(message)
        else:
            # Handle Text objects directly
            console.console.print(message, markup=self._markup)

        # Bottom separator (no server list for user messages)
        console.console.print()
        console.console.print("─" * console.console.size.width, style="dim")
        console.console.print()

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: Optional[str] = None,
        message_count: int = 0,
        agent_name: Optional[str] = None,
        aggregator=None,
        arguments: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Display information about a loaded prompt template.

        Args:
            prompt_name: The name of the prompt that was loaded (should be namespaced)
            description: Optional description of the prompt
            message_count: Number of messages added to the conversation history
            agent_name: Name of the agent using the prompt
            aggregator: Optional aggregator instance to use for server highlighting
            arguments: Optional dictionary of arguments passed to the prompt template
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Get server name from the namespaced prompt_name
        mcp_server_name = None
        if SEP in prompt_name:
            # Extract the server from the namespaced prompt name
            mcp_server_name = prompt_name.split(SEP)[0]
        elif aggregator and aggregator.server_names:
            # Fallback to first server if not namespaced
            mcp_server_name = aggregator.server_names[0]

        # Build the server list with highlighting
        display_server_list = Text()
        if aggregator:
            for server_name in await aggregator.list_servers():
                style = "green" if server_name == mcp_server_name else "dim white"
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
            self._render_content_smartly(content, check_markdown_markers=True)

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
