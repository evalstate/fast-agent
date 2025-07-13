from typing import Optional, Union

from mcp.types import CallToolResult
from rich.panel import Panel
from rich.text import Text

from mcp_agent import console
from mcp_agent.mcp.common import SEP
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

# Constants
HUMAN_INPUT_TOOL_NAME = "__human_input__"


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

    def show_tool_result(self, result: CallToolResult, name: Optional[str] = None) -> None:
        """Display a tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        style = "red" if result.isError else "magenta"

        panel = Panel(
            Text(str(result.content), overflow="..."),
            title=f"[TOOL RESULT]{f' ({name})' if name else ''}",
            title_align="right",
            style=style,
            border_style="white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result.content)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    def show_oai_tool_result(self, result, name: Optional[str] = None) -> None:
        """Display an OpenAI tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        panel = Panel(
            Text(str(result), overflow="..."),
            title=f"[TOOL RESULT]{f' ({name})' if name else ''}",
            title_align="right",
            style="magenta",
            border_style="white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    def show_tool_call(
        self, available_tools, tool_name, tool_args, name: Optional[str] = None
    ) -> None:
        """Display a tool call in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        display_tool_list = self._format_tool_list(available_tools, tool_name)

        panel = Panel(
            Text(str(tool_args), overflow="ellipsis"),
            title=f"[TOOL CALL]{f' ({name})' if name else ''}",
            title_align="left",
            style="magenta",
            border_style="white",
            subtitle=display_tool_list,
            subtitle_align="left",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(tool_args)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    async def show_tool_update(self, aggregator: MCPAggregator | None, updated_server: str) -> None:
        """Show a tool update for a server"""
        if not self.config or not self.config.logger.show_tools:
            return

        display_server_list = Text()

        if aggregator:
            for server_name in await aggregator.list_servers():
                style = "green" if updated_server == server_name else "dim white"
                display_server_list.append(f"[{server_name}] ", style)

        panel = Panel(
            f"[dim green]Updating tools for server {updated_server}[/]",
            title="[TOOL UPDATE]",
            title_align="left",
            style="green",
            border_style="white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print("\n")
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    def _format_tool_list(self, available_tools, selected_tool_name):
        """Format the list of available tools, highlighting the selected one."""
        display_tool_list = Text()
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
                style = "magenta" if tool_call_name == selected_tool_name else "dim white"
                shortened_name = parts[1] if len(parts[1]) <= 12 else parts[1][:11] + "…"
                display_tool_list.append(f"[{shortened_name}] ", style)

        return display_tool_list

    async def show_assistant_message(
        self,
        message_text: Union[str, Text],
        aggregator=None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
        name: Optional[str] = None,
    ) -> None:
        """Display an assistant message in a formatted panel."""
        if not self.config or not self.config.logger.show_chat:
            return

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

        panel = Panel(
            message_text,
            title=f"[{title}]{f' ({name})' if name else ''}",
            title_align="left",
            style="green",
            border_style="white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

    def show_user_message(
        self, message, model: Optional[str], chat_turn: int, name: Optional[str] = None
    ) -> None:
        """Display a user message in a formatted panel."""
        if not self.config or not self.config.logger.show_chat:
            return

        subtitle_text = Text(f"{model or 'unknown'}", style="dim white")
        if chat_turn > 0:
            subtitle_text.append(f" turn {chat_turn}", style="dim white")

        panel = Panel(
            message,
            title=f"{f'({name}) [USER]' if name else '[USER]'}",
            title_align="right",
            style="blue",
            border_style="white",
            padding=(1, 2),
            subtitle=subtitle_text,
            subtitle_align="left",
        )
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")

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
        from rich.markdown import Markdown
        from rich.text import Text
        
        if self.config and not self.config.logger.show_chat:
            return
            
        if not parallel_agent or not hasattr(parallel_agent, 'fan_out_agents'):
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
            model = 'unknown'
            if hasattr(agent, '_llm') and agent._llm and hasattr(agent._llm, 'default_request_params'):
                model = getattr(agent._llm.default_request_params, 'model', 'unknown')
            
            # Get usage information
            tokens = 0
            tool_calls = 0
            if hasattr(agent, 'usage_accumulator') and agent.usage_accumulator:
                summary = agent.usage_accumulator.get_summary()
                tokens = summary.get('cumulative_input_tokens', 0) + summary.get('cumulative_output_tokens', 0)
                tool_calls = summary.get('cumulative_tool_calls', 0)
            
            agent_results.append({
                'name': agent.name,
                'model': model,
                'content': content,
                'tokens': tokens,
                'tool_calls': tool_calls
            })
        
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
            if result['tokens'] > 0:
                right_parts.append(f"{result['tokens']:,} tokens")
            if result['tool_calls'] > 0:
                right_parts.append(f"{result['tool_calls']} tools")
            
            right = f"[dim]{' • '.join(right_parts) if right_parts else 'no usage data'}[/dim]"
            
            # Calculate padding to right-align usage info
            width = console.console.size.width
            left_text = Text.from_markup(left)
            right_text = Text.from_markup(right)
            padding = max(1, width - left_text.cell_len - right_text.cell_len)
            
            console.console.print(left + " " * padding + right, markup=self._markup)
            console.console.print()
            
            # Display content as markdown if it looks like markdown, otherwise as text
            content = result['content']
            if any(marker in content for marker in ['##', '**', '*', '`', '---', '###']):
                md = Markdown(content)
                console.console.print(md, markup=self._markup)
            else:
                console.console.print(content, markup=self._markup)
        
        # Summary
        console.console.print()
        console.console.print("─" * console.console.size.width, style="dim")
        
        total_tokens = sum(result['tokens'] for result in agent_results)
        total_tools = sum(result['tool_calls'] for result in agent_results)
        
        summary_parts = [f"{len(agent_results)} models"]
        if total_tokens > 0:
            summary_parts.append(f"{total_tokens:,} tokens")
        if total_tools > 0:
            summary_parts.append(f"{total_tools} tools")
        
        summary_text = " • ".join(summary_parts)
        console.console.print(f"[dim]{summary_text}[/dim]")
        console.console.print()
