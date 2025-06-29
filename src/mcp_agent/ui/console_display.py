from typing import Optional, Union
import time

from mcp.types import CallToolResult
from rich.panel import Panel
from rich.text import Text

from mcp_agent import console
from mcp_agent.mcp.common import SEP
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.pubsub import get_pubsub_manager, PubSubChannel

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
        
        # PubSub setup
        self.pubsub_channel: Optional[PubSubChannel] = None
        print(f"Pubsub Channel:{self.pubsub_channel}")
        print(f"Config current: {self.config}")

        if config and hasattr(config, 'pubsub_config'):
            pubsub_config = getattr(config, 'pubsub_config', {})
            print(f"what is pubsub config? {pubsub_config}")
            channel_name = pubsub_config.get('channel_name', 'console_display')
            print(f"what is pubsub channel name? {channel_name}")

            backend = pubsub_config.get('backend', 'memory')
            print(f"what is backend? {backend}")

            backend_config = None
            
            if backend == 'redis':
                backend_config = pubsub_config.get('redis', None)
            elif backend == 'msk':
                print("THIS IS CORRECT ANSWER")
                backend_config = pubsub_config.get('msk', None)
            elif backend == 'kafka':
                backend_config = pubsub_config.get('kafka', None)
            
            self.pubsub_manager = get_pubsub_manager(backend=backend, backend_config=backend_config)
            self.pubsub_channel = self.pubsub_manager.get_or_create_channel(channel_name)

    async def show_tool_result(self, result: CallToolResult) -> None:
        """Display a tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        style = "red" if result.isError else "magenta"
        print("THIS SHOULD SHOW UP")

        panel = Panel(
            Text(str(result.content), overflow="..."),
            title="[TOOL RESULT]",
            title_align="right",
            style=style,
            border_style="bold white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result.content)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            print("THIS SHOULD SHOW UP")
            print(result.content)

            message_data = {
                "type": "tool_result",
                "content": str(result.content),
                "is_error": result.isError,
                "style": style,
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

    async def show_oai_tool_result(self, result) -> None:
        """Display an OpenAI tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        panel = Panel(
            Text(str(result), overflow="..."),
            title="[TOOL RESULT]",
            title_align="right",
            style="magenta",
            border_style="bold white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            print("THIS SHOULD SHOW UP")
            print(result.content)
            message_data = {
                "type": "oai_tool_result",
                "content": str(result),
                "style": "magenta",
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

    async def show_tool_call(self, available_tools, tool_name, tool_args) -> None:
        """Display a tool call in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        display_tool_list = self._format_tool_list(available_tools, tool_name)

        panel = Panel(
            Text(str(tool_args), overflow="ellipsis"),
            title="[TOOL CALL]",
            title_align="left",
            style="magenta",
            border_style="bold white",
            subtitle=display_tool_list,
            subtitle_align="left",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(tool_args)) > 360:
                panel.height = 8

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            print("THIS SHOULD SHOW UP")
            print(result.content)
            message_data = {
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "available_tools": [str(tool) for tool in available_tools],
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

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
            border_style="bold white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print("\n")
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            server_list = []
            if aggregator:
                server_list = await aggregator.list_servers()
            
            message_data = {
                "type": "tool_update",
                "updated_server": updated_server,
                "server_list": server_list,
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

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
            border_style="bold white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            server_list = []
            if aggregator:
                server_list = await aggregator.list_servers()
            
            message_data = {
                "type": "assistant_message",
                "message": str(message_text),
                "title": title,
                "name": name,
                "highlight_tool": highlight_namespaced_tool,
                "server_list": server_list,
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

    async def show_user_message(
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
            border_style="bold white",
            padding=(1, 2),
            subtitle=subtitle_text,
            subtitle_align="left",
        )
        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            message_data = {
                "type": "user_message",
                "message": str(message),
                "model": model,
                "chat_turn": chat_turn,
                "name": name,
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)

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
            border_style="bold white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )

        console.console.print(panel, markup=self._markup)
        console.console.print("\n")
        
        # Publish to pubsub if channel is available
        if self.pubsub_channel:
            server_list = []
            if aggregator:
                server_list = await aggregator.list_servers()
            
            message_data = {
                "type": "prompt_loaded",
                "prompt_name": prompt_name,
                "description": description,
                "message_count": message_count,
                "agent_name": agent_name,
                "arguments": arguments,
                "server_list": server_list,
                "timestamp": time.time()
            }
            await self.pubsub_channel.publish(message_data)
