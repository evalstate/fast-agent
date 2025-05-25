"""
PubSubDisplay wrapper for ConsoleDisplay.

This module provides a wrapper around the ConsoleDisplay class that publishes
all display output to a pub/sub channel in addition to the console.
"""

from typing import Any, Dict, List, Optional, Union

from mcp.types import CallToolResult
from rich.text import Text

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.mcp.pubsub import PubSubChannel, get_pubsub_manager
from mcp_agent.mcp.pubsub_formatter import PubSubFormatter
from mcp_agent.ui.console_display import ConsoleDisplay

logger = get_logger(__name__)


class PubSubDisplay:
    """
    Wrapper around ConsoleDisplay that publishes all display output to a pub/sub channel.
    
    This class implements the same interface as ConsoleDisplay but adds pub/sub publishing.
    All methods delegate to the wrapped ConsoleDisplay instance for console output.
    """

    def __init__(self, config=None, channel_id: Optional[str] = None) -> None:
        """
        Initialize the PubSub display wrapper.
        
        Args:
            config: Configuration object containing display preferences
            channel_id: Optional channel ID to use for pub/sub messages
        """
        self.console_display = ConsoleDisplay(config=config)
        self.config = config
        
        # Extract PubSub configuration from config if available
        use_redis = False
        redis_config = None
        
        if config and hasattr(config, 'pubsub_config'):
            pubsub_config = getattr(config, 'pubsub_config', {})
            
            # Get custom channel ID if provided
            if channel_id is None and 'channel_name' in pubsub_config:
                channel_id = f"agent_{pubsub_config['channel_name']}"
                
            # Get Redis configuration
            use_redis = pubsub_config.get('use_redis', False)
            if use_redis and 'redis' in pubsub_config:
                redis_config = pubsub_config['redis']
        
        # PubSub setup
        self.pubsub_manager = get_pubsub_manager(use_redis=use_redis, redis_config=redis_config)
        self.channel: Optional[PubSubChannel] = None
        self.channel_id = channel_id
        
        if channel_id:
            self.set_channel(channel_id)
        
        logger_message = f"Initialized PubSubDisplay with channel_id: {channel_id}"
        if use_redis:
            logger_message += " (using Redis)"
        logger.debug(logger_message)

    def set_channel(self, channel_id: str) -> None:
        """
        Set the pub/sub channel for this display.
        
        Args:
            channel_id: Channel identifier
        """
        self.channel_id = channel_id
        self.channel = self.pubsub_manager.get_or_create_channel(channel_id)
        logger.debug(f"Set PubSubDisplay channel to: {channel_id}")

    async def show_tool_result(self, result: CallToolResult) -> None:
        """
        Display a tool result in a formatted panel and publish to pub/sub.
        
        Args:
            result: The tool result to display
        """
        # Show in console
        self.console_display.show_tool_result(result)
        
        # Publish to pub/sub if channel is set
        if self.channel:
            message = PubSubFormatter.format_tool_result(
                result=result,
                channel_id=self.channel_id
            )
            await self.channel.publish(message)
            logger.debug(f"Published tool result to channel: {self.channel_id}")

    async def show_oai_tool_result(self, result) -> None:
        """
        Display an OpenAI tool result in a formatted panel and publish to pub/sub.
        
        Args:
            result: The tool result to display
        """
        # Show in console
        self.console_display.show_oai_tool_result(result)
        
        # Publish to pub/sub if channel is set
        if self.channel:
            formatted_content = {
                "content": str(result),
                "isError": False,
            }
            message = PubSubFormatter.format_raw(
                message_type="tool_result",
                content=formatted_content,
                channel_id=self.channel_id
            )
            await self.channel.publish(message)
            logger.debug(f"Published OAI tool result to channel: {self.channel_id}")

    async def show_tool_call(self, available_tools, tool_name, tool_args) -> None:
        """
        Display a tool call in a formatted panel and publish to pub/sub.
        
        Args:
            available_tools: List of available tools
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
        """
        # Show in console
        self.console_display.show_tool_call(available_tools, tool_name, tool_args)
        
        # Publish to pub/sub if channel is set
        if self.channel:
            message = PubSubFormatter.format_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                available_tools=available_tools,
                channel_id=self.channel_id
            )
            await self.channel.publish(message)
            logger.debug(f"Published tool call to channel: {self.channel_id}")

    async def show_assistant_message(
        self,
        message_text: Union[str, Text],
        aggregator: Optional[MCPAggregator] = None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
        name: Optional[str] = None,
    ) -> None:
        """
        Display an assistant message in a formatted panel and publish to pub/sub.
        
        Args:
            message_text: The message text to display
            aggregator: Optional aggregator for server information
            highlight_namespaced_tool: Optional tool to highlight
            title: Title for the message panel
            name: Optional agent name
        """
        # Show in console
        await self.console_display.show_assistant_message(
            message_text=message_text,
            aggregator=aggregator,
            highlight_namespaced_tool=highlight_namespaced_tool,
            title=title,
            name=name
        )
        
        # Publish to pub/sub if channel is set
        if self.channel:
            metadata = {
                "highlight_tool": highlight_namespaced_tool,
                "title": title,
                "name": name
            }
            message = PubSubFormatter.format_assistant_message(
                message=message_text,
                channel_id=self.channel_id,
                metadata=metadata
            )
            await self.channel.publish(message)
            logger.debug(f"Published assistant message to channel: {self.channel_id}")

    async def show_user_message(
        self, message, model: Optional[str], chat_turn: int, name: Optional[str] = None
    ) -> None:
        """
        Display a user message in a formatted panel and publish to pub/sub.
        
        Args:
            message: The message content
            model: Model identifier
            chat_turn: Current chat turn number
            name: Optional user name
        """
        # Show in console
        self.console_display.show_user_message(message, model, chat_turn, name)
        
        # Publish to pub/sub if channel is set
        if self.channel:
            metadata = {
                "model": model,
                "chat_turn": chat_turn,
                "name": name
            }
            message_obj = PubSubFormatter.format_user_message(
                message=message,
                channel_id=self.channel_id,
                metadata=metadata
            )
            await self.channel.publish(message_obj)
            logger.debug(f"Published user message to channel: {self.channel_id}")

    async def show_prompt_loaded(
        self,
        prompt_name: str,
        description: Optional[str] = None,
        message_count: int = 0,
        agent_name: Optional[str] = None,
        aggregator: Optional[MCPAggregator] = None,
        arguments: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Display information about a loaded prompt template and publish to pub/sub.
        
        Args:
            prompt_name: The name of the prompt that was loaded
            description: Optional description of the prompt
            message_count: Number of messages added to the conversation history
            agent_name: Name of the agent using the prompt
            aggregator: Optional aggregator instance to use for server highlighting
            arguments: Optional dictionary of arguments passed to the prompt template
        """
        # Show in console
        await self.console_display.show_prompt_loaded(
            prompt_name=prompt_name,
            description=description,
            message_count=message_count,
            agent_name=agent_name,
            aggregator=aggregator,
            arguments=arguments
        )
        
        # Publish to pub/sub if channel is set
        if self.channel:
            content = {
                "prompt_name": prompt_name,
                "description": description,
                "message_count": message_count,
                "agent_name": agent_name,
                "arguments": arguments
            }
            message = PubSubFormatter.format_raw(
                message_type="prompt_loaded",
                content=content,
                channel_id=self.channel_id
            )
            await self.channel.publish(message)
            logger.debug(f"Published prompt loaded info to channel: {self.channel_id}")

    async def show_tool_update(self, aggregator: Optional[MCPAggregator], updated_server: str) -> None:
        """
        Show a tool update for a server and publish to pub/sub.
        
        Args:
            aggregator: Aggregator instance
            updated_server: Name of the updated server
        """
        # Show in console
        await self.console_display.show_tool_update(aggregator, updated_server)
        
        # Publish to pub/sub if channel is set
        if self.channel:
            content = {
                "updated_server": updated_server
            }
            message = PubSubFormatter.format_raw(
                message_type="tool_update",
                content=content,
                channel_id=self.channel_id
            )
            await self.channel.publish(message)
            logger.debug(f"Published tool update info to channel: {self.channel_id}")