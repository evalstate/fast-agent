"""
Formatter module for PubSub system.

This module handles the formatting of messages published through the PubSub system,
converting different types of content into a standardized format.
"""

import json
from typing import Any, Dict, List, Optional, Union

from mcp.types import (
    CallToolResult,
    PromptMessage,
    TextContent,
)
from rich.text import Text

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)


class PubSubMessage:
    """
    Standardized message format for the PubSub system.
    """

    def __init__(
        self,
        message_type: str,
        content: Any,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a PubSub message.
        
        Args:
            message_type: Type of message (e.g., "user", "assistant", "tool_call", "tool_result")
            content: Message content (can be text, structured data, etc.)
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
        """
        self.message_type = message_type
        self.content = content
        self.channel_id = channel_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "type": self.message_type,
            "content": self.content,
            "channel_id": self.channel_id,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """
        String representation of the message.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())


class PubSubFormatter:
    """
    Formatter for converting various content types into PubSubMessages.
    """

    @staticmethod
    def format_user_message(
        message: Union[str, PromptMessage, PromptMessageMultipart],
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PubSubMessage:
        """
        Format a user message.
        
        Args:
            message: User message content
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
            
        Returns:
            Formatted PubSubMessage
        """
        if isinstance(message, str):
            content = message
        elif isinstance(message, PromptMessage):
            content = message.content
        elif isinstance(message, PromptMessageMultipart):
            content = message.text_content()
        else:
            content = str(message)

        return PubSubMessage(
            message_type="user",
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )

    @staticmethod
    def format_assistant_message(
        message: Union[str, Text, PromptMessage, PromptMessageMultipart],
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PubSubMessage:
        """
        Format an assistant message.
        
        Args:
            message: Assistant message content
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
            
        Returns:
            Formatted PubSubMessage
        """
        if isinstance(message, str):
            content = message
        elif isinstance(message, Text):
            content = str(message)
        elif isinstance(message, PromptMessage):
            content = message.content
        elif isinstance(message, PromptMessageMultipart):
            content = message.text_content()
        else:
            content = str(message)

        return PubSubMessage(
            message_type="assistant",
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )

    @staticmethod
    def format_tool_call(
        tool_name: str,
        tool_args: Any,
        available_tools: Optional[List[Any]] = None,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PubSubMessage:
        """
        Format a tool call.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            available_tools: Optional list of available tools
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
            
        Returns:
            Formatted PubSubMessage
        """
        content = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "available_tools": available_tools,
        }

        return PubSubMessage(
            message_type="tool_call",
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )

    @staticmethod
    def format_tool_result(
        result: CallToolResult,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PubSubMessage:
        """
        Format a tool result.
        
        Args:
            result: Result of a tool call
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
            
        Returns:
            Formatted PubSubMessage
        """
        # Extract text content from the result
        content_text = ""
        if result.content:
            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    content_text += content_item.text

        formatted_content = {
            "content": content_text,
            "isError": result.isError,
        }

        return PubSubMessage(
            message_type="tool_result",
            content=formatted_content,
            channel_id=channel_id,
            metadata=metadata,
        )

    @staticmethod
    def format_raw(
        message_type: str,
        content: Any,
        channel_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PubSubMessage:
        """
        Format raw content as a PubSubMessage.
        
        Args:
            message_type: Type of message
            content: Raw message content
            channel_id: Optional channel identifier
            metadata: Optional additional metadata
            
        Returns:
            Formatted PubSubMessage
        """
        return PubSubMessage(
            message_type=message_type,
            content=content,
            channel_id=channel_id,
            metadata=metadata,
        )