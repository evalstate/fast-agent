"""
Prompt class for easily creating and working with MCP prompt content.

This implementation lives in the fast_agent namespace as part of the
migration away from fast_agent. A compatibility shim remains at
fast_agent.core.prompt importing this Prompt.
"""

from pathlib import Path
from typing import Dict, List, Literal, Union

from mcp import CallToolRequest
from mcp.types import ContentBlock, PromptMessage

from fast_agent.mcp.mcp_content import Assistant, MCPPrompt, User
from fast_agent.types import LlmStopReason, PromptMessageExtended


class Prompt:
    """
    A helper class for working with MCP prompt content.

    This class provides static methods to create:
    - PromptMessage instances
    - PromptMessageExtended instances
    - Lists of messages for conversations

    All methods intelligently handle various content types:
    - Strings become TextContent
    - Image file paths become ImageContent
    - Other file paths become EmbeddedResource
    - TextContent objects are used directly
    - ImageContent objects are used directly
    - EmbeddedResource objects are used directly
    - Pre-formatted messages pass through unchanged
    """

    @classmethod
    def user(
        cls,
        *content_items: Union[
            str, Path, bytes, dict, ContentBlock, PromptMessage, PromptMessageExtended
        ],
    ) -> PromptMessageExtended:
        """
        Create a user PromptMessageExtended with various content items.
        """
        # Handle PromptMessage and PromptMessageExtended directly
        if len(content_items) == 1:
            item = content_items[0]
            if isinstance(item, PromptMessage):
                return PromptMessageExtended(role="user", content=[item.content])
            elif isinstance(item, PromptMessageExtended):
                # Keep the content but change role to user
                return PromptMessageExtended(role="user", content=item.content)

        # Use the content factory for other types
        messages = User(*content_items)
        return PromptMessageExtended(role="user", content=[msg["content"] for msg in messages])

    @classmethod
    def assistant(
        cls,
        *content_items: Union[
            str, Path, bytes, dict, ContentBlock, PromptMessage, PromptMessageExtended
        ],
        stop_reason: LlmStopReason | None = None,
        tool_calls: Dict[str, CallToolRequest] | None = None,
    ) -> PromptMessageExtended:
        """
        Create an assistant PromptMessageExtended with various content items.
        """
        # Handle PromptMessage and PromptMessageExtended directly
        if len(content_items) == 1:
            item = content_items[0]
            if isinstance(item, PromptMessage):
                return PromptMessageExtended(
                    role="assistant",
                    content=[item.content],
                    stop_reason=stop_reason,
                    tool_calls=tool_calls,
                )
            elif isinstance(item, PromptMessageExtended):
                # Keep the content but change role to assistant
                return PromptMessageExtended(
                    role="assistant",
                    content=item.content,
                    stop_reason=stop_reason,
                    tool_calls=tool_calls,
                )

        # Use the content factory for other types
        messages = Assistant(*content_items)
        return PromptMessageExtended(
            role="assistant",
            content=[msg["content"] for msg in messages],
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    @classmethod
    def message(
        cls,
        *content_items: Union[
            str, Path, bytes, dict, ContentBlock, PromptMessage, PromptMessageExtended
        ],
        role: Literal["user", "assistant"] = "user",
    ) -> PromptMessageExtended:
        """
        Create a PromptMessageExtended with the specified role and content items.
        """
        # Handle PromptMessage and PromptMessageExtended directly
        if len(content_items) == 1:
            item = content_items[0]
            if isinstance(item, PromptMessage):
                return PromptMessageExtended(role=role, content=[item.content])
            elif isinstance(item, PromptMessageExtended):
                # Keep the content but change role as specified
                return PromptMessageExtended(role=role, content=item.content)

        # Use the content factory for other types
        messages = MCPPrompt(*content_items, role=role)
        return PromptMessageExtended(
            role=messages[0]["role"] if messages else role,
            content=[msg["content"] for msg in messages],
        )

    @classmethod
    def conversation(cls, *messages) -> List[PromptMessage]:
        """
        Create a list of PromptMessages from various inputs.
        """
        result = []

        for item in messages:
            if isinstance(item, PromptMessageExtended):
                # Convert PromptMessageExtended to a list of PromptMessages
                result.extend(item.from_multipart())
            elif isinstance(item, dict) and "role" in item and "content" in item:
                # Convert a single message dict to PromptMessage
                result.append(PromptMessage(**item))
            elif isinstance(item, list):
                # Process each item in the list
                for msg in item:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result.append(PromptMessage(**msg))
            # Ignore other types

        return result

    @classmethod
    def from_multipart(cls, multipart: List[PromptMessageExtended]) -> List[PromptMessage]:
        """
        Convert a list of PromptMessageExtended objects to PromptMessages.
        """
        result = []
        for mp in multipart:
            result.extend(mp.from_multipart())
        return result
