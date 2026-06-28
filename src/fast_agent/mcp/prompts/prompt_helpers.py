"""
Helper functions for working with PromptMessage and PromptMessageExtended objects.

These utilities simplify extracting content from nested message structures
without repetitive type checking.
"""

from typing import TYPE_CHECKING

from mcp.types import PromptMessage

from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


class MessageContent:
    """
    Helper class for working with message content in both PromptMessage and
    PromptMessageExtended objects.
    """

    @staticmethod
    def get_first_text(message: "PromptMessage | PromptMessageExtended") -> str | None:
        """
        Get the first available text content from a message.

        Args:
            message: A PromptMessage or PromptMessageExtended

        Returns:
            First text content or None if no text content exists
        """
        if isinstance(message, PromptMessage):
            return get_text(message.content)

        for content in message.content:
            text = get_text(content)
            if text is not None:
                return text

        return None
