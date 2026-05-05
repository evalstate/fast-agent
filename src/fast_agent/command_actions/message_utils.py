"""Message helpers useful to plugin command actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import TextContent

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


def replace_last_text(message: PromptMessageExtended, text: str) -> bool:
    """Replace the last text content block on a message, adding one if absent."""
    for index in range(len(message.content) - 1, -1, -1):
        block = message.content[index]
        if isinstance(block, TextContent):
            message.content[index] = TextContent(type="text", text=text)
            return True

    message.content.append(TextContent(type="text", text=text))
    return False
