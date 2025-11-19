from __future__ import annotations

from enum import Enum


class MessageType(Enum):
    """Types of messages that can be displayed."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


MESSAGE_CONFIGS: dict[MessageType, dict[str, str]] = {
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
        "block_color": "magenta",
        "arrow": "▶",
        "arrow_style": "dim magenta",
        "highlight_color": "magenta",
    },
}


__all__ = ["MessageType", "MESSAGE_CONFIGS"]
