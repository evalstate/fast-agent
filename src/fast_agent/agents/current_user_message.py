"""Task-local external user input available to built-in tools."""

from __future__ import annotations

from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_types import ContentBlock

    from fast_agent.types import PromptMessageExtended


@dataclass(frozen=True, slots=True)
class CurrentUserMessage:
    """Content-only snapshot of the current eligible external user message."""

    content: tuple[ContentBlock, ...]


_current_user_message: ContextVar[CurrentUserMessage | None] = ContextVar(
    "current_user_message",
    default=None,
)


def snapshot_current_user_message(
    messages: list[PromptMessageExtended],
) -> CurrentUserMessage | None:
    """Copy the latest current external user message without its metadata."""
    for message in reversed(messages):
        if message.role == "user" and not message.tool_results and not message.is_template:
            return CurrentUserMessage(content=tuple(deepcopy(message.content)))
    return None


def set_current_user_message(
    message: CurrentUserMessage | None,
) -> Token[CurrentUserMessage | None]:
    """Set the current task's external user message."""
    return _current_user_message.set(message)


def reset_current_user_message(token: Token[CurrentUserMessage | None]) -> None:
    """Restore the parent task's external user message."""
    _current_user_message.reset(token)


def get_current_user_message() -> CurrentUserMessage | None:
    """Return the current task's external user message, if one exists."""
    return _current_user_message.get()
