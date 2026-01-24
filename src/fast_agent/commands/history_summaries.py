"""History summary helpers for command renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@dataclass(slots=True)
class HistoryMessageSnippet:
    role: str
    snippet: str


@dataclass(slots=True)
class HistoryOverview:
    message_count: int
    user_message_count: int
    assistant_message_count: int
    tool_calls: int
    tool_successes: int
    tool_errors: int
    recent_messages: list[HistoryMessageSnippet]


def _extract_message_text(message: "PromptMessageExtended") -> str:
    if hasattr(message, "all_text"):
        text = message.all_text() or message.first_text() or ""
    else:
        content = getattr(message, "content", None)
        if isinstance(content, list) and content:
            text = get_text(content[0]) or ""
        else:
            text = ""
    return text


def build_history_overview(
    messages: list["PromptMessageExtended"],
    *,
    recent_count: int = 5,
) -> HistoryOverview:
    summary = ConversationSummary(messages=messages)
    recent_messages: list[HistoryMessageSnippet] = []

    if recent_count > 0 and messages:
        for message in messages[-recent_count:]:
            role = getattr(message, "role", "message")
            if hasattr(role, "value"):
                role = role.value

            text = _extract_message_text(message)
            snippet = " ".join(text.split())
            if not snippet:
                snippet = "(no text content)"
            if len(snippet) > 60:
                snippet = f"{snippet[:57]}..."
            recent_messages.append(HistoryMessageSnippet(role=str(role), snippet=snippet))

    return HistoryOverview(
        message_count=summary.message_count,
        user_message_count=summary.user_message_count,
        assistant_message_count=summary.assistant_message_count,
        tool_calls=summary.tool_calls,
        tool_successes=summary.tool_successes,
        tool_errors=summary.tool_errors,
        recent_messages=recent_messages,
    )
