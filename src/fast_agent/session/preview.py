"""Helpers for rendering lightweight session resume previews."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _pending_tool_call_summary(message: PromptMessageExtended) -> str | None:
    tool_calls = message.tool_calls or {}
    if message.stop_reason != LlmStopReason.TOOL_USE or not tool_calls:
        return None

    tool_names = [call.params.name for call in tool_calls.values() if call.params.name]
    if not tool_names:
        count = len(tool_calls)
        return "Pending tool call" if count == 1 else f"Pending tool calls ({count})"

    if len(tool_names) == 1:
        return f"Pending tool call: {tool_names[0]}"

    return f"Pending tool calls: {', '.join(tool_names)}"


def find_last_assistant_preview_text(history: list[PromptMessageExtended]) -> str | None:
    """Return the best available assistant preview from saved history."""
    for message in reversed(history):
        if message.role != "assistant":
            continue

        text = message.last_text()
        if text:
            return text

        pending_summary = _pending_tool_call_summary(message)
        if pending_summary:
            return pending_summary

    return None
