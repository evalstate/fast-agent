"""Shared LLM tool-call error formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.utils.count_display import plural_label

if TYPE_CHECKING:
    from collections.abc import Sequence


def format_incomplete_tool_call_error(incomplete_tools: Sequence[str]) -> str:
    tool_call_label = plural_label(len(incomplete_tools), "tool call")
    return (
        f"Streaming completed but {tool_call_label} never finished: {', '.join(incomplete_tools)}"
    )
