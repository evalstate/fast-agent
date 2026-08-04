from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.constants import FAST_AGENT_SUBAGENT_RESULT_METADATA
from fast_agent.mcp.helpers.content_helpers import tool_result_text_for_llm

if TYPE_CHECKING:
    from mcp_types import CallToolResult


@dataclass(frozen=True, slots=True)
class SubagentResultPresentation:
    message_text: str
    name: str
    model: str | None
    bottom_items: list[str] | None
    highlight_indexes: list[int] | None


def build_subagent_result_presentation(
    result: CallToolResult,
) -> SubagentResultPresentation:
    meta = result.meta
    details = meta.get(FAST_AGENT_SUBAGENT_RESULT_METADATA) if isinstance(meta, Mapping) else None
    alias = details.get("alias") if isinstance(details, Mapping) else None
    label = details.get("label") if isinstance(details, Mapping) else None
    child_name = details.get("child_agent_name") if isinstance(details, Mapping) else None
    model_spec = details.get("model_spec") if isinstance(details, Mapping) else None
    child_session_id = details.get("child_session_id") if isinstance(details, Mapping) else None
    display_label = (
        alias if isinstance(alias, str) else label if isinstance(label, str) else child_name
    )
    bottom_items = [f"session {child_session_id}"] if isinstance(child_session_id, str) else None
    return SubagentResultPresentation(
        message_text=tool_result_text_for_llm(result),
        name=f"subagent: {display_label}" if isinstance(display_label, str) else "subagent",
        model=model_spec if isinstance(model_spec, str) else None,
        bottom_items=bottom_items,
        highlight_indexes=[0] if bottom_items else None,
    )
