"""Retain complete MCP tool results that exceed the model-facing budget."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.transient_artifacts import TOOL_RESULT_ARTIFACT_MAX_BYTES

if TYPE_CHECKING:
    from mcp_types import CallToolResult

    from fast_agent.tools.transient_artifacts import TransientArtifactStore

_PRODUCER = "tool-result"


async def spool_tool_result(
    store: TransientArtifactStore,
    result: CallToolResult,
    *,
    tool_name: str,
) -> str | None:
    """Write the complete result to a transient artifact and return its model-facing notice.

    Structured content is retained as indented JSON so it can be read by line range,
    and only when it fits whole: partial JSON is not parseable. Otherwise the text
    content is retained, cut on a line boundary if it exceeds the limit.
    """

    if result.structured_content is not None:
        retained = await store.write_complete_text(
            producer=_PRODUCER,
            suffix=".json",
            content=json.dumps(result.structured_content, ensure_ascii=False, indent=2),
            description=f"{tool_name} result",
            max_bytes=TOOL_RESULT_ARTIFACT_MAX_BYTES,
        )
        if retained is not None:
            return retained.notice

    text = "\n".join(text for block in result.content if (text := get_text(block)) is not None)
    if not text:
        return None
    retained = await store.write_text(
        producer=_PRODUCER,
        suffix=".txt",
        content=text,
        description=(
            f"{tool_name} result"
            if result.structured_content is None
            else f"text version of the {tool_name} result"
        ),
        max_bytes=TOOL_RESULT_ARTIFACT_MAX_BYTES,
    )
    return retained.notice
