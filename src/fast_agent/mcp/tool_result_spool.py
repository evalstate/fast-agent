"""Retain complete MCP tool results that exceed the model-facing budget."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ptc.store import AUTO_RETAINED_MAX_BYTES
from fast_agent.tools.transient_artifacts import TOOL_RESULT_ARTIFACT_MAX_BYTES

if TYPE_CHECKING:
    from mcp_types import CallToolResult

    from fast_agent.ptc.store import SandboxStore
    from fast_agent.tools.transient_artifacts import TransientArtifactStore

_PRODUCER = "tool-result"
PTC_STORE_RESULT_MAX_BYTES = AUTO_RETAINED_MAX_BYTES
_STORE_LINE_BOUNDARY_WINDOW_CHARS = 64 * 1024


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
            content=json.dumps(
                result.structured_content, ensure_ascii=False, indent=2, sort_keys=True
            ),
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


def _json_utf8_len(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8"))


def _truncate_text_for_store(text: str, max_bytes: int) -> tuple[str, bool]:
    """Return a prefix of ``text`` whose UTF-8 JSON encoding fits within ``max_bytes``."""

    if max_bytes <= 2:
        return "", False
    if _json_utf8_len(text) <= max_bytes:
        return text, True

    lo = 0
    hi = min(len(text), max_bytes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _json_utf8_len(text[:mid]) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return "", False

    window_start = max(0, lo - _STORE_LINE_BOUNDARY_WINDOW_CHARS)
    last_newline = text.rfind("\n", window_start, lo)
    cutoff = last_newline + 1 if last_newline > 0 else lo
    return text[:cutoff], False


def _describe_structured_value(value: object) -> str:
    if isinstance(value, dict):
        return f"dict, {len(value)} keys"
    if isinstance(value, list):
        return f"list, {len(value)} items"
    return type(value).__name__


def store_tool_result(
    store: SandboxStore,
    result: CallToolResult,
    *,
    tool_name: str,
    ptc_tool_name: str,
) -> str | None:
    """Keep an oversized tool result in the PTC sandbox store and return its model notice.

    Used when no model-readable spool file environment is available (e.g. PTC without shell).
    Structured content is stored whole when it fits within ``PTC_STORE_RESULT_MAX_BYTES``;
    otherwise the text payload is stored, cut on a line boundary if it exceeds the cap.
    """

    if result.structured_content is not None:
        key = store.retain_output(
            result.structured_content,
            max_bytes=PTC_STORE_RESULT_MAX_BYTES,
        )
        if key is not None:
            shape = _describe_structured_value(result.structured_content)
            return (
                f"The complete {tool_name} result ({shape}) is kept in the sandbox store "
                f"under key '{key}'. In your next {ptc_tool_name} run, read it with "
                f"`data = await load('{key}')`, inspect its shape, and print only the "
                "selected items. Do not print it whole."
            )

    text = "\n".join(text for block in result.content if (text := get_text(block)) is not None)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False, indent=2, sort_keys=True)
    if not text:
        return None

    retained_text, complete = _truncate_text_for_store(text, PTC_STORE_RESULT_MAX_BYTES)
    if not retained_text:
        return None
    key = store.retain_output(retained_text, max_bytes=PTC_STORE_RESULT_MAX_BYTES)
    if key is None:
        return None

    description = (
        f"{tool_name} result"
        if result.structured_content is None
        else f"text version of the {tool_name} result"
    )
    availability = (
        f"The complete {description} (str, {len(retained_text)} characters) is kept in the "
        f"sandbox store under key '{key}'."
        if complete
        else (
            f"The first {len(retained_text)} characters of the {description} (str) are kept "
            f"in the sandbox store under key '{key}'."
        )
    )
    return (
        f"{availability} In your next {ptc_tool_name} run, read it with "
        f"`text = await load('{key}')` and print only the part you need, e.g. "
        "`print(text[start:end])` or "
        "`print([line for line in text.splitlines() if 'needle' in line][:50])`. "
        "Do not print it whole."
    )
