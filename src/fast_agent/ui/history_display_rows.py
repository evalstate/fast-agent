"""History display row extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.constants import (
    FAST_AGENT_COMPACTION_CHANNEL,
    FAST_AGENT_TIMING,
    FAST_AGENT_TOOL_TIMING,
)
from fast_agent.history.tool_activities import remote_tool_activities
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.ui.history_display_models import HistoryDisplayRow, ToolResultSummary
from fast_agent.utils.text import collapse_whitespace, strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mcp.types import CallToolRequest, CallToolResult

    from fast_agent.types import PromptMessageExtended

NON_TEXT_MARKER = "^"
TOOL_DETAIL_STYLE = "dim magenta"


@dataclass(frozen=True, slots=True)
class _TextSummary:
    normalized: str
    chars: int
    preview: str
    non_text: bool


@dataclass(frozen=True, slots=True)
class _ToolResultRows:
    rows: list[HistoryDisplayRow]
    names: list[str]
    total_chars: int
    has_non_text: bool
    has_error: bool
    timing_ms: float | str | None


@dataclass(frozen=True, slots=True)
class _ProviderRows:
    rows: list[HistoryDisplayRow]
    total_chars: int
    has_non_text: bool
    has_error: bool


def _format_tool_detail(prefix: str, names: Sequence[str]) -> Text:
    detail = Text(prefix, style=TOOL_DETAIL_STYLE)
    if names:
        detail.append(", ".join(names), style=TOOL_DETAIL_STYLE)
    return detail


def _preview_text(value: str | None, limit: int = 80) -> str:
    normalized = collapse_whitespace(value)
    if not normalized:
        return "<no text>"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _has_non_text_content(message: "PromptMessageExtended") -> bool:
    for block in message.content:
        block_type = block.type
        if block_type and block_type != "text":
            return True
    return False


def _extract_tool_result_summary(result: "CallToolResult", *, limit: int = 80) -> ToolResultSummary:
    preview: str | None = None
    total_chars = 0
    saw_non_text = False

    for block in result.content:
        text = get_text(block)
        if text:
            normalized = collapse_whitespace(text)
            if preview is None:
                preview = _preview_text(normalized, limit=limit)
            total_chars += len(normalized)
        else:
            saw_non_text = True

    if preview is not None:
        return ToolResultSummary(
            preview=preview,
            chars=total_chars,
            non_text=saw_non_text,
        )
    return ToolResultSummary(
        preview=f"{NON_TEXT_MARKER} non-text tool result",
        chars=0,
        non_text=True,
    )


def _extract_timing_ms(message: "PromptMessageExtended") -> float | None:
    """Extract timing duration in milliseconds from message channels."""
    channels = message.channels
    if not channels:
        return None

    timing_blocks = channels.get(FAST_AGENT_TIMING, [])
    if not timing_blocks:
        return None

    timing_text = get_text(timing_blocks[0])
    if not timing_text:
        return None

    try:
        timing_data = json.loads(timing_text)
        return timing_data.get("duration_ms")
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None


def _extract_tool_timings(
    message: "PromptMessageExtended",
) -> dict[str, dict[str, float | str | None]]:
    """Extract tool timing data from message channels."""
    channels = message.channels
    if not channels:
        return {}

    timing_blocks = channels.get(FAST_AGENT_TOOL_TIMING, [])
    if not timing_blocks:
        return {}

    timing_text = get_text(timing_blocks[0])
    if not timing_text:
        return {}

    try:
        raw_data = json.loads(timing_text)
        normalized: dict[str, dict[str, float | str | None]] = {}
        for tool_id, value in raw_data.items():
            if isinstance(value, dict):
                normalized[tool_id] = value
            else:
                normalized[tool_id] = {
                    "timing_ms": value,
                    "transport_channel": None,
                }
        return normalized
    except (json.JSONDecodeError, TypeError):
        return {}


def _history_row(
    *,
    role: str,
    timeline_role: str,
    chars: int,
    preview: str,
    details: Text | None,
    non_text: bool,
    has_tool_request: bool,
    hide_summary: bool,
    include_in_timeline: bool,
    is_error: bool,
    timing_ms: float | str | None,
    label: str | None = None,
    arrow: str | None = None,
) -> HistoryDisplayRow:
    return HistoryDisplayRow(
        role=role,
        timeline_role=timeline_role,
        chars=chars,
        preview=preview,
        details=details,
        non_text=non_text,
        has_tool_request=has_tool_request,
        hide_summary=hide_summary,
        include_in_timeline=include_in_timeline,
        is_error=is_error,
        timing_ms=timing_ms,
        label=label,
        arrow=arrow,
    )


def _message_role(message: "PromptMessageExtended") -> str:
    return strip_casefold(str(message.role)) if message.role else "assistant"


def _message_text_summary(message: "PromptMessageExtended") -> _TextSummary:
    try:
        text = message.first_text() or ""
    except Exception:  # pragma: no cover - defensive
        text = ""
    normalized = collapse_whitespace(text)
    chars = len(normalized)
    return _TextSummary(
        normalized=normalized,
        chars=chars,
        preview=_preview_text(text),
        non_text=_has_non_text_content(message) or chars == 0,
    )


def _tool_call_names(
    tool_calls: Mapping[str, "CallToolRequest"] | None,
    call_name_lookup: dict[str, str],
) -> list[str]:
    if not tool_calls:
        return []

    names: list[str] = []
    for call_id, call in tool_calls.items():
        params = call.params
        name = params.name or call_id
        call_name_lookup[call_id] = name
        names.append(name)
    return names


def _tool_result_rows(
    tool_results: Mapping[str, "CallToolResult"] | None,
    call_name_lookup: dict[str, str],
    tool_timings: Mapping[str, Mapping[str, float | str | None]],
) -> _ToolResultRows:
    rows: list[HistoryDisplayRow] = []
    names: list[str] = []
    total_chars = 0
    has_non_text = False
    has_error = False
    last_timing_ms: float | str | None = None

    if not tool_results:
        return _ToolResultRows(rows, names, 0, False, False, None)

    for call_id, result in tool_results.items():
        tool_name = call_name_lookup.get(call_id, call_id)
        names.append(tool_name)
        result_summary = _extract_tool_result_summary(result)
        total_chars += result_summary.chars
        has_non_text = has_non_text or result_summary.non_text
        is_error = result.isError
        has_error = has_error or is_error
        tool_timing_info = tool_timings.get(call_id)
        last_timing_ms = tool_timing_info.get("timing_ms") if tool_timing_info else None
        rows.append(
            _history_row(
                role="tool",
                timeline_role="tool",
                chars=result_summary.chars,
                preview=result_summary.preview,
                details=_format_tool_detail("result→", [tool_name]),
                non_text=result_summary.non_text,
                has_tool_request=False,
                hide_summary=False,
                include_in_timeline=False,
                is_error=is_error,
                timing_ms=last_timing_ms,
            )
        )

    return _ToolResultRows(rows, names, total_chars, has_non_text, has_error, last_timing_ms)


def _provider_call_preview(arguments: object) -> str:
    try:
        return json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


def _provider_tool_rows(message: "PromptMessageExtended") -> _ProviderRows:
    rows: list[HistoryDisplayRow] = []
    total_chars = 0
    has_non_text = False
    has_error = False

    for event in remote_tool_activities(message):
        if event.kind == "call":
            arguments_text = _provider_call_preview(event.arguments)
            rows.append(
                _history_row(
                    role="tool",
                    timeline_role="tool",
                    chars=len(collapse_whitespace(arguments_text)),
                    preview=_preview_text(arguments_text),
                    details=Text(event.tool_name, style=TOOL_DETAIL_STYLE),
                    non_text=False,
                    has_tool_request=False,
                    hide_summary=False,
                    include_in_timeline=False,
                    is_error=False,
                    timing_ms=None,
                    label=event.type_label,
                    arrow="◀",
                )
            )
            continue

        if event.result is None:
            continue
        result_summary = _extract_tool_result_summary(event.result)
        total_chars += result_summary.chars
        has_non_text = has_non_text or result_summary.non_text
        has_error = has_error or event.is_error
        rows.append(
            _history_row(
                role="tool",
                timeline_role="tool",
                chars=result_summary.chars,
                preview=result_summary.preview,
                details=Text(event.tool_name, style=TOOL_DETAIL_STYLE),
                non_text=result_summary.non_text,
                has_tool_request=False,
                hide_summary=False,
                include_in_timeline=False,
                is_error=event.is_error,
                timing_ms=None,
                label=event.type_label,
                arrow="▶",
            )
        )

    return _ProviderRows(rows, total_chars, has_non_text, has_error)


def _combine_detail_sections(sections: list[Text]) -> Text | None:
    if not sections:
        return None
    if len(sections) == 1:
        return sections[0]

    details = Text()
    for index, section in enumerate(sections):
        if index > 0:
            details.append(" ")
        details.append_text(section)
    return details


def _compaction_preview(normalized_text: str, *, limit: int = 80) -> str:
    """Preview the checkpoint summary content, skipping the boilerplate notice."""
    from fast_agent.history.compaction import SUMMARY_NOTICE

    text = normalized_text
    marker = "[COMPACTED HISTORY]"
    if text.startswith(marker):
        text = text[len(marker) :].lstrip()
    notice = collapse_whitespace(SUMMARY_NOTICE)
    if text.startswith(notice):
        text = text[len(notice) :].lstrip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def build_history_rows(history: Sequence["PromptMessageExtended"]) -> list[HistoryDisplayRow]:
    rows: list[HistoryDisplayRow] = []
    call_name_lookup: dict[str, str] = {}

    for message in history:
        role = _message_role(message)
        text_summary = _message_text_summary(message)
        timing_ms = _extract_timing_ms(message)

        if message.channels and FAST_AGENT_COMPACTION_CHANNEL in message.channels:
            rows.append(
                _history_row(
                    role=role,
                    timeline_role=role,
                    chars=text_summary.chars,
                    preview=_compaction_preview(text_summary.normalized),
                    details=None,
                    non_text=False,
                    has_tool_request=False,
                    hide_summary=False,
                    include_in_timeline=True,
                    is_error=False,
                    timing_ms=timing_ms,
                    label="compacted",
                    arrow="≡",
                )
            )
            continue

        detail_sections: list[Text] = []
        row_non_text = text_summary.non_text
        hide_in_summary = False
        timeline_role = role

        tool_call_names = _tool_call_names(message.tool_calls, call_name_lookup)
        has_tool_request = bool(message.tool_calls)
        if tool_call_names:
            detail_sections.append(_format_tool_detail("tool→", tool_call_names))
            row_non_text = row_non_text and text_summary.chars == 0

        preview = text_summary.preview
        if not text_summary.normalized and message.tool_calls:
            preview = "(issuing tool request)"

        tool_result_rows = _tool_result_rows(
            message.tool_results,
            call_name_lookup,
            _extract_tool_timings(message),
        )
        if message.tool_results:
            timing_ms = tool_result_rows.timing_ms
            if role == "user":
                timeline_role = "tool"
                hide_in_summary = True
            if tool_result_rows.names:
                detail_sections.append(_format_tool_detail("result→", tool_result_rows.names))

        provider_rows = _provider_tool_rows(message)
        tool_result_total_chars = tool_result_rows.total_chars + provider_rows.total_chars
        row_chars = (
            tool_result_total_chars
            if timeline_role == "tool" and tool_result_total_chars > 0
            else text_summary.chars
        )

        rows.extend(provider_rows.rows)
        rows.append(
            _history_row(
                role=role,
                timeline_role=timeline_role,
                chars=row_chars,
                preview=preview,
                details=_combine_detail_sections(detail_sections),
                non_text=(
                    row_non_text or tool_result_rows.has_non_text or provider_rows.has_non_text
                ),
                has_tool_request=has_tool_request,
                hide_summary=hide_in_summary,
                include_in_timeline=True,
                is_error=tool_result_rows.has_error or provider_rows.has_error,
                timing_ms=timing_ms,
            )
        )
        rows.extend(tool_result_rows.rows)

    return rows
