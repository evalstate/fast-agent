"""History summary helpers for command renderers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_TOOL_TIMING, FAST_AGENT_USAGE
from fast_agent.history.tool_activities import message_tool_call_count, message_tool_error_count
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


@dataclass(slots=True)
class HistoryToolTiming:
    timing_ms: float | None
    transport_channel: str | None


@dataclass(slots=True)
class HistoryTurnSummary:
    turn_index: int
    user_snippet: str
    assistant_snippet: str
    tool_calls: int
    tool_errors: int
    llm_time_ms: float | None
    tool_time_ms: float | None
    turn_time_ms: float | None
    ttft_ms: float | None
    response_ms: float | None
    output_tokens: int | None
    tps: float | None


@dataclass(slots=True)
class HistoryTurnReport:
    turn_count: int
    total_tool_calls: int
    total_tool_errors: int
    total_llm_time_ms: float
    total_tool_time_ms: float
    total_turn_time_ms: float
    average_turn_time_ms: float | None
    average_tool_time_ms: float | None
    average_ttft_ms: float | None
    average_response_ms: float | None
    average_tps: float | None
    turns: list[HistoryTurnSummary]


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


def _normalize_text(value: str | None) -> str:
    return "" if not value else " ".join(value.split())


def _preview_text(value: str | None, *, limit: int = 60) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return "(no text content)"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_channel_payload(
    message: "PromptMessageExtended",
    channel_name: str,
) -> dict[str, Any] | None:
    channels = getattr(message, "channels", None)
    if not isinstance(channels, Mapping):
        return None

    channel_blocks = channels.get(channel_name)
    if not channel_blocks:
        return None

    channel_text = get_text(channel_blocks[0])
    if not channel_text:
        return None

    try:
        payload = json.loads(channel_text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def extract_message_timing_payload(
    message: "PromptMessageExtended",
) -> dict[str, Any] | None:
    return _extract_channel_payload(message, FAST_AGENT_TIMING)


def extract_message_duration_ms(message: "PromptMessageExtended") -> float | None:
    payload = extract_message_timing_payload(message)
    if payload is None:
        return None
    return _coerce_float(payload.get("duration_ms"))


def extract_message_tool_timings(
    message: "PromptMessageExtended",
) -> dict[str, HistoryToolTiming]:
    payload = _extract_channel_payload(message, FAST_AGENT_TOOL_TIMING)
    if payload is None:
        return {}

    timings: dict[str, HistoryToolTiming] = {}
    for tool_id, value in payload.items():
        if isinstance(value, Mapping):
            timings[tool_id] = HistoryToolTiming(
                timing_ms=_coerce_float(value.get("timing_ms")),
                transport_channel=(
                    value.get("transport_channel")
                    if isinstance(value.get("transport_channel"), str)
                    else None
                ),
            )
            continue
        timings[tool_id] = HistoryToolTiming(
            timing_ms=_coerce_float(value),
            transport_channel=None,
        )
    return timings


def extract_message_usage_payload(
    message: "PromptMessageExtended",
) -> dict[str, Any] | None:
    return _extract_channel_payload(message, FAST_AGENT_USAGE)


def extract_message_output_tokens(message: "PromptMessageExtended") -> int | None:
    payload = extract_message_usage_payload(message)
    if payload is None:
        return None
    turn_payload = payload.get("turn")
    if not isinstance(turn_payload, Mapping):
        return None
    return _coerce_int(turn_payload.get("output_tokens"))


def _extract_message_metric_ms(
    message: "PromptMessageExtended",
    *,
    keys: Sequence[str],
) -> float | None:
    timing_payload = extract_message_timing_payload(message)
    if timing_payload is not None:
        for key in keys:
            metric_value = _coerce_float(timing_payload.get(key))
            if metric_value is not None:
                return metric_value

    usage_payload = extract_message_usage_payload(message)
    if usage_payload is None:
        return None

    for section_name in ("turn", "raw_usage"):
        section = usage_payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key in keys:
            metric_value = _coerce_float(section.get(key))
            if metric_value is not None:
                return metric_value
    return None


def extract_message_ttft_ms(message: "PromptMessageExtended") -> float | None:
    return _extract_message_metric_ms(
        message,
        keys=(
            "ttft_ms",
            "first_activity_ms",
            "time_to_first_token_ms",
            "first_token_ms",
            "first_token_latency_ms",
        ),
    )


def extract_message_response_ms(message: "PromptMessageExtended") -> float | None:
    return _extract_message_metric_ms(
        message,
        keys=(
            "time_to_response_ms",
            "first_response_ms",
            "response_ms",
            "ttft_ms",
        ),
    )


def group_history_turns(
    messages: list["PromptMessageExtended"],
) -> list[tuple[int, list["PromptMessageExtended"]]]:
    turns: list[tuple[int, list[PromptMessageExtended]]] = []
    current: list[PromptMessageExtended] = []
    current_start = 0
    saw_assistant = False

    for idx, message in enumerate(messages):
        is_new_user = message.role == "user" and not message.tool_results
        if is_new_user:
            if not current:
                current = [message]
                current_start = idx
                saw_assistant = False
                continue
            if not saw_assistant:
                current.append(message)
                continue
            turns.append((current_start, current))
            current = [message]
            current_start = idx
            saw_assistant = False
            continue

        if current:
            current.append(message)
            if message.role == "assistant":
                saw_assistant = True

    if current:
        turns.append((current_start, current))
    return turns


def collect_user_turns(
    messages: list["PromptMessageExtended"],
) -> list[tuple[int, "PromptMessageExtended"]]:
    turns = group_history_turns(messages)
    user_turns: list[tuple[int, PromptMessageExtended]] = []
    for offset, turn in turns:
        if not turn:
            continue
        first = turn[0]
        if first.role != "user" or first.tool_results:
            continue
        user_turns.append((offset, first))
    return user_turns


def _summarize_turn_messages(
    turn: Sequence["PromptMessageExtended"],
) -> tuple[str, str]:
    user_parts: list[str] = []
    assistant_parts: list[str] = []

    for message in turn:
        text = _extract_message_text(message)
        if message.role == "user" and not message.tool_results:
            normalized = _normalize_text(text)
            if normalized:
                user_parts.append(normalized)
        elif message.role == "assistant":
            normalized = _normalize_text(text)
            if normalized:
                assistant_parts.append(normalized)

    user_text = " / ".join(user_parts)
    assistant_text = assistant_parts[-1] if assistant_parts else ""
    return _preview_text(user_text), _preview_text(assistant_text)


def _calculate_tokens_per_second(
    *,
    output_tokens: int | None,
    llm_time_ms: float | None,
    response_ms: float | None,
) -> float | None:
    if output_tokens is None or output_tokens <= 0 or llm_time_ms is None or llm_time_ms <= 0:
        return None

    effective_ms = llm_time_ms
    if response_ms is not None and 0 < response_ms < llm_time_ms:
        effective_ms = llm_time_ms - response_ms
    if effective_ms <= 0:
        return None
    return output_tokens / (effective_ms / 1000.0)


def build_history_turn_report(messages: list["PromptMessageExtended"]) -> HistoryTurnReport:
    turn_rows: list[HistoryTurnSummary] = []

    total_tool_calls = 0
    total_tool_errors = 0
    total_llm_time_ms = 0.0
    total_tool_time_ms = 0.0
    total_turn_time_ms = 0.0

    known_turn_times: list[float] = []
    known_tool_times: list[float] = []
    known_ttfts: list[float] = []
    known_responses: list[float] = []
    known_tps: list[float] = []

    turns = [
        turn
        for _, turn in group_history_turns(messages)
        if turn and turn[0].role == "user" and not turn[0].tool_results
    ]

    for turn_index, turn in enumerate(turns, start=1):
        user_snippet, assistant_snippet = _summarize_turn_messages(turn)

        tool_calls = 0
        tool_errors = 0
        llm_time_ms = 0.0
        saw_llm_time = False
        tool_time_ms = 0.0
        saw_tool_time = False
        output_tokens_total = 0
        saw_output_tokens = False
        ttft_ms: float | None = None
        response_ms: float | None = None
        first_start: float | None = None
        last_end: float | None = None

        for message in turn:
            tool_calls += message_tool_call_count(message)

            if message.tool_results:
                for tool_timing in extract_message_tool_timings(message).values():
                    if tool_timing.timing_ms is None:
                        continue
                    tool_time_ms += tool_timing.timing_ms
                    saw_tool_time = True
            tool_errors += message_tool_error_count(message)

            if message.role != "assistant":
                continue

            duration_ms = extract_message_duration_ms(message)
            if duration_ms is not None:
                llm_time_ms += duration_ms
                saw_llm_time = True

            output_tokens = extract_message_output_tokens(message)
            if output_tokens is not None:
                output_tokens_total += output_tokens
                saw_output_tokens = True

            if ttft_ms is None:
                ttft_ms = extract_message_ttft_ms(message)
            if response_ms is None:
                response_ms = extract_message_response_ms(message)

            timing_payload = extract_message_timing_payload(message)
            if timing_payload is None:
                continue

            start_time = _coerce_float(timing_payload.get("start_time"))
            end_time = _coerce_float(timing_payload.get("end_time"))
            if start_time is not None and (first_start is None or start_time < first_start):
                first_start = start_time
            if end_time is not None and (last_end is None or end_time > last_end):
                last_end = end_time

        tool_time_value = tool_time_ms if saw_tool_time else None
        llm_time_value = llm_time_ms if saw_llm_time else None
        output_tokens_value = output_tokens_total if saw_output_tokens else None

        turn_time_ms: float | None = None
        if first_start is not None and last_end is not None and last_end >= first_start:
            turn_time_ms = round((last_end - first_start) * 1000.0, 2)
        elif llm_time_value is not None or tool_time_value is not None:
            turn_time_ms = round(float((llm_time_value or 0.0) + (tool_time_value or 0.0)), 2)

        tps = _calculate_tokens_per_second(
            output_tokens=output_tokens_value,
            llm_time_ms=llm_time_value,
            response_ms=response_ms,
        )

        turn_rows.append(
            HistoryTurnSummary(
                turn_index=turn_index,
                user_snippet=user_snippet,
                assistant_snippet=assistant_snippet,
                tool_calls=tool_calls,
                tool_errors=tool_errors,
                llm_time_ms=llm_time_value,
                tool_time_ms=tool_time_value,
                turn_time_ms=turn_time_ms,
                ttft_ms=ttft_ms,
                response_ms=response_ms,
                output_tokens=output_tokens_value,
                tps=tps,
            )
        )

        total_tool_calls += tool_calls
        total_tool_errors += tool_errors
        total_llm_time_ms += llm_time_ms
        total_tool_time_ms += tool_time_ms
        if turn_time_ms is not None:
            total_turn_time_ms += turn_time_ms
            known_turn_times.append(turn_time_ms)
        if tool_time_value is not None:
            known_tool_times.append(tool_time_value)
        if ttft_ms is not None:
            known_ttfts.append(ttft_ms)
        if response_ms is not None:
            known_responses.append(response_ms)
        if tps is not None:
            known_tps.append(tps)

    average_turn_time_ms = (
        sum(known_turn_times) / len(known_turn_times) if known_turn_times else None
    )
    average_tool_time_ms = (
        sum(known_tool_times) / len(known_tool_times) if known_tool_times else None
    )
    average_ttft_ms = sum(known_ttfts) / len(known_ttfts) if known_ttfts else None
    average_response_ms = (
        sum(known_responses) / len(known_responses) if known_responses else None
    )
    average_tps = sum(known_tps) / len(known_tps) if known_tps else None

    return HistoryTurnReport(
        turn_count=len(turn_rows),
        total_tool_calls=total_tool_calls,
        total_tool_errors=total_tool_errors,
        total_llm_time_ms=total_llm_time_ms,
        total_tool_time_ms=total_tool_time_ms,
        total_turn_time_ms=total_turn_time_ms,
        average_turn_time_ms=average_turn_time_ms,
        average_tool_time_ms=average_tool_time_ms,
        average_ttft_ms=average_ttft_ms,
        average_response_ms=average_response_ms,
        average_tps=average_tps,
        turns=turn_rows,
    )


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
