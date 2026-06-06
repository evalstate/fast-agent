"""History summary helpers for command renderers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fast_agent.commands.summary_utils import json_object
from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_TOOL_TIMING, FAST_AGENT_USAGE
from fast_agent.history.tool_activities import message_tool_call_count, message_tool_error_count
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary
from fast_agent.utils.numeric import nonnegative_int_or_none, nonnegative_number_or_none
from fast_agent.utils.text import collapse_whitespace

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


type JsonObject = dict[str, object]


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


@dataclass(frozen=True, slots=True)
class HistoryTurnPreview:
    user: str
    assistant: str


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    start_index: int
    messages: list["PromptMessageExtended"]

    @property
    def first_message(self) -> "PromptMessageExtended | None":
        return self.messages[0] if self.messages else None

    @property
    def first_user_message(self) -> "PromptMessageExtended | None":
        first = self.first_message
        if first is None or first.role != "user" or first.tool_results:
            return None
        return first

    @property
    def is_user_turn(self) -> bool:
        return self.first_user_message is not None


@dataclass(slots=True)
class _TurnMetricAccumulator:
    tool_calls: int = 0
    tool_errors: int = 0
    llm_time_ms: float = 0.0
    saw_llm_time: bool = False
    tool_time_ms: float = 0.0
    saw_tool_time: bool = False
    output_tokens: int = 0
    saw_output_tokens: bool = False
    ttft_ms: float | None = None
    response_ms: float | None = None
    first_start: float | None = None
    last_end: float | None = None


@dataclass(slots=True)
class _ReportMetricAccumulator:
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    total_llm_time_ms: float = 0.0
    total_tool_time_ms: float = 0.0
    total_turn_time_ms: float = 0.0
    known_turn_times: list[float] = field(default_factory=list)
    known_tool_times: list[float] = field(default_factory=list)
    known_ttfts: list[float] = field(default_factory=list)
    known_responses: list[float] = field(default_factory=list)
    known_tps: list[float] = field(default_factory=list)


def _extract_message_text(message: "PromptMessageExtended") -> str:
    return message.all_text()


def _preview_text(value: str | None, *, limit: int = 60) -> str:
    normalized = collapse_whitespace(value)
    if not normalized:
        return "(no text content)"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _coerce_float(value: object) -> float | None:
    numeric_value = nonnegative_number_or_none(value)
    return float(numeric_value) if numeric_value is not None else None


def _coerce_int(value: object) -> int | None:
    return nonnegative_int_or_none(value)


def _json_object_or_none(value: object) -> JsonObject | None:
    if not isinstance(value, Mapping):
        return None
    return json_object(value)


def _extract_channel_payload(
    message: "PromptMessageExtended",
    channel_name: str,
) -> JsonObject | None:
    channels = message.channels
    if not isinstance(channels, Mapping):
        return None

    channel_blocks = channels.get(channel_name)
    if not channel_blocks:
        return None

    for channel_block in channel_blocks:
        channel_text = get_text(channel_block)
        if not channel_text:
            continue

        try:
            payload = json.loads(channel_text)
        except (TypeError, ValueError):
            continue
        json_payload = _json_object_or_none(payload)
        if json_payload is not None:
            return json_payload
    return None


def extract_message_timing_payload(
    message: "PromptMessageExtended",
) -> JsonObject | None:
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
        timing_payload = _json_object_or_none(value)
        if timing_payload is not None:
            transport_channel = timing_payload.get("transport_channel")
            timings[tool_id] = HistoryToolTiming(
                timing_ms=_coerce_float(timing_payload.get("timing_ms")),
                transport_channel=(
                    transport_channel if isinstance(transport_channel, str) else None
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
) -> JsonObject | None:
    return _extract_channel_payload(message, FAST_AGENT_USAGE)


def extract_message_output_tokens(message: "PromptMessageExtended") -> int | None:
    payload = extract_message_usage_payload(message)
    if payload is None:
        return None
    turn_payload = _json_object_or_none(payload.get("turn"))
    if turn_payload is None:
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
        section = _json_object_or_none(usage_payload.get(section_name))
        if section is None:
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


def _is_user_turn_start(message: "PromptMessageExtended") -> bool:
    return message.role == "user" and not message.tool_results


def group_history_turns(
    messages: list["PromptMessageExtended"],
) -> list[HistoryTurn]:
    turns: list[HistoryTurn] = []
    current: list[PromptMessageExtended] = []
    current_start = 0
    saw_assistant = False

    for idx, message in enumerate(messages):
        if _is_user_turn_start(message):
            if saw_assistant:
                turns.append(HistoryTurn(start_index=current_start, messages=current))
                current = []

            if not current:
                current_start = idx
                saw_assistant = False
            current.append(message)
            continue

        if not current:
            continue

        current.append(message)
        saw_assistant = saw_assistant or message.role == "assistant"

    if current:
        turns.append(HistoryTurn(start_index=current_start, messages=current))
    return turns


def collect_user_turns(
    messages: list["PromptMessageExtended"],
) -> list[HistoryTurn]:
    return [turn for turn in group_history_turns(messages) if turn.is_user_turn]


def _summarize_turn_messages(
    turn: Sequence["PromptMessageExtended"],
) -> HistoryTurnPreview:
    user_parts: list[str] = []
    assistant_parts: list[str] = []

    for message in turn:
        text = _extract_message_text(message)
        if message.role == "user" and not message.tool_results:
            normalized = collapse_whitespace(text)
            if normalized:
                user_parts.append(normalized)
        elif message.role == "assistant":
            normalized = collapse_whitespace(text)
            if normalized:
                assistant_parts.append(normalized)

    user_text = " / ".join(user_parts)
    assistant_text = assistant_parts[-1] if assistant_parts else ""
    return HistoryTurnPreview(
        user=_preview_text(user_text),
        assistant=_preview_text(assistant_text),
    )


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


def _record_message_tool_metrics(
    metrics: _TurnMetricAccumulator,
    message: "PromptMessageExtended",
) -> None:
    metrics.tool_calls += message_tool_call_count(message)
    metrics.tool_errors += message_tool_error_count(message)

    if not message.tool_results:
        return

    for tool_timing in extract_message_tool_timings(message).values():
        if tool_timing.timing_ms is None:
            continue
        metrics.tool_time_ms += tool_timing.timing_ms
        metrics.saw_tool_time = True


def _record_message_timing_bounds(
    metrics: _TurnMetricAccumulator,
    message: "PromptMessageExtended",
) -> None:
    timing_payload = extract_message_timing_payload(message)
    if timing_payload is None:
        return

    start_time = _coerce_float(timing_payload.get("start_time"))
    end_time = _coerce_float(timing_payload.get("end_time"))
    if start_time is not None and (
        metrics.first_start is None or start_time < metrics.first_start
    ):
        metrics.first_start = start_time
    if end_time is not None and (metrics.last_end is None or end_time > metrics.last_end):
        metrics.last_end = end_time


def _record_message_assistant_metrics(
    metrics: _TurnMetricAccumulator,
    message: "PromptMessageExtended",
) -> None:
    if message.role != "assistant":
        return

    duration_ms = extract_message_duration_ms(message)
    if duration_ms is not None:
        metrics.llm_time_ms += duration_ms
        metrics.saw_llm_time = True

    output_tokens = extract_message_output_tokens(message)
    if output_tokens is not None:
        metrics.output_tokens += output_tokens
        metrics.saw_output_tokens = True

    if metrics.ttft_ms is None:
        metrics.ttft_ms = extract_message_ttft_ms(message)
    if metrics.response_ms is None:
        metrics.response_ms = extract_message_response_ms(message)

    _record_message_timing_bounds(metrics, message)


def _summarize_turn_metrics(turn: HistoryTurn) -> _TurnMetricAccumulator:
    metrics = _TurnMetricAccumulator()
    for message in turn.messages:
        _record_message_tool_metrics(metrics, message)
        _record_message_assistant_metrics(metrics, message)
    return metrics


def _known_value(value: float, *, saw_value: bool) -> float | None:
    return value if saw_value else None


def _turn_elapsed_time_ms(
    metrics: _TurnMetricAccumulator,
    *,
    llm_time_ms: float | None,
    tool_time_ms: float | None,
) -> float | None:
    if (
        metrics.first_start is not None
        and metrics.last_end is not None
        and metrics.last_end >= metrics.first_start
    ):
        return round((metrics.last_end - metrics.first_start) * 1000.0, 2)
    if llm_time_ms is None and tool_time_ms is None:
        return None
    return round(float((llm_time_ms or 0.0) + (tool_time_ms or 0.0)), 2)


def _build_turn_summary(
    *,
    turn_index: int,
    turn: HistoryTurn,
) -> tuple[HistoryTurnSummary, _TurnMetricAccumulator]:
    preview = _summarize_turn_messages(turn.messages)
    metrics = _summarize_turn_metrics(turn)
    tool_time_value = _known_value(metrics.tool_time_ms, saw_value=metrics.saw_tool_time)
    llm_time_value = _known_value(metrics.llm_time_ms, saw_value=metrics.saw_llm_time)
    output_tokens_value = metrics.output_tokens if metrics.saw_output_tokens else None
    turn_time_ms = _turn_elapsed_time_ms(
        metrics,
        llm_time_ms=llm_time_value,
        tool_time_ms=tool_time_value,
    )
    tps = _calculate_tokens_per_second(
        output_tokens=output_tokens_value,
        llm_time_ms=llm_time_value,
        response_ms=metrics.response_ms,
    )

    return (
        HistoryTurnSummary(
            turn_index=turn_index,
            user_snippet=preview.user,
            assistant_snippet=preview.assistant,
            tool_calls=metrics.tool_calls,
            tool_errors=metrics.tool_errors,
            llm_time_ms=llm_time_value,
            tool_time_ms=tool_time_value,
            turn_time_ms=turn_time_ms,
            ttft_ms=metrics.ttft_ms,
            response_ms=metrics.response_ms,
            output_tokens=output_tokens_value,
            tps=tps,
        ),
        metrics,
    )


def _record_report_turn_metrics(
    report_metrics: _ReportMetricAccumulator,
    turn_summary: HistoryTurnSummary,
    turn_metrics: _TurnMetricAccumulator,
) -> None:
    report_metrics.total_tool_calls += turn_metrics.tool_calls
    report_metrics.total_tool_errors += turn_metrics.tool_errors
    report_metrics.total_llm_time_ms += turn_metrics.llm_time_ms
    report_metrics.total_tool_time_ms += turn_metrics.tool_time_ms
    if turn_summary.turn_time_ms is not None:
        report_metrics.total_turn_time_ms += turn_summary.turn_time_ms
        report_metrics.known_turn_times.append(turn_summary.turn_time_ms)
    if turn_summary.tool_time_ms is not None:
        report_metrics.known_tool_times.append(turn_summary.tool_time_ms)
    if turn_summary.ttft_ms is not None:
        report_metrics.known_ttfts.append(turn_summary.ttft_ms)
    if turn_summary.response_ms is not None:
        report_metrics.known_responses.append(turn_summary.response_ms)
    if turn_summary.tps is not None:
        report_metrics.known_tps.append(turn_summary.tps)


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_history_turn_report(messages: list["PromptMessageExtended"]) -> HistoryTurnReport:
    turn_rows: list[HistoryTurnSummary] = []
    report_metrics = _ReportMetricAccumulator()

    turns = collect_user_turns(messages)

    for turn_index, turn in enumerate(turns, start=1):
        turn_summary, turn_metrics = _build_turn_summary(
            turn_index=turn_index,
            turn=turn,
        )
        turn_rows.append(turn_summary)
        _record_report_turn_metrics(report_metrics, turn_summary, turn_metrics)

    return HistoryTurnReport(
        turn_count=len(turn_rows),
        total_tool_calls=report_metrics.total_tool_calls,
        total_tool_errors=report_metrics.total_tool_errors,
        total_llm_time_ms=report_metrics.total_llm_time_ms,
        total_tool_time_ms=report_metrics.total_tool_time_ms,
        total_turn_time_ms=report_metrics.total_turn_time_ms,
        average_turn_time_ms=_average(report_metrics.known_turn_times),
        average_tool_time_ms=_average(report_metrics.known_tool_times),
        average_ttft_ms=_average(report_metrics.known_ttfts),
        average_response_ms=_average(report_metrics.known_responses),
        average_tps=_average(report_metrics.known_tps),
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
        recent_messages.extend(
            HistoryMessageSnippet(
                role=str(message.role),
                snippet=_preview_text(_extract_message_text(message)),
            )
            for message in messages[-recent_count:]
        )

    return HistoryOverview(
        message_count=summary.message_count,
        user_message_count=summary.user_message_count,
        assistant_message_count=summary.assistant_message_count,
        tool_calls=summary.tool_calls,
        tool_successes=summary.tool_successes,
        tool_errors=summary.tool_errors,
        recent_messages=recent_messages,
    )
