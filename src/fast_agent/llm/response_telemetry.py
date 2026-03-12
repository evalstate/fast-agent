from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import TextContent

from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_USAGE

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.llm.stream_types import StreamChunk
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class TimingListenerSource(Protocol):
    def add_stream_listener(
        self, listener: "Callable[[StreamChunk], None]"
    ) -> "Callable[[], None]": ...

    def add_tool_stream_listener(
        self, listener: "Callable[[str, dict[str, Any] | None], None]"
    ) -> "Callable[[], None]": ...


class RequestTimingCapture:
    def __init__(self, *, start_time: float) -> None:
        self.start_time = start_time
        self.first_activity_time: float | None = None
        self.first_response_time: float | None = None

    def _mark_activity(self, observed_time: float) -> None:
        if self.first_activity_time is None:
            self.first_activity_time = observed_time

    def _mark_response(self, observed_time: float) -> None:
        if self.first_response_time is None:
            self.first_response_time = observed_time

    def observe_stream_chunk(self, chunk: StreamChunk) -> None:
        if not chunk.text.strip():
            return

        observed_time = time.perf_counter()
        self._mark_activity(observed_time)
        if not chunk.is_reasoning:
            self._mark_response(observed_time)

    def observe_tool_event(self, event_type: str, payload: dict[str, Any] | None) -> None:
        del payload
        if event_type != "start":
            return

        observed_time = time.perf_counter()
        self._mark_activity(observed_time)
        self._mark_response(observed_time)

    def _elapsed_ms(self, observed_time: float | None) -> float | None:
        if observed_time is None or observed_time < self.start_time:
            return None
        return round((observed_time - self.start_time) * 1000, 2)

    @property
    def ttft_ms(self) -> float | None:
        return self._elapsed_ms(self.first_activity_time)

    @property
    def time_to_response_ms(self) -> float | None:
        return self._elapsed_ms(self.first_response_time)


def start_request_timing_capture(
    source: TimingListenerSource,
) -> tuple[RequestTimingCapture, "Callable[[], None]"]:
    capture = RequestTimingCapture(start_time=time.perf_counter())
    remove_stream_listener = source.add_stream_listener(capture.observe_stream_chunk)
    remove_tool_stream_listener = source.add_tool_stream_listener(capture.observe_tool_event)

    def cleanup() -> None:
        remove_tool_stream_listener()
        remove_stream_listener()

    return capture, cleanup


def add_timing_channel(
    response: "PromptMessageExtended",
    start_time: float,
    end_time: float,
    *,
    ttft_ms: float | None = None,
    time_to_response_ms: float | None = None,
) -> None:
    """Add timing data to response channels if not already present."""
    duration_ms = round((end_time - start_time) * 1000, 2)
    channels = dict(response.channels or {})
    if FAST_AGENT_TIMING in channels:
        return

    timing_data = {
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
    }
    if ttft_ms is not None:
        timing_data["ttft_ms"] = ttft_ms
    if time_to_response_ms is not None:
        timing_data["time_to_response_ms"] = time_to_response_ms

    channels[FAST_AGENT_TIMING] = [TextContent(type="text", text=json.dumps(timing_data))]
    response.channels = channels


def append_usage_channel(
    response: "PromptMessageExtended",
    usage_accumulator: "UsageAccumulator | None",
) -> None:
    usage_payload = build_usage_payload(usage_accumulator)
    if not usage_payload:
        return

    channels = dict(response.channels or {})
    if FAST_AGENT_USAGE in channels:
        return

    channels[FAST_AGENT_USAGE] = [
        TextContent(type="text", text=json.dumps(usage_payload))
    ]
    response.channels = channels


def build_usage_payload(usage_accumulator: "UsageAccumulator | None") -> dict[str, Any] | None:
    if not usage_accumulator or not usage_accumulator.turns:
        return None

    turn_usage = usage_accumulator.turns[-1]
    return {
        "turn": turn_usage.model_dump(mode="json", exclude={"raw_usage"}),
        "raw_usage": serialize_raw_usage(turn_usage.raw_usage),
        "summary": usage_accumulator.get_summary(),
    }


def serialize_raw_usage(raw_usage: object) -> object:
    for attr in ("model_dump", "dict"):
        method = getattr(raw_usage, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                continue
    raw_dict = getattr(raw_usage, "__dict__", None)
    if isinstance(raw_dict, dict):
        try:
            return dict(raw_dict)
        except Exception:
            pass
    return str(raw_usage)
