"""Tool timing metadata types shared across agents and UI."""

from typing import TypeAlias, TypedDict


class ToolTimingInfo(TypedDict):
    timing_ms: float
    transport_channel: str | None


ToolTimings: TypeAlias = dict[str, ToolTimingInfo]
