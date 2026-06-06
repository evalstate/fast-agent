"""Typed models for history display extraction and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.text import Text


@dataclass(frozen=True, slots=True)
class ToolResultSummary:
    preview: str
    chars: int
    non_text: bool


@dataclass(frozen=True, slots=True)
class HistoryChromeBar:
    bar: Text
    detail: Text


@dataclass(frozen=True, slots=True)
class HistoryDisplayRow:
    role: str
    timeline_role: str
    chars: int
    preview: str
    details: Text | None
    non_text: bool
    has_tool_request: bool
    hide_summary: bool
    include_in_timeline: bool
    is_error: bool
    timing_ms: float | str | None
    label: str | None = None
    arrow: str | None = None


@dataclass(frozen=True, slots=True)
class HistoryTimelineEntry:
    role: str
    chars: int
    non_text: bool
    is_error: bool
