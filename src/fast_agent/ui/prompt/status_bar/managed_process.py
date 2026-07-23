"""Prompt-toolbar indicator for active managed shell processes."""

from __future__ import annotations

from fast_agent.constants import MAX_MANAGED_SHELL_PROCESSES
from fast_agent.ui.binary_indicator import render_glyph_indicator

MANAGED_PROCESS_GLYPH = "↻"
MANAGED_PROCESS_ACTIVE_COLOR = "ansiblack"
MANAGED_PROCESS_CAPACITY_WARNING_COLOR = "ansiblack"
MANAGED_PROCESS_IDLE_COLOR = "ansibrightblack"
MANAGED_PROCESS_ACTIVE_FOREGROUND = "ansiyellow"
MANAGED_PROCESS_CAPACITY_WARNING_FOREGROUND = "ansired"
MANAGED_PROCESS_CAPACITY_WARNING_THRESHOLD = MAX_MANAGED_SHELL_PROCESSES * 0.75


def render_managed_process_indicator(active_count: int) -> str:
    """Render managed-process activity without exposing a numeric count."""
    if active_count <= 0:
        return render_glyph_indicator(
            glyph=f"{MANAGED_PROCESS_GLYPH} ",
            color=MANAGED_PROCESS_IDLE_COLOR,
        )
    color = (
        MANAGED_PROCESS_CAPACITY_WARNING_COLOR
        if active_count > MANAGED_PROCESS_CAPACITY_WARNING_THRESHOLD
        else MANAGED_PROCESS_ACTIVE_COLOR
    )
    foreground = (
        MANAGED_PROCESS_CAPACITY_WARNING_FOREGROUND
        if active_count > MANAGED_PROCESS_CAPACITY_WARNING_THRESHOLD
        else MANAGED_PROCESS_ACTIVE_FOREGROUND
    )
    return render_glyph_indicator(
        glyph=f"{MANAGED_PROCESS_GLYPH} ",
        color=color,
        foreground=foreground,
    )
