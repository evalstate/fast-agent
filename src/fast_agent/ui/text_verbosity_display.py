"""Text verbosity gauge rendering for the TUI toolbar."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec

BRAILLE_FILL = {0: "⣿", 1: "⣀", 2: "⣤", 3: "⣶", 4: "⣿"}
FULL_BLOCK = "⣿"
INACTIVE_COLOR = "ansibrightblack"

VERBOSITY_LEVELS = {
    "low": 2,
    "medium": 3,
    "high": 4,
}

VERBOSITY_COLORS = {
    "low": "ansigreen",
    "medium": "ansiyellow",
    "high": "ansired",
}


def render_text_verbosity_gauge(
    setting: "TextVerbosityLevel | None",
    spec: "TextVerbositySpec | None",
) -> str | None:
    if spec is None:
        return None

    effective = setting or spec.default
    if effective is None:
        return f"<style bg='{INACTIVE_COLOR}'>{FULL_BLOCK}</style>"

    level = VERBOSITY_LEVELS.get(effective, 0)
    if level <= 0:
        return f"<style bg='{INACTIVE_COLOR}'>{FULL_BLOCK}</style>"

    char = BRAILLE_FILL.get(level, BRAILLE_FILL[4])
    color = VERBOSITY_COLORS.get(effective, "ansiyellow")
    return f"<style bg='{color}'>{char}</style>"
