"""Text verbosity gauge rendering for the TUI toolbar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.ui.gauge_glyph_palette import STANDALONE_GAUGE_GLYPHS, GaugeGlyphPalette

if TYPE_CHECKING:
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec

FULL_BLOCK = STANDALONE_GAUGE_GLYPHS.full_block
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
    *,
    glyph_palette: GaugeGlyphPalette = STANDALONE_GAUGE_GLYPHS,
) -> str | None:
    if spec is None:
        return None

    effective = setting or spec.default
    if effective is None:
        return f"<style bg='{INACTIVE_COLOR}'>{glyph_palette.full_block}</style>"

    level = VERBOSITY_LEVELS.get(effective, 0)
    if level <= 0:
        return f"<style bg='{INACTIVE_COLOR}'>{glyph_palette.full_block}</style>"

    char = glyph_palette.char_for_level(level)
    color = VERBOSITY_COLORS.get(effective, "ansiyellow")
    return f"<style bg='{color}'>{char}</style>"
