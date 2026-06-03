"""Braille glyph palettes used by toolbar gauge renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fast_agent.ui.binary_indicator import render_glyph_indicator

MAX_GAUGE_LEVEL: Final[int] = 4


@dataclass(frozen=True, slots=True)
class GaugeGlyphPalette:
    """Glyph palette for a single-cell toolbar gauge."""

    full_block: str
    level_chars: tuple[str, str, str, str]

    def char_for_level(self, level: int) -> str:
        """Return the glyph for a non-zero gauge level."""
        normalized_level = min(max(level, 1), len(self.level_chars))
        return self.level_chars[normalized_level - 1]


@dataclass(frozen=True, slots=True)
class GaugeState:
    level: int
    color: str


def render_gauge_state(
    state: GaugeState,
    *,
    glyph_palette: GaugeGlyphPalette,
    inactive_color: str,
) -> str:
    glyph = (
        glyph_palette.full_block
        if state.level <= 0
        else glyph_palette.char_for_level(state.level)
    )
    color = inactive_color if state.level <= 0 else state.color
    return render_glyph_indicator(glyph=glyph, color=color)


STANDALONE_GAUGE_GLYPHS: Final[GaugeGlyphPalette] = GaugeGlyphPalette(
    full_block="⣿",
    level_chars=("⣀", "⣤", "⣶", "⣿"),
)

PAIRED_REASONING_GAUGE_GLYPHS: Final[GaugeGlyphPalette] = GaugeGlyphPalette(
    full_block="⢸",
    level_chars=("⢀", "⢠", "⢰", "⢸"),
)

PAIRED_VERBOSITY_GAUGE_GLYPHS: Final[GaugeGlyphPalette] = GaugeGlyphPalette(
    full_block="⡇",
    level_chars=("⡀", "⡄", "⡆", "⡇"),
)
