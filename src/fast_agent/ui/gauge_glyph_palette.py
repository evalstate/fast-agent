"""Braille glyph palettes used by toolbar gauge renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MAX_GAUGE_LEVEL: Final[int] = 4


@dataclass(frozen=True, slots=True)
class GaugeGlyphPalette:
    """Glyph palette for a single-cell toolbar gauge."""

    full_block: str
    level_chars: tuple[str, str, str, str]

    def char_for_level(self, level: int) -> str:
        """Return the glyph for a non-zero gauge level."""
        normalized_level = min(max(level, 1), MAX_GAUGE_LEVEL)
        return self.level_chars[normalized_level - 1]


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
