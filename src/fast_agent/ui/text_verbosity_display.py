"""Text verbosity gauge rendering for the TUI toolbar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.ui.gauge_glyph_palette import (
    STANDALONE_GAUGE_GLYPHS,
    GaugeGlyphPalette,
    GaugeState,
    render_gauge_state,
)

if TYPE_CHECKING:
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec

INACTIVE_COLOR = "ansibrightblack"


VERBOSITY_GAUGE_STEPS = {
    "low": GaugeState(level=2, color="ansigreen"),
    "medium": GaugeState(level=3, color="ansiyellow"),
    "high": GaugeState(level=4, color="ansired"),
}


def _inactive_verbosity_state() -> GaugeState:
    return GaugeState(level=0, color=INACTIVE_COLOR)


def _effective_text_verbosity(
    setting: "TextVerbosityLevel | None",
    spec: "TextVerbositySpec",
) -> "TextVerbosityLevel | None":
    effective = setting or spec.default
    if effective not in spec.allowed:
        return None
    return effective


def _text_verbosity_gauge_state(
    setting: "TextVerbosityLevel | None",
    spec: "TextVerbositySpec",
) -> GaugeState:
    effective = _effective_text_verbosity(setting, spec)
    if effective is None:
        return _inactive_verbosity_state()

    return VERBOSITY_GAUGE_STEPS.get(effective, _inactive_verbosity_state())


def render_text_verbosity_gauge(
    setting: "TextVerbosityLevel | None",
    spec: "TextVerbositySpec | None",
    *,
    glyph_palette: GaugeGlyphPalette = STANDALONE_GAUGE_GLYPHS,
) -> str | None:
    if spec is None:
        return None

    return render_gauge_state(
        _text_verbosity_gauge_state(setting, spec),
        glyph_palette=glyph_palette,
        inactive_color=INACTIVE_COLOR,
    )
