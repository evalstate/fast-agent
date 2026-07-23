"""Model chip rendering for the TUI toolbar."""

from __future__ import annotations

from fast_agent.ui.binary_indicator import render_glyph_indicator

MODEL_CHIP_COLOR = "ansigreen"


def _join_model_chip_indicators(
    *,
    service_tier_indicator: str | None,
    web_search_indicator: str | None,
    web_fetch_indicator: str | None,
) -> str:
    return "".join(
        indicator
        for indicator in (
            service_tier_indicator,
            web_search_indicator,
            web_fetch_indicator,
        )
        if indicator is not None
    )


def render_model_chip(
    *,
    model_label: str,
    web_search_indicator: str | None = None,
    web_fetch_indicator: str | None = None,
    service_tier_indicator: str | None = None,
) -> str:
    indicators = _join_model_chip_indicators(
        service_tier_indicator=service_tier_indicator,
        web_search_indicator=web_search_indicator,
        web_fetch_indicator=web_fetch_indicator,
    )
    return f"{render_glyph_indicator(glyph=model_label, color=MODEL_CHIP_COLOR)}{indicators}"
