"""Model chip rendering for the TUI toolbar."""

from __future__ import annotations

MODEL_CHIP_COLOR = "ansigreen"


def render_model_chip(
    *,
    model_label: str,
    web_search_indicator: str | None = None,
    web_fetch_indicator: str | None = None,
    service_tier_indicator: str | None = None,
    attachment_indicator: str | None = None,
) -> str:
    indicators = "".join(
        indicator
        for indicator in (
            service_tier_indicator,
            web_search_indicator,
            web_fetch_indicator,
            attachment_indicator,
        )
        if indicator is not None
    )
    return f"<style bg='{MODEL_CHIP_COLOR}'>{model_label}</style>{indicators}"
