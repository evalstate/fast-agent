"""Web-search indicator rendering for the TUI toolbar."""

from __future__ import annotations

WEB_SEARCH_GLYPH = "⊕"
WEB_SEARCH_ENABLED_COLOR = "ansigreen"
WEB_SEARCH_DISABLED_COLOR = "ansibrightblack"


def render_web_search_indicator(*, supported: bool, enabled: bool) -> str | None:
    if not supported:
        return None

    color = WEB_SEARCH_ENABLED_COLOR if enabled else WEB_SEARCH_DISABLED_COLOR
    return f"<style bg='{color}'>{WEB_SEARCH_GLYPH}</style>"
