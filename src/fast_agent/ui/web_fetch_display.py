"""Web-fetch indicator rendering for the TUI toolbar."""

from __future__ import annotations

WEB_FETCH_GLYPH = " ⇣"
WEB_FETCH_ENABLED_COLOR = "ansigreen"
WEB_FETCH_DISABLED_COLOR = "ansibrightblack"


def render_web_fetch_indicator(*, supported: bool, enabled: bool) -> str | None:
    if not supported:
        return None

    color = WEB_FETCH_ENABLED_COLOR if enabled else WEB_FETCH_DISABLED_COLOR
    return f"<style bg='{color}'>{WEB_FETCH_GLYPH}</style>"
