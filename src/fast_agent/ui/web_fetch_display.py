"""Web-fetch indicator rendering for the TUI toolbar."""

from __future__ import annotations

from fast_agent.ui.binary_indicator import (
    TOOLBAR_BINARY_DISABLED_COLOR,
    TOOLBAR_BINARY_ENABLED_COLOR,
)
from fast_agent.ui.model_binary_toggles import WEB_FETCH_TOGGLE, render_model_binary_indicator

WEB_FETCH_GLYPH = WEB_FETCH_TOGGLE.glyph
WEB_FETCH_ENABLED_COLOR = TOOLBAR_BINARY_ENABLED_COLOR
WEB_FETCH_DISABLED_COLOR = TOOLBAR_BINARY_DISABLED_COLOR


def render_web_fetch_indicator(*, supported: bool, enabled: bool) -> str | None:
    return render_model_binary_indicator(
        WEB_FETCH_TOGGLE,
        supported=supported,
        enabled=enabled,
    )
