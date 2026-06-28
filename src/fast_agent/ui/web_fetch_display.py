"""Web-fetch indicator rendering for the TUI toolbar."""

from __future__ import annotations

from fast_agent.ui.model_binary_toggles import WEB_FETCH_TOGGLE, render_model_binary_indicator


def render_web_fetch_indicator(*, supported: bool, enabled: bool) -> str | None:
    return render_model_binary_indicator(
        WEB_FETCH_TOGGLE,
        supported=supported,
        enabled=enabled,
    )
