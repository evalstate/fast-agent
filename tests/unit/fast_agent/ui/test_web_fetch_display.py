from fast_agent.ui.binary_indicator import (
    TOOLBAR_BINARY_DISABLED_COLOR,
    TOOLBAR_BINARY_ENABLED_COLOR,
)
from fast_agent.ui.model_binary_toggles import WEB_FETCH_TOGGLE
from fast_agent.ui.prompt.status_bar.web_fetch import render_web_fetch_indicator


def test_render_web_fetch_indicator_hidden_when_unsupported() -> None:
    assert render_web_fetch_indicator(supported=False, enabled=False) is None


def test_render_web_fetch_indicator_dim_when_disabled() -> None:
    indicator = render_web_fetch_indicator(supported=True, enabled=False)

    assert indicator == f"<style bg='{TOOLBAR_BINARY_DISABLED_COLOR}'>{WEB_FETCH_TOGGLE.glyph}</style>"


def test_render_web_fetch_indicator_green_when_enabled() -> None:
    indicator = render_web_fetch_indicator(supported=True, enabled=True)

    assert indicator == f"<style bg='{TOOLBAR_BINARY_ENABLED_COLOR}'>{WEB_FETCH_TOGGLE.glyph}</style>"
