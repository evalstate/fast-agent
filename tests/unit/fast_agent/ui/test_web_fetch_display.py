from fast_agent.ui.web_fetch_display import (
    WEB_FETCH_DISABLED_COLOR,
    WEB_FETCH_ENABLED_COLOR,
    WEB_FETCH_GLYPH,
    render_web_fetch_indicator,
)


def test_render_web_fetch_indicator_hidden_when_unsupported() -> None:
    assert render_web_fetch_indicator(supported=False, enabled=False) is None


def test_render_web_fetch_indicator_dim_when_disabled() -> None:
    indicator = render_web_fetch_indicator(supported=True, enabled=False)

    assert indicator == f"<style bg='{WEB_FETCH_DISABLED_COLOR}'>{WEB_FETCH_GLYPH}</style>"


def test_render_web_fetch_indicator_green_when_enabled() -> None:
    indicator = render_web_fetch_indicator(supported=True, enabled=True)

    assert indicator == f"<style bg='{WEB_FETCH_ENABLED_COLOR}'>{WEB_FETCH_GLYPH}</style>"
