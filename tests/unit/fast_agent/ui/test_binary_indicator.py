from fast_agent.ui.binary_indicator import (
    TOOLBAR_BINARY_DISABLED_COLOR,
    TOOLBAR_BINARY_ENABLED_COLOR,
    render_binary_indicator,
    render_glyph_indicator,
    render_supported_glyph_indicator,
    render_toolbar_binary_indicator,
)


def test_render_glyph_indicator_uses_color() -> None:
    assert render_glyph_indicator(glyph="x", color="green") == "<style bg='green'>x</style>"


def test_render_glyph_indicator_escapes_prompt_toolkit_html() -> None:
    assert render_glyph_indicator(glyph="<model&x>", color="bad'color") == (
        "<style bg='bad&#x27;color'>&lt;model&amp;x&gt;</style>"
    )


def test_render_supported_glyph_indicator_hides_when_unsupported() -> None:
    assert render_supported_glyph_indicator(supported=False, glyph="x", color="green") is None


def test_render_supported_glyph_indicator_uses_color() -> None:
    assert (
        render_supported_glyph_indicator(supported=True, glyph="x", color="green")
        == "<style bg='green'>x</style>"
    )


def test_render_binary_indicator_hides_when_unsupported() -> None:
    assert (
        render_binary_indicator(
            supported=False,
            enabled=True,
            glyph="x",
            enabled_color="green",
            disabled_color="dim",
        )
        is None
    )


def test_render_binary_indicator_uses_enabled_or_disabled_color() -> None:
    assert (
        render_binary_indicator(
            supported=True,
            enabled=True,
            glyph="x",
            enabled_color="green",
            disabled_color="dim",
        )
        == "<style bg='green'>x</style>"
    )
    assert (
        render_binary_indicator(
            supported=True,
            enabled=False,
            glyph="x",
            enabled_color="green",
            disabled_color="dim",
        )
        == "<style bg='dim'>x</style>"
    )


def test_render_toolbar_binary_indicator_uses_standard_toolbar_colors() -> None:
    assert (
        render_toolbar_binary_indicator(supported=True, enabled=True, glyph="x")
        == f"<style bg='{TOOLBAR_BINARY_ENABLED_COLOR}'>x</style>"
    )
    assert (
        render_toolbar_binary_indicator(supported=True, enabled=False, glyph="x")
        == f"<style bg='{TOOLBAR_BINARY_DISABLED_COLOR}'>x</style>"
    )
