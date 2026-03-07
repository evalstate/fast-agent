from fast_agent.llm.text_verbosity import TextVerbositySpec
from fast_agent.ui.gauge_glyph_palette import PAIRED_VERBOSITY_GAUGE_GLYPHS
from fast_agent.ui.text_verbosity_display import render_text_verbosity_gauge


def test_text_verbosity_gauge_is_hidden_without_spec() -> None:
    assert render_text_verbosity_gauge(setting="medium", spec=None) is None


def test_text_verbosity_gauge_uses_default_when_setting_is_unset() -> None:
    gauge = render_text_verbosity_gauge(
        setting=None,
        spec=TextVerbositySpec(default="medium"),
    )

    assert gauge == "<style bg='ansiyellow'>⣶</style>"


def test_text_verbosity_gauge_uses_three_step_three_high_braille_progression() -> None:
    spec = TextVerbositySpec(default="medium")

    low_gauge = render_text_verbosity_gauge(setting="low", spec=spec)
    medium_gauge = render_text_verbosity_gauge(setting="medium", spec=spec)
    high_gauge = render_text_verbosity_gauge(setting="high", spec=spec)

    assert low_gauge == "<style bg='ansigreen'>⣤</style>"
    assert medium_gauge == "<style bg='ansiyellow'>⣶</style>"
    assert high_gauge == "<style bg='ansired'>⣿</style>"


def test_text_verbosity_gauge_can_render_paired_palette() -> None:
    gauge = render_text_verbosity_gauge(
        setting="medium",
        spec=TextVerbositySpec(default="medium"),
        glyph_palette=PAIRED_VERBOSITY_GAUGE_GLYPHS,
    )

    assert gauge == "<style bg='ansiyellow'>⡆</style>"
