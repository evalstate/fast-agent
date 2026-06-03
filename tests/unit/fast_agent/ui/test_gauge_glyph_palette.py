from fast_agent.ui.gauge_glyph_palette import GaugeGlyphPalette, GaugeState, render_gauge_state


def test_gauge_palette_clamps_to_its_configured_levels() -> None:
    palette = GaugeGlyphPalette(
        full_block="full",
        level_chars=("one", "two", "three", "four"),
    )

    assert palette.char_for_level(-1) == "one"
    assert palette.char_for_level(0) == "one"
    assert palette.char_for_level(3) == "three"
    assert palette.char_for_level(99) == "four"


def test_render_gauge_state_uses_inactive_full_block_for_zero_or_negative_levels() -> None:
    palette = GaugeGlyphPalette(
        full_block="full",
        level_chars=("one", "two", "three", "four"),
    )

    assert (
        render_gauge_state(
            GaugeState(level=0, color="active"),
            glyph_palette=palette,
            inactive_color="inactive",
        )
        == "<style bg='inactive'>full</style>"
    )
    assert (
        render_gauge_state(
            GaugeState(level=-1, color="active"),
            glyph_palette=palette,
            inactive_color="inactive",
        )
        == "<style bg='inactive'>full</style>"
    )


def test_render_gauge_state_uses_clamped_level_glyph_and_active_color() -> None:
    palette = GaugeGlyphPalette(
        full_block="full",
        level_chars=("one", "two", "three", "four"),
    )

    assert (
        render_gauge_state(
            GaugeState(level=99, color="active"),
            glyph_palette=palette,
            inactive_color="inactive",
        )
        == "<style bg='active'>four</style>"
    )
