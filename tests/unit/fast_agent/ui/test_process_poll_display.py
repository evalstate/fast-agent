from fast_agent.ui.process_poll_display import (
    _cell_glyph,
    _countdown_cell_count,
    _countdown_cycle_count,
    _remaining_slots_for_wait,
    _remaining_units_for_wait,
    _track_from_remaining_units,
    format_process_output_activity,
    format_process_output_size,
    format_process_poll_countdown_track,
)


def test_process_output_activity_highlights_recent_then_goes_quiet() -> None:
    assert (
        format_process_output_activity(
            has_observed_output=False,
            seconds_since_last_output=90,
        )
        is None
    )

    hot = format_process_output_activity(
        has_observed_output=True,
        seconds_since_last_output=4,
    )
    assert hot is not None
    assert hot.text == "output"
    assert hot.style == "bold bright_green"

    warm = format_process_output_activity(
        has_observed_output=True,
        seconds_since_last_output=12,
    )
    assert warm is not None
    assert warm.text == "output"
    assert warm.style == "green"

    quiet = format_process_output_activity(
        has_observed_output=True,
        seconds_since_last_output=90,
    )
    assert quiet is not None
    assert quiet.text == "quiet"
    assert quiet.style is None


def test_process_output_size_uses_compact_decimal_units() -> None:
    assert format_process_output_size(None) is None
    assert format_process_output_size(0) is None
    assert format_process_output_size(999) == "999B"
    assert format_process_output_size(1_250) == "1.2KB"
    assert format_process_output_size(12_500) == "12.5KB"
    assert format_process_output_size(1_250_000) == "1.2MB"


def test_cell_glyph_fills_top_to_bottom_left_column_first() -> None:
    assert _cell_glyph(0) == " "
    assert _cell_glyph(1) == "⠁"  # dot 1
    assert _cell_glyph(2) == "⠃"  # 1+2
    assert _cell_glyph(4) == "⡇"  # left column full
    assert _cell_glyph(5) == "⡏"  # left full + top-right
    assert _cell_glyph(6) == "⡟"
    assert _cell_glyph(8) == "⣿"


def test_process_poll_countdown_track_drains_right_to_left() -> None:
    assert format_process_poll_countdown_track(wait_seconds=None, elapsed_seconds=0) is None
    assert format_process_poll_countdown_track(wait_seconds=0, elapsed_seconds=0) is None

    full = format_process_poll_countdown_track(wait_seconds=200, elapsed_seconds=0)
    first_blink = format_process_poll_countdown_track(
        wait_seconds=200,
        elapsed_seconds=1,
        blink_next=True,
    )
    first_drop = format_process_poll_countdown_track(wait_seconds=200, elapsed_seconds=9)
    last_dot = format_process_poll_countdown_track(wait_seconds=200, elapsed_seconds=199)
    empty = format_process_poll_countdown_track(wait_seconds=200, elapsed_seconds=200)
    overdue = format_process_poll_countdown_track(wait_seconds=30, elapsed_seconds=45)

    assert full == "⣿⣿⣿"
    assert first_blink == "⣿⣿⡿"
    assert empty == "   "
    assert overdue == " "
    assert first_drop is not None and last_dot is not None
    # Right cell drops first.
    assert first_drop == "⣿⣿" + _cell_glyph(7)
    # Final filled step is the last dot in drain order; blank only at deadline.
    assert last_dot == _cell_glyph(1) + "  "
    assert len(full) == 3


def test_process_poll_countdown_blinks_toward_next_state() -> None:
    assert format_process_poll_countdown_track(wait_seconds=240, elapsed_seconds=0) == "⣿⣿⣿"
    assert format_process_poll_countdown_track(
        wait_seconds=240,
        elapsed_seconds=1,
        blink_next=True,
    ) == "⣿⣿⡿"
    assert format_process_poll_countdown_track(
        wait_seconds=240,
        elapsed_seconds=2,
        blink_next=False,
    ) == "⣿⣿⣿"
    assert format_process_poll_countdown_track(
        wait_seconds=240,
        elapsed_seconds=10,
        blink_next=False,
    ) == "⣿⣿⡿"


def test_process_poll_countdown_scales_track_to_wait_budget() -> None:
    assert _countdown_cell_count(1) == 1
    assert _countdown_cell_count(80) == 1
    assert _countdown_cell_count(81) == 2
    assert _countdown_cell_count(160) == 2
    assert _countdown_cell_count(161) == 3
    assert _countdown_cell_count(600) == 3

    assert format_process_poll_countdown_track(wait_seconds=50, elapsed_seconds=0) == "⣿"
    assert format_process_poll_countdown_track(wait_seconds=100, elapsed_seconds=0) == "⣿⣿"
    assert format_process_poll_countdown_track(wait_seconds=200, elapsed_seconds=0) == "⣿⣿⣿"


def test_countdown_steps_cover_full_range() -> None:
    frames = [_track_from_remaining_units(n) for n in range(25)]
    assert frames[0] == "   "
    assert frames[24] == "⣿⣿⣿"
    assert frames[23] == "⣿⣿" + _cell_glyph(7)
    assert frames[16] == "⣿⣿ "
    assert frames[8] == "⣿  "


def test_step_timing_uses_adaptive_dots_with_10s_cap_and_rotation() -> None:
    # Short waits use one adaptively sized sweep.
    assert _countdown_cycle_count(50) == 1
    assert _countdown_cycle_count(240) == 1
    # Over 240s: extra sweeps so no dot interval exceeds 10s.
    assert _countdown_cycle_count(241) == 2
    assert _countdown_cycle_count(600) == 3  # ceil(600/240)

    assert _remaining_slots_for_wait(wait_seconds=27, elapsed_seconds=0) == 8
    assert _remaining_slots_for_wait(wait_seconds=27, elapsed_seconds=26) == 1
    assert _remaining_slots_for_wait(wait_seconds=27, elapsed_seconds=27) == 0
    assert format_process_poll_countdown_track(
        wait_seconds=27,
        elapsed_seconds=26,
    ) == _cell_glyph(1)
    assert format_process_poll_countdown_track(wait_seconds=27, elapsed_seconds=27) == " "

    assert _remaining_units_for_wait(wait_seconds=600, elapsed_seconds=0) == 24
    mid = _remaining_units_for_wait(wait_seconds=600, elapsed_seconds=300)
    assert 0 <= mid <= 24
    assert _remaining_units_for_wait(wait_seconds=600, elapsed_seconds=600) == 0
    assert format_process_poll_countdown_track(wait_seconds=600, elapsed_seconds=600) == "   "
