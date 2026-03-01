from __future__ import annotations

from fast_agent.ui.model_picker import _SplitListPicker


def test_models_window_vertical_scroll_tracks_picker_scroll_state() -> None:
    picker = _SplitListPicker(config_path=None)
    picker.state.source = "all"

    provider_index: int | None = None
    for index, _ in enumerate(picker.snapshot.providers):
        picker.state.provider_index = index
        picker.state.model_index = 0
        picker.state.model_scroll_top = 0
        picker._sync_model_scroll()
        if len(picker.current_models) > picker.LIST_VISIBLE_ROWS:
            provider_index = index
            break

    assert provider_index is not None

    picker.state.provider_index = provider_index
    picker.state.model_index = 0
    picker.state.model_scroll_top = 0
    picker._sync_model_scroll()

    assert picker.model_window.vertical_scroll == 0

    for _ in range(picker.LIST_VISIBLE_ROWS + 1):
        picker._move_model(1)

    assert picker.state.model_scroll_top > 0
    assert picker.model_window.vertical_scroll == picker.state.model_scroll_top

    cursor = picker._model_cursor_position()
    assert cursor is not None
    assert cursor.y == picker.state.model_index
