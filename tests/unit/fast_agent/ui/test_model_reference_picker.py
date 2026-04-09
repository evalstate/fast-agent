"""
Testing notes:

- This module owns the lightweight prompt-toolkit behavior of the model
  reference picker.
- Small synthetic picker items are appropriate here because the contract under
  test is window focus/scrolling, not model-reference discovery.
"""

from __future__ import annotations

from fast_agent.ui.model_reference_picker import (
    ModelReferencePickerItem,
    _ReferencePicker,
)


def _build_items(count: int) -> tuple[ModelReferencePickerItem, ...]:
    return tuple(
        ModelReferencePickerItem(
            token=f"MODEL_{index}",
            priority="recommended",
            status="recommended",
            summary=f"Summary {index}",
            current_value=None,
            references=(f"agent_{index}",),
        )
        for index in range(count)
    )


def test_reference_picker_uses_prompt_toolkit_initial_focus() -> None:
    picker = _ReferencePicker(_build_items(3))

    assert picker.app.layout.has_focus(picker.selection_window)


def test_reference_picker_window_scrolls_to_keep_cursor_visible() -> None:
    picker = _ReferencePicker(_build_items(12))
    picker.state.index = 11

    content = picker.selection_control.create_content(width=80, height=picker.LIST_VISIBLE_ROWS)

    picker.selection_window._scroll_without_linewrapping(
        content,
        width=80,
        height=picker.LIST_VISIBLE_ROWS,
    )

    assert picker.selection_window.vertical_scroll == 11 - picker.LIST_VISIBLE_ROWS + 1
