from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.controls import FormattedTextControl


def picker_terminal_size() -> Size:
    app = get_app_or_none()
    if app is not None:
        return app.output.get_size()
    cols, rows = shutil.get_terminal_size((100, 20))
    return Size(rows=rows, columns=cols)


def truncate_picker_text(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if get_cwidth(value) <= width:
        return value
    if width == 1:
        return "…"
    budget = width - 1
    parts: list[str] = []
    used = 0
    for char in value:
        char_width = max(0, get_cwidth(char))
        if used + char_width > budget:
            break
        parts.append(char)
        used += char_width
    return "".join(parts) + "…"


def build_single_list_picker_app(
    *,
    title: str,
    selection_control: FormattedTextControl,
    details_control: FormattedTextControl,
    key_bindings: KeyBindings,
    visible_rows: int,
    details_rows: int,
    style_map: dict[str, str],
) -> tuple[Application[object], Window]:
    """Build a standard single-list picker layout."""

    def details_height() -> Dimension:
        # Keep up to three selection rows before reserving space for details.
        rows = min(details_rows, max(0, picker_terminal_size().rows - 2 - 3))
        return Dimension(preferred=rows, max=rows)

    def selection_height() -> Dimension:
        rows = min(
            visible_rows, max(1, picker_terminal_size().rows - 2 - details_height().preferred)
        )
        return Dimension(preferred=rows, max=rows)

    selection_window = Window(
        selection_control,
        wrap_lines=False,
        height=selection_height,
        dont_extend_height=True,
        ignore_content_width=True,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin(display_arrows=False)],
    )
    details_window = Window(
        details_control,
        height=details_height,
        dont_extend_height=True,
    )
    body = HSplit(
        [
            Frame(selection_window, title=title),
            details_window,
        ]
    )
    app = Application(
        layout=Layout(body, focused_element=selection_window),
        key_bindings=key_bindings,
        style=Style.from_dict(style_map),
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )
    return app, selection_window
