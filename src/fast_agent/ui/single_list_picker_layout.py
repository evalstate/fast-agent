from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.controls import FormattedTextControl


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

    selection_window = Window(
        selection_control,
        wrap_lines=False,
        height=Dimension.exact(visible_rows),
        dont_extend_height=True,
        ignore_content_width=True,
        always_hide_cursor=True,
        right_margins=[ScrollbarMargin(display_arrows=False)],
    )
    details_window = Window(
        details_control,
        height=Dimension.exact(details_rows),
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
