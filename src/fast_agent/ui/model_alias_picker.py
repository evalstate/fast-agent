from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from fast_agent.llm.model_alias_diagnostics import ModelAliasSetupItem

StyleFragments = list[tuple[str, str]]


@dataclass
class _AliasPickerState:
    index: int = 0
    scroll_top: int = 0


class _AliasPicker:
    LIST_VISIBLE_ROWS = 10

    def __init__(self, items: tuple[ModelAliasSetupItem, ...]) -> None:
        if not items:
            raise ValueError("No model alias setup items available.")
        self.items = items
        self.state = _AliasPickerState()
        self.selection_control = FormattedTextControl(
            self._render_rows,
            show_cursor=False,
            get_cursor_position=self._cursor_position,
        )
        self.details_control = FormattedTextControl(self._render_details)
        self.selection_window = Window(
            self.selection_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        details_window = Window(
            self.details_control,
            height=Dimension.exact(4),
            dont_extend_height=True,
        )
        body = HSplit(
            [
                Frame(self.selection_window, title="Aliases to configure"),
                details_window,
            ]
        )
        self.app = Application(
            layout=Layout(body),
            key_bindings=self._create_key_bindings(),
            style=Style.from_dict(
                {
                    "selected": "reverse",
                    "item": "#dddddd",
                    "required": "#ffcc66",
                    "repair": "#ff8080",
                    "recommended": "#7fd4d4",
                    "muted": "#777777",
                    "status": "#dddddd",
                }
            ),
            full_screen=False,
            mouse_support=False,
        )
        self._sync_scroll()

    @property
    def current_item(self) -> ModelAliasSetupItem:
        return self.items[self.state.index]

    def _terminal_cols(self) -> int:
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.index)

    def _move(self, delta: int) -> None:
        self.state.index = (self.state.index + delta) % len(self.items)
        self._sync_scroll()

    def _sync_scroll(self) -> None:
        visible = self.LIST_VISIBLE_ROWS
        max_top = max(0, len(self.items) - visible)
        top = min(self.state.scroll_top, max_top)
        index = self.state.index
        if index < top:
            top = index
        elif index >= top + visible:
            top = index - visible + 1
        self.state.scroll_top = max(0, min(top, max_top))
        self.selection_window.vertical_scroll = self.state.scroll_top

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _go_up(event) -> None:  # type: ignore[no-untyped-def]
            del event
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _go_down(event) -> None:  # type: ignore[no-untyped-def]
            del event
            self._move(1)

        @kb.add("enter")
        def _accept(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result=self.current_item.token)

        @kb.add("escape")
        @kb.add("q")
        def _cancel(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result=None)

        return kb

    def _render_rows(self) -> StyleFragments:
        width = self._terminal_cols()
        status_width = 25
        token_width = max(18, width - status_width - 8)
        fragments: StyleFragments = []
        for index, item in enumerate(self.items):
            style = "class:selected " if index == self.state.index else ""
            priority_style = {
                "required": "class:required",
                "repair": "class:repair",
                "recommended": "class:recommended",
            }[item.priority]
            token_text = item.token[: token_width - 1]
            if len(item.token) > token_width:
                token_text = item.token[: token_width - 2] + "…"
            status_text = f"{item.priority}/{item.status}"
            fragments.extend(
                [
                    (style, " "),
                    (f"{style}class:item", token_text.ljust(token_width)),
                    (style, "  "),
                    (f"{style}{priority_style}", status_text.ljust(status_width)),
                    (style, "\n"),
                ]
            )
        return fragments

    def _render_details(self) -> StyleFragments:
        item = self.current_item
        references_text = ", ".join(item.references) if item.references else "No references"
        current_value = item.current_value if item.current_value is not None else "<unset>"
        lines = [
            ("", f"{item.summary}\n"),
            ("class:muted", f"current: {current_value}\n"),
            ("class:muted", f"used by: {references_text}\n"),
            (
                "class:muted",
                "Enter = configure alias • Esc = enter a different alias manually",
            ),
        ]
        return lines

    async def run_async(self) -> str | None:
        return await self.app.run_async()


async def run_model_alias_picker_async(
    items: tuple[ModelAliasSetupItem, ...],
) -> str | None:
    """Run the interactive alias picker and return the selected token."""
    picker = _AliasPicker(items)
    return await picker.run_async()
