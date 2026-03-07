from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

StyleFragments = list[tuple[str, str]]

type ModelAliasPickerPriority = Literal["required", "repair", "recommended", "configured"]
type ModelAliasPickerAction = Literal["set", "unset", "custom"]


@dataclass(frozen=True)
class ModelAliasPickerItem:
    token: str
    priority: ModelAliasPickerPriority
    status: str
    summary: str
    current_value: str | None
    references: tuple[str, ...]
    removable: bool = False


@dataclass(frozen=True)
class ModelAliasPickerResult:
    action: ModelAliasPickerAction
    token: str | None


@dataclass
class _AliasPickerState:
    index: int = 0
    scroll_top: int = 0


class _AliasPicker:
    LIST_VISIBLE_ROWS = 10

    def __init__(self, items: tuple[ModelAliasPickerItem, ...]) -> None:
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
            height=Dimension.exact(5),
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
                    "required": "ansiyellow",
                    "repair": "ansiyellow",
                    "recommended": "ansigreen",
                    "configured": "ansiwhite",
                    "muted": "ansibrightblack",
                }
            ),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._sync_scroll()

    @property
    def current_item(self) -> ModelAliasPickerItem | None:
        if self.state.index >= len(self.items):
            return None
        return self.items[self.state.index]

    def _terminal_cols(self) -> int:
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.index)

    def _move(self, delta: int) -> None:
        row_count = len(self.items) + 1
        self.state.index = (self.state.index + delta) % row_count
        self._sync_scroll()

    def _sync_scroll(self) -> None:
        visible = self.LIST_VISIBLE_ROWS
        row_count = len(self.items) + 1
        max_top = max(0, row_count - visible)
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
            if self.current_item is None:
                event.app.exit(result=ModelAliasPickerResult(action="custom", token=None))
                return
            event.app.exit(result=ModelAliasPickerResult(action="set", token=self.current_item.token))

        @kb.add("delete")
        @kb.add("backspace")
        @kb.add("x")
        def _remove(event) -> None:  # type: ignore[no-untyped-def]
            item = self.current_item
            if item is None or not item.removable:
                return
            event.app.exit(result=ModelAliasPickerResult(action="unset", token=item.token))

        @kb.add("escape")
        @kb.add("q")
        @kb.add("c-c")
        def _cancel(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result=None)

        return kb

    def _row_style(
        self,
        *,
        selected: bool,
        priority: ModelAliasPickerPriority,
    ) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        parts.append(f"class:{priority}")
        return " ".join(parts)

    def _render_rows(self) -> StyleFragments:
        rows: list[ModelAliasPickerItem | None] = [*self.items, None]
        width = self._terminal_cols()
        status_width = 34
        token_width = max(18, width - status_width - 4)
        fragments: StyleFragments = []
        for index, item in enumerate(rows):
            selected = index == self.state.index
            if item is None:
                priority = "configured"
                token_text = "Add a new alias"
                status_text = "manual entry"
            else:
                priority = item.priority
                token_text = item.token[: token_width - 1]
                if len(item.token) > token_width:
                    token_text = item.token[: token_width - 2] + "…"
                if item.priority == "configured" and item.current_value is not None:
                    status_text = item.current_value
                else:
                    status_text = item.status
                if len(status_text) > status_width:
                    status_text = status_text[: status_width - 1] + "…"
            line_style = self._row_style(selected=selected, priority=priority)
            cursor = "❯ " if selected else "  "
            fragments.append(
                (
                    line_style,
                    f"{cursor}{token_text.ljust(token_width)}  {status_text.ljust(status_width)}\n",
                )
            )
        return fragments

    def _render_details(self) -> StyleFragments:
        item = self.current_item
        if item is None:
            return [
                ("", "Create or update a different alias token.\n"),
                ("class:muted", "current: <manual entry>\n"),
                ("class:muted", "used by: custom setup path\n"),
                ("class:muted", "Enter = type token manually • Esc/Ctrl+C = cancel\n"),
                ("class:muted", "Delete/X removes the selected configured alias."),
            ]

        references_text = ", ".join(item.references) if item.references else "No references"
        current_value = item.current_value if item.current_value is not None else "<unset>"
        remove_hint = (
            "Delete/X = remove alias"
            if item.removable
            else "Delete/X = unavailable for this row"
        )
        return [
            ("", f"{item.summary}\n"),
            ("class:muted", f"current: {current_value}\n"),
            ("class:muted", f"used by: {references_text}\n"),
            ("class:muted", "Enter = configure alias • Esc/Ctrl+C = cancel\n"),
            ("class:muted", remove_hint),
        ]

    async def run_async(self) -> ModelAliasPickerResult | None:
        return await self.app.run_async()


async def run_model_alias_picker_async(
    items: tuple[ModelAliasPickerItem, ...],
) -> ModelAliasPickerResult | None:
    """Run the interactive alias picker."""
    picker = _AliasPicker(items)
    return await picker.run_async()
