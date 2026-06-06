from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl

from fast_agent.ui.single_list_picker_layout import build_single_list_picker_app

StyleFragments = list[tuple[str, str]]

type ModelReferencePickerPriority = Literal["required", "repair", "recommended", "configured"]
type ModelReferencePickerAction = Literal["set", "unset", "custom", "done"]


@dataclass(frozen=True)
class ModelReferencePickerItem:
    token: str
    priority: ModelReferencePickerPriority
    status: str
    summary: str
    current_value: str | None
    references: tuple[str, ...]
    removable: bool = False


@dataclass(frozen=True)
class ModelReferencePickerResult:
    action: ModelReferencePickerAction
    token: str | None


@dataclass
class _ReferencePickerState:
    index: int = 0


@dataclass(frozen=True)
class _PickerControlRow:
    token_text: str
    status_text: str
    priority: ModelReferencePickerPriority = "configured"


_CUSTOM_ROW = _PickerControlRow(token_text="Add a new reference", status_text="manual entry")
_DONE_ROW = _PickerControlRow(token_text="Done", status_text="exit setup")
_CUSTOM_RESULT = ModelReferencePickerResult(action="custom", token=None)
_DONE_RESULT = ModelReferencePickerResult(action="done", token=None)


class _ReferencePicker:
    LIST_VISIBLE_ROWS = 10

    def __init__(self, items: tuple[ModelReferencePickerItem, ...]) -> None:
        self.items = items
        self.state = _ReferencePickerState()
        self.selection_control = FormattedTextControl(
            self._render_rows,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._cursor_position,
        )
        self.details_control = FormattedTextControl(self._render_details)
        self.app, self.selection_window = build_single_list_picker_app(
            title="References to configure",
            selection_control=self.selection_control,
            details_control=self.details_control,
            key_bindings=self._create_key_bindings(),
            visible_rows=self.LIST_VISIBLE_ROWS,
            details_rows=5,
            style_map={
                "selected": "reverse",
                "required": "ansiyellow",
                "repair": "ansiyellow",
                "recommended": "ansigreen",
                "configured": "ansiwhite",
                "muted": "ansibrightblack",
            },
        )

    @property
    def current_item(self) -> ModelReferencePickerItem | None:
        if self.state.index >= len(self.items):
            return None
        return self.items[self.state.index]

    def _is_custom_row(self) -> bool:
        return self.state.index == len(self.items)

    def _is_done_row(self) -> bool:
        return self.state.index == len(self.items) + 1

    def _terminal_cols(self) -> int:
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.index)

    def _move(self, delta: int) -> None:
        row_count = len(self.items) + 2
        self.state.index = (self.state.index + delta) % row_count

    def _accept_result(self) -> ModelReferencePickerResult | None:
        if self._is_done_row():
            return _DONE_RESULT
        if self._is_custom_row():
            return _CUSTOM_RESULT
        item = self.current_item
        if item is None:
            return None
        return ModelReferencePickerResult(action="set", token=item.token)

    def _remove_result(self) -> ModelReferencePickerResult | None:
        item = self.current_item
        if item is None or not item.removable:
            return None
        return ModelReferencePickerResult(action="unset", token=item.token)

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _go_up(event) -> None:
            del event
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _go_down(event) -> None:
            del event
            self._move(1)

        @kb.add("enter")
        def _accept(event) -> None:
            result = self._accept_result()
            if result is None:
                return
            event.app.exit(result=result)

        @kb.add("delete")
        @kb.add("backspace")
        @kb.add("x")
        def _remove(event) -> None:
            result = self._remove_result()
            if result is None:
                return
            event.app.exit(result=result)

        @kb.add("escape")
        @kb.add("q")
        @kb.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        return kb

    def _row_style(
        self,
        *,
        selected: bool,
        priority: ModelReferencePickerPriority,
    ) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        parts.append(f"class:{priority}")
        return " ".join(parts)

    def _render_rows(self) -> StyleFragments:
        rows: list[ModelReferencePickerItem | _PickerControlRow] = [
            *self.items,
            _CUSTOM_ROW,
            _DONE_ROW,
        ]
        width = self._terminal_cols()
        status_width = 34
        token_width = max(18, width - status_width - 4)
        fragments: StyleFragments = []
        for index, item in enumerate(rows):
            selected = index == self.state.index
            if isinstance(item, _PickerControlRow):
                priority = item.priority
                token_text = item.token_text
                status_text = item.status_text
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
        if self._is_custom_row():
            return [
                ("", "Create or update a different reference token.\n"),
                ("class:muted", "current: <manual entry>\n"),
                ("class:muted", "used by: custom setup path\n"),
                ("class:muted", "Enter = type token manually • Esc/Ctrl+C = cancel\n"),
                ("class:muted", "Delete/X removes the selected configured reference."),
            ]
        if self._is_done_row():
            return [
                ("", "Finish model setup and return to the shell.\n"),
                ("class:muted", "current: <none>\n"),
                ("class:muted", "used by: setup session\n"),
                ("class:muted", "Enter = exit setup • Esc/Ctrl+C = cancel\n"),
                ("class:muted", "Delete/X removes the selected configured reference."),
            ]

        if item is None:
            return [
                ("", "No reference selected.\n"),
                ("class:muted", "Enter = unavailable • Esc/Ctrl+C = cancel\n"),
            ]
        references_text = ", ".join(item.references) if item.references else "No references"
        current_value = item.current_value if item.current_value is not None else "<unset>"
        remove_hint = (
            "Delete/X = remove reference"
            if item.removable
            else "Delete/X = unavailable for this row"
        )
        return [
            ("", f"{item.summary}\n"),
            ("class:muted", f"current: {current_value}\n"),
            ("class:muted", f"used by: {references_text}\n"),
            ("class:muted", "Enter = configure reference • Esc/Ctrl+C = cancel\n"),
            ("class:muted", remove_hint),
        ]

    async def run_async(self) -> ModelReferencePickerResult | None:
        result = await self.app.run_async()
        if result is None:
            return None
        if isinstance(result, ModelReferencePickerResult):
            return result
        return None


async def run_model_reference_picker_async(
    items: tuple[ModelReferencePickerItem, ...],
) -> ModelReferencePickerResult | None:
    """Run the interactive reference picker."""
    picker = _ReferencePicker(items)
    return await picker.run_async()
