from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl

if TYPE_CHECKING:
    from fast_agent.llm.llamacpp_discovery import LlamaCppModelListing

from fast_agent.ui.single_list_picker_layout import build_single_list_picker_app

StyleFragments = list[tuple[str, str]]
type LlamaCppPickerAction = Literal["generate_overlay"]


@dataclass(frozen=True)
class LlamaCppModelPickerResult:
    action: LlamaCppPickerAction
    model_id: str


@dataclass
class _LlamaCppPickerState:
    index: int = 0


class _LlamaCppModelPicker:
    LIST_VISIBLE_ROWS = 10

    def __init__(self, models: tuple[LlamaCppModelListing, ...]) -> None:
        if not models:
            raise ValueError("The llama.cpp model picker requires at least one model.")

        self.models = models
        self.state = _LlamaCppPickerState()
        self.selection_control = FormattedTextControl(
            self._render_rows,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._cursor_position,
        )
        self.details_control = FormattedTextControl(self._render_details)
        self.app, self.selection_window = build_single_list_picker_app(
            title="Discovered llama.cpp models",
            selection_control=self.selection_control,
            details_control=self.details_control,
            key_bindings=self._create_key_bindings(),
            visible_rows=self.LIST_VISIBLE_ROWS,
            details_rows=5,
            style_map={
                "selected": "reverse",
                "muted": "ansibrightblack",
                "owner": "ansicyan",
            },
        )

    @property
    def current_model(self) -> LlamaCppModelListing:
        return self.models[self.state.index]

    def _terminal_cols(self) -> int:
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.index)

    def _move(self, delta: int) -> None:
        self.state.index = (self.state.index + delta) % len(self.models)

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
            event.app.exit(
                result=LlamaCppModelPickerResult(
                    action="generate_overlay",
                    model_id=self.current_model.model_id,
                )
            )

        @kb.add("escape")
        @kb.add("q")
        @kb.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        return kb

    def _render_rows(self) -> StyleFragments:
        width = self._terminal_cols()
        owner_width = 16
        context_width = 16
        model_width = max(20, width - owner_width - context_width - 6)
        fragments: StyleFragments = []
        for index, model in enumerate(self.models):
            selected = index == self.state.index
            style = "class:selected" if selected else ""
            owner = model.owned_by or "-"
            training_context = (
                str(model.training_context_window)
                if model.training_context_window is not None
                else "-"
            )
            model_text = model.model_id
            if len(model_text) > model_width:
                model_text = f"{model_text[: model_width - 1]}…"
            cursor = "❯ " if selected else "  "
            fragments.append(
                (
                    style,
                    f"{cursor}{model_text.ljust(model_width)}  {owner.ljust(owner_width)}  {training_context.rjust(context_width)}\n",
                )
            )
        return fragments

    def _render_details(self) -> StyleFragments:
        model = self.current_model
        owner = model.owned_by or "<unknown>"
        training_context = (
            str(model.training_context_window)
            if model.training_context_window is not None
            else "<unknown>"
        )
        return [
            ("", f"{model.model_id}\n"),
            ("class:muted", f"owner: {owner}\n"),
            ("class:muted", f"training context: {training_context}\n"),
            ("class:muted", "Enter = generate overlay for the selected model\n"),
            ("class:muted", "Esc/Ctrl+C = cancel"),
        ]

    async def run_async(self) -> LlamaCppModelPickerResult | None:
        result = await self.app.run_async()
        if result is None:
            return None
        if isinstance(result, LlamaCppModelPickerResult):
            return result
        return None


async def run_llamacpp_model_picker_async(
    models: tuple[LlamaCppModelListing, ...],
) -> LlamaCppModelPickerResult | None:
    """Run the interactive llama.cpp model picker."""

    picker = _LlamaCppModelPicker(models)
    return await picker.run_async()
