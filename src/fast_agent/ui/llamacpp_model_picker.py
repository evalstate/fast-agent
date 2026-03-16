from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from fast_agent.llm.llamacpp_discovery import LlamaCppModelListing

StyleFragments = list[tuple[str, str]]
type LlamaCppPickerAction = Literal["start_now", "start_now_with_shell", "generate_overlay"]


@dataclass(frozen=True)
class LlamaCppModelPickerResult:
    action: LlamaCppPickerAction
    model_id: str


@dataclass(frozen=True)
class _PickerActionOption:
    key: LlamaCppPickerAction
    label: str
    summary: str


@dataclass
class _LlamaCppPickerState:
    model_index: int = 0
    action_index: int = 0
    focus: Literal["models", "actions"] = "models"


class _LlamaCppModelPicker:
    LIST_VISIBLE_ROWS = 10
    _ACTION_OPTIONS: tuple[_PickerActionOption, ...] = (
        _PickerActionOption(
            key="start_now",
            label="Start now",
            summary="Write the overlay and immediately launch fast-agent go.",
        ),
        _PickerActionOption(
            key="start_now_with_shell",
            label="Start now (with shell)",
            summary="Write the overlay and immediately launch fast-agent go -x.",
        ),
        _PickerActionOption(
            key="generate_overlay",
            label="Generate overlay",
            summary="Write a reusable overlay and return to the shell.",
        ),
    )

    def __init__(self, models: tuple[LlamaCppModelListing, ...]) -> None:
        if not models:
            raise ValueError("The llama.cpp model picker requires at least one model.")

        self.models = models
        self.state = _LlamaCppPickerState()
        self.model_control = FormattedTextControl(
            self._render_models,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._model_cursor_position,
        )
        self.action_control = FormattedTextControl(
            self._render_actions,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._action_cursor_position,
        )
        self.details_control = FormattedTextControl(self._render_details)

        self.model_window = Window(
            self.model_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        self.action_window = Window(
            self.action_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
        )
        details_window = Window(
            self.details_control,
            height=Dimension.exact(6),
            dont_extend_height=True,
        )
        columns = VSplit(
            [
                Frame(self.model_window, title="Discovered llama.cpp models"),
                Frame(
                    self.action_window,
                    title="Actions",
                    width=Dimension.exact(28),
                ),
            ],
            padding=1,
        )
        body = HSplit(
            [
                columns,
                Window(height=1, char="─", style="class:muted"),
                details_window,
            ]
        )
        self.app = Application(
            layout=Layout(body, focused_element=self.model_window),
            key_bindings=self._create_key_bindings(),
            style=Style.from_dict(
                {
                    "selected": "reverse",
                    "focus": "ansicyan",
                    "muted": "ansibrightblack",
                }
            ),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    @property
    def current_model(self) -> LlamaCppModelListing:
        return self.models[self.state.model_index]

    @property
    def current_action(self) -> _PickerActionOption:
        return self._ACTION_OPTIONS[self.state.action_index]

    def _terminal_cols(self) -> int:
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _model_cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.model_index)

    def _action_cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.action_index)

    def _move_models(self, delta: int) -> None:
        self.state.model_index = (self.state.model_index + delta) % len(self.models)

    def _move_actions(self, delta: int) -> None:
        self.state.action_index = (self.state.action_index + delta) % len(self._ACTION_OPTIONS)

    def _focus_models(self) -> None:
        self.state.focus = "models"
        self.app.layout.focus(self.model_window)

    def _focus_actions(self) -> None:
        self.state.focus = "actions"
        self.app.layout.focus(self.action_window)

    def _models_focused(self) -> bool:
        return self.state.focus == "models"

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _go_up(event) -> None:
            del event
            if self._models_focused():
                self._move_models(-1)
            else:
                self._move_actions(-1)

        @kb.add("down")
        @kb.add("j")
        def _go_down(event) -> None:
            del event
            if self._models_focused():
                self._move_models(1)
            else:
                self._move_actions(1)

        @kb.add("left")
        @kb.add("h")
        def _go_left(event) -> None:
            del event
            self._focus_models()

        @kb.add("right")
        @kb.add("l")
        @kb.add("tab")
        def _go_right(event) -> None:
            del event
            self._focus_actions()

        @kb.add("s-tab")
        def _go_back(event) -> None:
            del event
            self._focus_models()

        @kb.add("enter")
        def _accept(event) -> None:
            if self._models_focused():
                self._focus_actions()
                return
            event.app.exit(
                result=LlamaCppModelPickerResult(
                    action=self.current_action.key,
                    model_id=self.current_model.model_id,
                )
            )

        @kb.add("escape")
        @kb.add("q")
        @kb.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        return kb

    def _render_models(self) -> StyleFragments:
        width = self._terminal_cols()
        owner_width = 16
        context_width = 16
        model_width = max(20, width - owner_width - context_width - 36)
        fragments: StyleFragments = []
        for index, model in enumerate(self.models):
            selected = index == self.state.model_index
            style_parts: list[str] = []
            if selected:
                style_parts.append("class:selected")
            if selected and self._models_focused():
                style_parts.append("class:focus")
            style = " ".join(style_parts)
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

    def _render_actions(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, action in enumerate(self._ACTION_OPTIONS):
            selected = index == self.state.action_index
            style_parts: list[str] = []
            if selected:
                style_parts.append("class:selected")
            if selected and not self._models_focused():
                style_parts.append("class:focus")
            style = " ".join(style_parts)
            cursor = "❯ " if selected else "  "
            fragments.append((style, f"{cursor}{action.label}\n"))
        return fragments

    def _render_details(self) -> StyleFragments:
        model = self.current_model
        action = self.current_action
        owner = model.owned_by or "<unknown>"
        training_context = (
            str(model.training_context_window)
            if model.training_context_window is not None
            else "<unknown>"
        )
        focus_hint = "models" if self._models_focused() else "actions"
        return [
            ("", f"{model.model_id}\n"),
            ("class:muted", f"owner: {owner}\n"),
            ("class:muted", f"training context: {training_context}\n"),
            ("class:muted", f"selected action: {action.label} — {action.summary}\n"),
            (
                "class:muted",
                f"Focus: {focus_hint}. Left/right or Tab switches panes.\n",
            ),
            (
                "class:muted",
                "Enter on models = choose action • Enter on actions = continue • Esc/Ctrl+C = cancel",
            ),
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
