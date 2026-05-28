from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Literal

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from fast_agent.llm.llamacpp_discovery import LlamaCppModelListing

from fast_agent.ui.picker_theme import build_picker_style
from fast_agent.utils.async_utils import suppress_known_runtime_warnings

StyleFragments = list[tuple[str, str]]
type LlamaCppPickerAction = Literal[
    "start_now",
    "start_now_with_shell",
    "start_now_smart",
    "generate_overlay",
]


@dataclass(frozen=True)
class LlamaCppModelPickerResult:
    action: LlamaCppPickerAction
    model_id: str


@dataclass(frozen=True)
class LlamaCppModelPickerContext:
    runtime_context_window: int | None
    training_context_window: int | None = None


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
    ACTION_PANEL_WIDTH = 28
    _LAUNCH_ACTION_OPTIONS: tuple[_PickerActionOption, ...] = (
        _PickerActionOption(
            key="start_now",
            label="Start now ",
            summary="Write the overlay and immediately launch fast-agent go.",
        ),
        _PickerActionOption(
            key="start_now_with_shell",
            label="Start now (with shell) ",
            summary="Write the overlay and immediately launch fast-agent go -x.",
        ),
        _PickerActionOption(
            key="start_now_smart",
            label="Start now (Smart) ",
            summary="Write the overlay and immediately launch fast-agent go --smart -x.",
        ),
        _PickerActionOption(
            key="generate_overlay",
            label="Write overlay only ",
            summary="Write a reusable overlay and return to the shell.",
        ),
    )

    def __init__(
        self,
        models: tuple[LlamaCppModelListing, ...],
        runtime_context_loader: Callable[
            [str], Awaitable[LlamaCppModelPickerContext | int | None]
        ]
        | None = None,
    ) -> None:
        if not models:
            raise ValueError("The llama.cpp model picker requires at least one model.")

        self.models = models
        self._runtime_context_loader = runtime_context_loader
        self._runtime_context_by_model: dict[str, int | None] = {}
        self._training_context_by_model: dict[str, int | None] = {}
        self._runtime_context_loaded: set[str] = set()
        self._runtime_context_loading: set[str] = set()
        self._runtime_context_errors: set[str] = set()
        self._runtime_context_tasks: set[asyncio.Task[None]] = set()
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
            height=Dimension.exact(7),
            dont_extend_height=True,
        )
        columns = VSplit(
            [
                Frame(self.model_window, title="Discovered llama.cpp models"),
                Frame(
                    self.action_window,
                    title="Actions",
                    width=Dimension.exact(self.ACTION_PANEL_WIDTH),
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
            style=build_picker_style(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    @property
    def current_model(self) -> LlamaCppModelListing:
        return self.models[self.state.model_index]

    @property
    def current_action(self) -> _PickerActionOption:
        return self.action_options[self.state.action_index]

    @property
    def action_options(self) -> tuple[_PickerActionOption, ...]:
        return self._LAUNCH_ACTION_OPTIONS

    def _terminal_cols(self) -> int:
        app = get_app_or_none()
        if app is not None:
            try:
                return max(1, app.output.get_size().columns)
            except Exception:
                pass
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _model_cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.model_index)

    def _action_cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.action_index)

    def _move_models(self, delta: int) -> None:
        self.state.model_index = (self.state.model_index + delta) % len(self.models)
        self._ensure_runtime_context_loading()

    def _move_actions(self, delta: int) -> None:
        self.state.action_index = (self.state.action_index + delta) % len(self.action_options)

    def _row_style(
        self,
        *,
        selected: bool,
        availability: Literal["active", "attention", "inactive"],
    ) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        parts.append(f"class:{availability}")
        return " ".join(parts)

    def _focus_models(self) -> None:
        self.state.focus = "models"
        self.app.layout.focus(self.model_window)
        self._ensure_runtime_context_loading()

    def _focus_actions(self) -> None:
        self.state.focus = "actions"
        self.app.layout.focus(self.action_window)

    def _models_focused(self) -> bool:
        return self.state.focus == "models"

    @staticmethod
    def _truncate_picker_text(value: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(value) <= width:
            return value
        if width == 1:
            return "…"
        return f"{value[: width - 1]}…"

    def _model_panel_width(self) -> int:
        return max(34, self._terminal_cols() - self.ACTION_PANEL_WIDTH - 8)

    @staticmethod
    def _training_context_label(training_context_window: int | None) -> str:
        if training_context_window is None:
            return "train ?"
        return f"train {training_context_window}"

    def _training_context_for_model(self, model: "LlamaCppModelListing") -> int | None:
        if model.model_id in self._training_context_by_model:
            return self._training_context_by_model[model.model_id]
        return model.training_context_window

    def _runtime_context_label(self, model_id: str) -> str:
        if model_id in self._runtime_context_errors:
            return "unavailable"
        if model_id in self._runtime_context_loading:
            return "loading..."
        if model_id not in self._runtime_context_loaded:
            return "not loaded"
        runtime_context = self._runtime_context_by_model.get(model_id)
        return "?" if runtime_context is None else str(runtime_context)

    def _ensure_runtime_context_loading(self) -> None:
        model_id = self.current_model.model_id
        if self._runtime_context_loader is None:
            return
        if model_id in self._runtime_context_loaded or model_id in self._runtime_context_loading:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        self._runtime_context_loading.add(model_id)
        task = asyncio.create_task(self._load_runtime_context(model_id))
        self._runtime_context_tasks.add(task)
        task.add_done_callback(self._runtime_context_tasks.discard)

    async def _load_runtime_context(self, model_id: str) -> None:
        try:
            assert self._runtime_context_loader is not None
            loaded_context = await self._runtime_context_loader(model_id)
            if isinstance(loaded_context, LlamaCppModelPickerContext):
                self._runtime_context_by_model[model_id] = loaded_context.runtime_context_window
                if loaded_context.training_context_window is not None:
                    self._training_context_by_model[model_id] = (
                        loaded_context.training_context_window
                    )
            else:
                self._runtime_context_by_model[model_id] = loaded_context
            self._runtime_context_loaded.add(model_id)
            self._runtime_context_errors.discard(model_id)
        except Exception:
            self._runtime_context_errors.add(model_id)
        finally:
            self._runtime_context_loading.discard(model_id)
            try:
                self.app.invalidate()
            except Exception:
                pass

    @classmethod
    def _model_row_label(
        cls,
        model: LlamaCppModelListing,
        *,
        width: int,
    ) -> str:
        return cls._truncate_picker_text(model.model_id, width)

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
        panel_width = self._model_panel_width()
        fragments: StyleFragments = []
        for index, model in enumerate(self.models):
            selected = index == self.state.model_index
            style = self._row_style(
                selected=selected,
                availability="active",
            )
            cursor = "❯ " if selected and self._models_focused() else "  "
            fragments.append(
                (
                    style,
                    f"{cursor}{self._model_row_label(model, width=panel_width)}\n",
                )
            )
        return fragments

    def _render_actions(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, action in enumerate(self.action_options):
            selected = index == self.state.action_index
            style = self._row_style(
                selected=selected,
                availability="active",
            )
            cursor = "❯ " if selected and not self._models_focused() else "  "
            fragments.append((style, f"{cursor}{action.label}\n"))
        return fragments

    def _render_details(self) -> StyleFragments:
        model = self.current_model
        action = self.current_action
        training_context = self._training_context_label(self._training_context_for_model(model)).replace(
            "train ",
            "",
        )
        runtime_context = self._runtime_context_label(model.model_id)
        focus_hint = "models" if self._models_focused() else "actions"
        return [
            ("", f"{model.model_id}\n"),
            (
                "class:focus",
                f"context: training: {training_context} / runtime: {runtime_context}\n",
            ),
            ("class:muted", f"selected action: {action.label} — {action.summary}\n"),
            ("class:muted", "Import writes a reusable overlay for this model.\n"),
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
        self._ensure_runtime_context_loading()
        try:
            with suppress_known_runtime_warnings():
                result = await self.app.run_async()
        finally:
            for task in tuple(self._runtime_context_tasks):
                task.cancel()
            if self._runtime_context_tasks:
                await asyncio.gather(*self._runtime_context_tasks, return_exceptions=True)
            self._runtime_context_tasks.clear()
        if result is None:
            return None
        if isinstance(result, LlamaCppModelPickerResult):
            return result
        return None


async def run_llamacpp_model_picker_async(
    models: tuple[LlamaCppModelListing, ...],
    runtime_context_loader: Callable[[str], Awaitable[LlamaCppModelPickerContext | int | None]]
    | None = None,
) -> LlamaCppModelPickerResult | None:
    """Run the interactive llama.cpp model picker."""

    picker = _LlamaCppModelPicker(models, runtime_context_loader=runtime_context_loader)
    return await picker.run_async()
