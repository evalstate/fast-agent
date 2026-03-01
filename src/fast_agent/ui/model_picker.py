from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.ui.model_picker_common import (
    REFER_TO_DOCS_PROVIDERS,
    ModelOption,
    ModelSource,
    ProviderOption,
    build_snapshot,
    model_options_for_provider,
)

StyleFragments = list[tuple[str, str]]


@dataclass(frozen=True)
class ModelPickerResult:
    provider: str
    provider_available: bool
    selected_model: str | None
    resolved_model: str | None
    source: ModelSource
    refer_to_docs: bool


@dataclass
class PickerState:
    provider_index: int
    model_index: int
    model_scroll_top: int
    focus: Literal["providers", "models"]
    source: ModelSource


class _SplitListPicker:
    LIST_VISIBLE_ROWS = 13

    def __init__(self, *, config_path: Path | None) -> None:
        self.snapshot = build_snapshot(config_path)
        if not self.snapshot.providers:
            raise ValueError("No providers found in model catalog.")

        self.state = PickerState(
            provider_index=self._initial_provider_index(),
            model_index=0,
            model_scroll_top=0,
            focus="providers",
            source="curated",
        )

        self.provider_control = FormattedTextControl(self._render_provider_panel)
        self.model_control = FormattedTextControl(
            self._render_model_panel,
            show_cursor=False,
            get_cursor_position=self._model_cursor_position,
        )
        self.status_control = FormattedTextControl(self._render_status_bar)

        provider_window = Window(
            self.provider_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
        )

        self.model_window = Window(
            self.model_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )

        picker_columns = VSplit(
            [
                Frame(
                    provider_window,
                    title="Providers",
                    width=lambda: self._provider_width(),
                ),
                Frame(self.model_window, title="Models"),
            ],
            padding=1,
        )

        body = HSplit(
            [
                picker_columns,
                Window(height=1, char="─", style="class:muted"),
                Window(self.status_control, height=2),
            ]
        )

        self.app = Application(
            layout=Layout(body),
            key_bindings=self._create_key_bindings(),
            style=Style.from_dict(
                {
                    "selected": "reverse",
                    "active": "ansigreen",
                    "inactive": "ansibrightblack",
                    "muted": "ansibrightblack",
                    "focus": "ansicyan",
                }
            ),
            full_screen=False,
            mouse_support=False,
        )

        self._sync_model_scroll()

    @property
    def current_provider(self) -> ProviderOption:
        return self.snapshot.providers[self.state.provider_index]

    def _provider_requires_docs_only(self) -> bool:
        return self.current_provider.provider in REFER_TO_DOCS_PROVIDERS

    @property
    def current_models(self) -> list[ModelOption]:
        return model_options_for_provider(
            self.snapshot,
            self.current_provider.provider,
            source=self.state.source,
        )

    def _selected_model(self) -> ModelOption | None:
        models = self.current_models
        if not models:
            return None
        self._clamp_model_index()
        self._sync_model_scroll()
        return models[self.state.model_index]

    def _model_cursor_position(self) -> Point | None:
        models = self.current_models
        if not models:
            return None
        self._clamp_model_index()
        return Point(x=0, y=self.state.model_index)

    def _terminal_cols(self) -> int:
        app = get_app_or_none()
        if app is not None:
            try:
                return max(1, app.output.get_size().columns)
            except Exception:
                pass
        return max(1, shutil.get_terminal_size((100, 20)).columns)

    def _provider_width(self) -> int:
        cols = self._terminal_cols()
        return max(30, min(42, cols // 3))

    def _initial_provider_index(self) -> int:
        for index, option in enumerate(self.snapshot.providers):
            if option.active:
                return index
        return 0

    def _clamp_model_index(self) -> None:
        model_count = len(self.current_models)
        if model_count == 0:
            self.state.model_index = 0
            return
        if self.state.model_index >= model_count:
            self.state.model_index = model_count - 1

    def _sync_model_scroll(self) -> None:
        models = self.current_models
        if not models:
            self.state.model_scroll_top = 0
            self.model_window.vertical_scroll = 0
            return

        visible = self.LIST_VISIBLE_ROWS
        max_top = max(0, len(models) - visible)
        top = min(self.state.model_scroll_top, max_top)
        index = self.state.model_index

        if index < top:
            top = index
        elif index >= top + visible:
            top = index - visible + 1

        self.state.model_scroll_top = max(0, min(top, max_top))
        self.model_window.vertical_scroll = self.state.model_scroll_top

    def _move_provider(self, delta: int) -> None:
        count = len(self.snapshot.providers)
        self.state.provider_index = (self.state.provider_index + delta) % count
        self.state.model_index = 0
        self.state.model_scroll_top = 0

    def _move_model(self, delta: int) -> None:
        models = self.current_models
        if not models:
            self.state.model_index = 0
            self.state.model_scroll_top = 0
            return
        self.state.model_index = (self.state.model_index + delta) % len(models)
        self._sync_model_scroll()

    def _toggle_source(self) -> None:
        self.state.source = "all" if self.state.source == "curated" else "curated"
        self.state.model_index = 0
        self.state.model_scroll_top = 0
        self._sync_model_scroll()

    def _row_style(self, *, selected: bool, available: bool) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        if available:
            parts.append("class:active")
        else:
            parts.append("class:inactive")
        return " ".join(parts)

    @staticmethod
    def _provider_display_name(config_name: str, default_name: str) -> str:
        if config_name == "codexresponses":
            return "Codex (Plan)"
        return default_name

    def _render_provider_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, option in enumerate(self.snapshot.providers):
            selected = index == self.state.provider_index
            cursor = "❯ " if self.state.focus == "providers" and selected else "  "
            line_style = self._row_style(selected=selected, available=option.active)
            availability = "available" if option.active else "not configured"
            provider_name = self._provider_display_name(
                option.provider.config_name,
                option.provider.display_name,
            )
            text = (
                f"{cursor}{provider_name:<16} "
                f"[{availability}] ({len(option.curated_entries)} curated)\n"
            )
            fragments.append((line_style, text))
        return fragments

    def _render_model_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        models = self.current_models
        self._clamp_model_index()
        self._sync_model_scroll()

        provider_available = self.current_provider.active
        if not models:
            fragments.append(("class:muted", "  No models in this scope.\n"))
            return fragments

        for index, model in enumerate(models):
            selected = index == self.state.model_index
            cursor = "❯ " if self.state.focus == "models" and selected else "  "
            line_style = self._row_style(selected=selected, available=provider_available)
            marker = "✓" if provider_available else "✗"
            fragments.append((line_style, f"{cursor}{marker} {model.label}\n"))

        return fragments

    def _render_status_bar(self) -> StyleFragments:
        provider = self.current_provider
        provider_name = self._provider_display_name(
            provider.provider.config_name,
            provider.provider.display_name,
        )
        scope = "curated" if self.state.source == "curated" else "all catalog"
        status = "available" if provider.active else "not configured"
        warning = ""
        if self._provider_requires_docs_only():
            warning = " · see docs"

        models = self.current_models
        model_count = len(models)
        model_position = self.state.model_index + 1 if model_count > 0 else 0

        return [
            (
                "class:focus",
                (
                    f"Provider: {provider_name} ({status}) | "
                    f"Scope: {scope} | Focus: {self.state.focus} | "
                    f"Model: {model_position}/{model_count}{warning}\n"
                ),
            ),
            (
                "class:muted",
                "Keys: ←/→ focus · ↑/↓ move · Tab swap · c scope · Enter select · q quit",
            ),
        ]

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("left")
        def _left(event) -> None:
            self.state.focus = "providers"
            event.app.invalidate()

        @kb.add("right")
        def _right(event) -> None:
            self.state.focus = "models"
            event.app.invalidate()

        @kb.add("tab")
        def _tab(event) -> None:
            self.state.focus = "models" if self.state.focus == "providers" else "providers"
            event.app.invalidate()

        @kb.add("up")
        def _up(event) -> None:
            if self.state.focus == "providers":
                self._move_provider(-1)
            else:
                self._move_model(-1)
            event.app.invalidate()

        @kb.add("down")
        def _down(event) -> None:
            if self.state.focus == "providers":
                self._move_provider(1)
            else:
                self._move_model(1)
            event.app.invalidate()

        @kb.add("c")
        def _toggle_scope(event) -> None:
            self._toggle_source()
            event.app.invalidate()

        @kb.add("enter")
        def _accept(event) -> None:
            selected_model = self._selected_model()
            if selected_model is None:
                return

            provider = self.current_provider
            if self._provider_requires_docs_only():
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.provider.config_name,
                        provider_available=provider.active,
                        selected_model=None,
                        resolved_model=None,
                        source=self.state.source,
                        refer_to_docs=True,
                    )
                )
                return

            event.app.exit(
                result=ModelPickerResult(
                    provider=provider.provider.config_name,
                    provider_available=provider.active,
                    selected_model=selected_model.spec,
                    resolved_model=selected_model.spec,
                    source=self.state.source,
                    refer_to_docs=False,
                )
            )

        @kb.add("q")
        @kb.add("escape")
        @kb.add("c-c")
        def _quit(event) -> None:
            event.app.exit(result=None)

        return kb

    def run(self) -> ModelPickerResult | None:
        result = self.app.run()
        if result is None:
            return None
        if isinstance(result, ModelPickerResult):
            return result
        return None

    async def run_async(self) -> ModelPickerResult | None:
        result = await self.app.run_async()
        if result is None:
            return None
        if isinstance(result, ModelPickerResult):
            return result
        return None


def run_model_picker(*, config_path: Path | None = None) -> ModelPickerResult | None:
    """Run the interactive model picker and return the selected model configuration."""
    picker = _SplitListPicker(config_path=config_path)
    return picker.run()


async def run_model_picker_async(*, config_path: Path | None = None) -> ModelPickerResult | None:
    """Run the interactive model picker from within an active asyncio event loop."""
    picker = _SplitListPicker(config_path=config_path)
    return await picker.run_async()
