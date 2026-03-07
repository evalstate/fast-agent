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
    GENERIC_CUSTOM_MODEL_SENTINEL,
    REFER_TO_DOCS_PROVIDERS,
    ModelOption,
    ModelSource,
    ProviderActivationAction,
    ProviderOption,
    build_snapshot,
    find_provider,
    model_identity,
    model_options_for_provider,
    provider_activation_action,
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
    activation_action: ProviderActivationAction | None = None


@dataclass
class PickerState:
    provider_index: int
    model_index: int
    model_scroll_top: int
    focus: Literal["providers", "models"]
    source: ModelSource


class _SplitListPicker:
    LIST_VISIBLE_ROWS = 15

    def __init__(
        self,
        *,
        config_path: Path | None,
        config_payload: dict[str, object] | None = None,
        initial_provider: str | None = None,
        initial_model_spec: str | None = None,
    ) -> None:
        self.snapshot = build_snapshot(config_path, config_payload=config_payload)
        if not self.snapshot.providers:
            raise ValueError("No providers found in model catalog.")
        self._initial_provider_name = initial_provider
        self._initial_model_spec = initial_model_spec.strip() if initial_model_spec else None

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
                    "attention": "ansiyellow",
                    "inactive": "ansibrightblack",
                    "muted": "ansibrightblack",
                    "focus": "ansicyan",
                }
            ),
            full_screen=False,
            mouse_support=False,
        )

        self._apply_initial_model_selection()
        self._sync_model_scroll()

    @property
    def current_provider(self) -> ProviderOption:
        return self.snapshot.providers[self.state.provider_index]

    def _provider_requires_docs_only(self) -> bool:
        return self.current_provider.provider in REFER_TO_DOCS_PROVIDERS

    def _provider_activation_action(
        self,
        option: ProviderOption | None = None,
    ) -> ProviderActivationAction | None:
        provider_option = option or self.current_provider
        return provider_activation_action(self.snapshot, provider_option.provider)

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
        if self._initial_provider_name:
            for index, option in enumerate(self.snapshot.providers):
                if option.provider.config_name == self._initial_provider_name:
                    return index
        for index, option in enumerate(self.snapshot.providers):
            if option.active:
                return index
        return 0

    def _apply_initial_model_selection(self) -> None:
        if not self._initial_model_spec:
            return

        provider_option = find_provider(
            self.snapshot,
            self.current_provider.provider.config_name,
        )
        for source in ("curated", "all"):
            models = model_options_for_provider(
                self.snapshot,
                provider_option.provider,
                source=source,
            )
            match_index = _find_initial_model_index(models, self._initial_model_spec)
            if match_index is None:
                continue
            self.state.source = source
            self.state.model_index = match_index
            self.state.model_scroll_top = 0
            self.state.focus = "models"
            return

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

    def _provider_availability_label(self, option: ProviderOption) -> str:
        if option.active:
            return "available"
        if self._provider_activation_action(option) is not None:
            return "sign in required"
        return "not configured"

    def _provider_availability_style(
        self,
        option: ProviderOption,
    ) -> Literal["active", "attention", "inactive"]:
        if option.active:
            return "active"
        if self._provider_activation_action(option) is not None:
            return "attention"
        return "inactive"

    @staticmethod
    def _provider_display_name(config_name: str, default_name: str) -> str:
        if config_name == "responses":
            return "OpenAI"
        if config_name == "openai":
            return "OpenAI (Legacy)"
        if config_name == "codexresponses":
            return "Codex (Plan)"
        if config_name == "generic":
            return "Local (Generic)"
        if config_name == "fast-agent":
            return "fast-agent"

        return default_name

    def _render_provider_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, option in enumerate(self.snapshot.providers):
            selected = index == self.state.provider_index
            cursor = "❯ " if self.state.focus == "providers" and selected else "  "
            line_style = self._row_style(
                selected=selected,
                availability=self._provider_availability_style(option),
            )
            availability = self._provider_availability_label(option)
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
            line_style = self._row_style(
                selected=selected,
                availability=(
                    "active"
                    if provider_available
                    else "attention"
                    if model.activation_action is not None
                    else "inactive"
                ),
            )
            marker = "✓" if provider_available else "!" if model.activation_action else "✗"
            fragments.append((line_style, f"{cursor}{marker} {model.label}\n"))

        return fragments

    def _render_status_bar(self) -> StyleFragments:
        provider = self.current_provider
        provider_name = self._provider_display_name(
            provider.provider.config_name,
            provider.provider.display_name,
        )
        scope = "curated" if self.state.source == "curated" else "all catalog"
        status = self._provider_availability_label(provider)
        warning = ""
        if self._provider_requires_docs_only():
            warning = " · see docs"
        elif self._provider_activation_action(provider) is not None:
            warning = " · press Enter to log in"

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
                "Keys: ←/→ focus · ↑/↓ move · Tab swap · c scope · Enter select/log in · q quit",
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
            if selected_model.activation_action is not None:
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.provider.config_name,
                        provider_available=provider.active,
                        selected_model=selected_model.spec,
                        resolved_model=None,
                        source=self.state.source,
                        refer_to_docs=False,
                        activation_action=selected_model.activation_action,
                    )
                )
                return

            if (
                provider.provider.config_name == "generic"
                and selected_model.spec == GENERIC_CUSTOM_MODEL_SENTINEL
            ):
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.provider.config_name,
                        provider_available=provider.active,
                        selected_model=selected_model.spec,
                        resolved_model=None,
                        source=self.state.source,
                        refer_to_docs=False,
                        activation_action=None,
                    )
                )
                return

            if self._provider_requires_docs_only():
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.provider.config_name,
                        provider_available=provider.active,
                        selected_model=None,
                        resolved_model=None,
                        source=self.state.source,
                        refer_to_docs=True,
                        activation_action=None,
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
                    activation_action=None,
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


def run_model_picker(
    *,
    config_path: Path | None = None,
    initial_provider: str | None = None,
) -> ModelPickerResult | None:
    """Run the interactive model picker and return the selected model configuration."""
    picker = _SplitListPicker(config_path=config_path, initial_provider=initial_provider)
    return picker.run()


async def run_model_picker_async(
    *,
    config_path: Path | None = None,
    config_payload: dict[str, object] | None = None,
    initial_provider: str | None = None,
    initial_model_spec: str | None = None,
) -> ModelPickerResult | None:
    """Run the interactive model picker from within an active asyncio event loop."""
    picker = _SplitListPicker(
        config_path=config_path,
        config_payload=config_payload,
        initial_provider=initial_provider,
        initial_model_spec=initial_model_spec,
    )
    return await picker.run_async()


def _find_initial_model_index(
    options: list[ModelOption],
    initial_model_spec: str,
) -> int | None:
    normalized_spec = initial_model_spec.strip()
    if not normalized_spec:
        return None

    for index, option in enumerate(options):
        if option.spec == normalized_spec or option.alias == normalized_spec:
            return index

    target_identity = model_identity(normalized_spec)
    if target_identity is None:
        return None

    for index, option in enumerate(options):
        if model_identity(option.spec) == target_identity:
            return index

    return None
