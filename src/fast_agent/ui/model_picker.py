from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    from pathlib import Path

from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import (
    GENERIC_CUSTOM_MODEL_SENTINEL,
    LLAMACPP_IMPORT_SENTINEL,
    LLAMACPP_PROVIDER_KEY,
    REFER_TO_DOCS_PROVIDERS,
    ModelAvailability,
    ModelOption,
    ModelSource,
    ProviderActivation,
    ProviderOption,
    build_snapshot,
    find_provider,
    model_identity,
    model_options_for_option,
    model_options_for_provider,
    provider_activation_action,
    provider_option_count_label,
)
from fast_agent.ui.picker_theme import build_picker_style
from fast_agent.utils.text import strip_to_none

StyleFragments = list[tuple[str, str]]


@dataclass(frozen=True)
class ModelPickerResult:
    provider: str
    provider_available: bool
    selected_model: str | None
    resolved_model: str | None
    source: ModelSource
    refer_to_docs: bool
    activation_action: ProviderActivation | None = None


@dataclass(frozen=True)
class ProviderAvailability:
    label: str
    style: ModelAvailability
    available: bool


@dataclass(frozen=True)
class ModelAvailabilityDisplay:
    availability: ModelAvailability
    marker: str


@dataclass
class PickerState:
    provider_index: int
    model_index: int
    source: ModelSource


PROVIDER_DISPLAY_NAME_OVERRIDES = {
    "responses": "OpenAI",
    "openai": "OpenAI (Legacy)",
    "codexresponses": "Codex (Plan)",
    "generic": "Generic (ollama)",
    "fast-agent": "fast-agent",
}


_MODEL_AVAILABILITY_MARKERS: dict[ModelAvailability, str] = {
    "active": "✓",
    "attention": "!",
    "inactive": "✗",
}


def _model_availability_display(
    model: ModelOption,
    *,
    provider_available: bool,
) -> ModelAvailabilityDisplay:
    if provider_available:
        availability: ModelAvailability = "active"
    elif model.activation_action is not None:
        availability = "attention"
    else:
        availability = "inactive"
    return ModelAvailabilityDisplay(
        availability=availability,
        marker=_MODEL_AVAILABILITY_MARKERS[availability],
    )


class _SplitListPicker:
    LIST_VISIBLE_ROWS = 15

    def __init__(
        self,
        *,
        config_path: Path | None,
        config_payload: dict[str, object] | None = None,
        start_path: Path | None = None,
        initial_provider: str | None = None,
        initial_model_spec: str | None = None,
    ) -> None:
        self.snapshot = build_snapshot(
            config_path,
            config_payload=config_payload,
            start_path=start_path,
        )
        if not self.snapshot.providers:
            raise ValueError("No providers found in model catalog.")
        self._initial_provider_name = initial_provider
        self._initial_model_spec = strip_to_none(initial_model_spec)

        self.state = PickerState(
            provider_index=self._initial_provider_index(),
            model_index=0,
            source="curated",
        )

        self.provider_control = FormattedTextControl(
            self._render_provider_panel,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._provider_cursor_position,
        )
        self.model_control = FormattedTextControl(
            self._render_model_panel,
            focusable=True,
            show_cursor=False,
            get_cursor_position=self._model_cursor_position,
        )
        self.status_control = FormattedTextControl(self._render_status_bar)

        self.provider_window = Window(
            self.provider_control,
            wrap_lines=False,
            height=Dimension.exact(self.LIST_VISIBLE_ROWS),
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=False)],
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
                    self.provider_window,
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
            layout=Layout(body, focused_element=self.provider_window),
            key_bindings=self._create_key_bindings(),
            style=build_picker_style(),
            full_screen=False,
            mouse_support=False,
        )

        self._apply_initial_model_selection()

    @property
    def current_provider(self) -> ProviderOption:
        return self.snapshot.providers[self.state.provider_index]

    def _provider_is_available(self, option: ProviderOption) -> bool:
        return self._provider_availability(option).available

    def _provider_requires_docs_only(self) -> bool:
        provider = self.current_provider.provider
        return provider in REFER_TO_DOCS_PROVIDERS if provider is not None else False

    def _provider_activation_action(
        self,
        option: ProviderOption | None = None,
    ) -> ProviderActivation | None:
        provider_option = option or self.current_provider
        if provider_option.provider is None:
            return None
        return provider_activation_action(self.snapshot, provider_option.provider)

    @property
    def current_models(self) -> list[ModelOption]:
        provider = self.current_provider.provider
        if provider is not None:
            return model_options_for_provider(
                self.snapshot,
                provider,
                source=self.state.source,
            )
        return model_options_for_option(
            self.current_provider,
            source=self.state.source,
        )

    def _selected_model(self) -> ModelOption | None:
        models = self.current_models
        if not models:
            return None
        self._clamp_model_index()
        return models[self.state.model_index]

    def _model_cursor_position(self) -> Point | None:
        models = self.current_models
        if not models:
            return None
        self._clamp_model_index()
        return Point(x=0, y=self.state.model_index)

    def _provider_cursor_position(self) -> Point | None:
        return Point(x=0, y=self.state.provider_index)

    def _providers_focused(self) -> bool:
        return self.app.layout.has_focus(self.provider_window)

    def _models_focused(self) -> bool:
        return self.app.layout.has_focus(self.model_window)

    def _focused_panel_name(self) -> str:
        return "models" if self._models_focused() else "providers"

    def _focus_providers(self) -> None:
        self.app.layout.focus(self.provider_window)

    def _focus_models(self) -> None:
        self.app.layout.focus(self.model_window)

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
                if option.option_key == self._initial_provider_name:
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
            self.current_provider.option_key,
        )
        for source in ("curated", "all"):
            if provider_option.provider is None:
                models = model_options_for_option(provider_option, source=source)
            else:
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
            self._focus_models()
            return

    def _clamp_model_index(self) -> None:
        model_count = len(self.current_models)
        if model_count == 0:
            self.state.model_index = 0
            return
        if self.state.model_index >= model_count:
            self.state.model_index = model_count - 1

    def _move_provider(self, delta: int) -> None:
        count = len(self.snapshot.providers)
        self.state.provider_index = (self.state.provider_index + delta) % count
        self.state.model_index = 0

    def _move_model(self, delta: int) -> None:
        models = self.current_models
        if not models:
            self.state.model_index = 0
            return
        self.state.model_index = (self.state.model_index + delta) % len(models)

    def _toggle_source(self) -> None:
        self.state.source = "all" if self.state.source == "curated" else "curated"
        self.state.model_index = 0

    def _row_style(
        self,
        *,
        selected: bool,
        availability: ModelAvailability,
    ) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        parts.append(f"class:{availability}")
        return " ".join(parts)

    def _provider_availability_label(self, option: ProviderOption) -> str:
        return self._provider_availability(option).label

    def _provider_availability_style(
        self,
        option: ProviderOption,
    ) -> ModelAvailability:
        return self._provider_availability(option).style

    def _provider_availability(self, option: ProviderOption) -> ProviderAvailability:
        if option.option_key == LLAMACPP_PROVIDER_KEY:
            return ProviderAvailability("available", "active", True)
        if option.overlay_group and not option.curated_entries:
            return ProviderAvailability("none yet", "inactive", False)
        if option.active:
            return ProviderAvailability("available", "active", True)
        if option.disabled_reason is not None:
            return ProviderAvailability("disabled", "attention", False)
        if self._provider_activation_action(option) is not None:
            return ProviderAvailability("sign in required", "attention", False)
        return ProviderAvailability("not configured", "inactive", False)

    @staticmethod
    def _provider_display_name(config_name: str, default_name: str) -> str:
        return PROVIDER_DISPLAY_NAME_OVERRIDES.get(config_name, default_name)

    @classmethod
    def _provider_display_name_for_option(cls, option: ProviderOption) -> str:
        if option.display_name is not None:
            return option.display_name
        provider = option.provider
        if provider is None:
            raise ValueError("Provider option requires display_name when provider is unset")
        return cls._provider_display_name(
            provider.config_name,
            provider.display_name,
        )

    def _model_panel_width(self) -> int:
        cols = self._terminal_cols()
        return max(42, cols - self._provider_width() - 8)

    @staticmethod
    def _truncate_picker_text(value: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(value) <= width:
            return value
        if width == 1:
            return "…"
        return f"{value[: width - 1]}…"

    @classmethod
    def _tabulate_model_label(cls, label: str, *, panel_width: int) -> str:
        if " → " not in label:
            return cls._truncate_picker_text(label, max(panel_width - 4, 8))

        left, right = label.split(" → ", 1)
        name_width = max(14, min(22, panel_width // 3))
        detail_width = max(18, panel_width - name_width - 2)
        return (
            f"{cls._truncate_picker_text(left.strip(), name_width).ljust(name_width)}"
            f"  {cls._truncate_picker_text(right.strip(), detail_width)}"
        )

    def _render_provider_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, option in enumerate(self.snapshot.providers):
            selected = index == self.state.provider_index
            cursor = "❯ " if self._providers_focused() and selected else "  "
            line_style = self._row_style(
                selected=selected,
                availability=self._provider_availability_style(option),
            )
            availability = self._provider_availability_label(option)
            provider_name = self._provider_display_name_for_option(option)
            count_label = provider_option_count_label(option)
            text = f"{cursor}{provider_name:<16} [{availability}] ({count_label})\n"
            fragments.append((line_style, text))
        return fragments

    def _render_model_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        models = self.current_models
        self._clamp_model_index()

        provider_available = self._provider_is_available(self.current_provider)
        if not models:
            empty_message = (
                "  No local overlays found.\n"
                if self.current_provider.overlay_group
                else "  No models in this scope.\n"
            )
            fragments.append(("class:muted", empty_message))
            return fragments

        for index, model in enumerate(models):
            selected = index == self.state.model_index
            cursor = "❯ " if self._models_focused() and selected else "  "
            availability_display = _model_availability_display(
                model,
                provider_available=provider_available,
            )
            line_style = self._row_style(
                selected=selected,
                availability=availability_display.availability,
            )
            fragments.append(
                (
                    line_style,
                    f"{cursor}{availability_display.marker} "
                    f"{self._tabulate_model_label(model.label, panel_width=self._model_panel_width())}\n",
                )
            )

        return fragments

    def _render_status_bar(self) -> StyleFragments:
        provider = self.current_provider
        provider_name = self._provider_display_name_for_option(provider)
        scope = "curated" if self.state.source == "curated" else "all catalog"
        status = self._provider_availability_label(provider)
        warning = ""
        if self._provider_requires_docs_only():
            warning = " · see docs"
        elif provider.disabled_reason is not None:
            warning = f" · {provider.disabled_reason}"
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
                    f"Scope: {scope} | Focus: {self._focused_panel_name()} | "
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
            self._left(event)

        @kb.add("right")
        def _right(event) -> None:
            self._right(event)

        @kb.add("tab")
        def _tab(event) -> None:
            self._tab(event)

        @kb.add("s-tab")
        def _shift_tab(event) -> None:
            self._shift_tab(event)

        @kb.add("up")
        def _up(event) -> None:
            self._up(event)

        @kb.add("down")
        def _down(event) -> None:
            self._down(event)

        @kb.add("c")
        def _toggle_scope(event) -> None:
            self._toggle_scope(event)

        @kb.add("enter")
        def _accept(event) -> None:
            self._accept(event)

        @kb.add("q")
        @kb.add("escape")
        @kb.add("c-c")
        def _quit(event) -> None:
            self._quit(event)

        return kb

    def _left(self, event) -> None:
        self._focus_providers()
        event.app.invalidate()

    def _right(self, event) -> None:
        self._focus_models()
        event.app.invalidate()

    def _tab(self, event) -> None:
        event.app.layout.focus_next()
        event.app.invalidate()

    def _shift_tab(self, event) -> None:
        event.app.layout.focus_previous()
        event.app.invalidate()

    def _up(self, event) -> None:
        if event.app.layout.has_focus(self.provider_window):
            self._move_provider(-1)
        else:
            self._move_model(-1)
        event.app.invalidate()

    def _down(self, event) -> None:
        if event.app.layout.has_focus(self.provider_window):
            self._move_provider(1)
        else:
            self._move_model(1)
        event.app.invalidate()

    def _toggle_scope(self, event) -> None:
        self._toggle_source()
        event.app.invalidate()

    def _accept(self, event) -> None:
        result = self._selected_result()
        if result is not None:
            event.app.exit(result=result)

    def _quit(self, event) -> None:
        event.app.exit(result=None)

    def _selected_result(self) -> ModelPickerResult | None:
        selected_model = self._selected_model()
        if selected_model is None:
            return None

        provider = self.current_provider
        selected_value = self._selected_model_value(provider, selected_model)

        if selected_model.activation_action is not None:
            return self._picker_result(
                provider,
                selected_model=selected_value,
                resolved_model=None,
                refer_to_docs=False,
                activation_action=selected_model.activation_action,
            )
        if self._is_generic_custom_model(provider, selected_model):
            return self._picker_result(
                provider,
                selected_model=selected_value,
                resolved_model=None,
                refer_to_docs=False,
            )
        if self._is_llamacpp_import_model(provider, selected_model):
            return self._picker_result(
                provider,
                selected_model=selected_model.spec,
                resolved_model=None,
                refer_to_docs=False,
            )
        if self._provider_requires_docs_only():
            return self._picker_result(
                provider,
                selected_model=None,
                resolved_model=None,
                refer_to_docs=True,
            )
        return self._picker_result(
            provider,
            selected_model=selected_value,
            resolved_model=selected_value,
            refer_to_docs=False,
        )

    def _picker_result(
        self,
        provider: ProviderOption,
        *,
        selected_model: str | None,
        resolved_model: str | None,
        refer_to_docs: bool,
        activation_action: ProviderActivation | None = None,
    ) -> ModelPickerResult:
        return ModelPickerResult(
            provider=provider.option_key,
            provider_available=self._provider_is_available(provider),
            selected_model=selected_model,
            resolved_model=resolved_model,
            source=self.state.source,
            refer_to_docs=refer_to_docs,
            activation_action=activation_action,
        )

    @staticmethod
    def _selected_model_value(
        provider: ProviderOption,
        selected_model: ModelOption,
    ) -> str:
        if provider.overlay_group and selected_model.preset_token is not None:
            return selected_model.preset_token
        return selected_model.spec

    @staticmethod
    def _is_generic_custom_model(
        provider: ProviderOption,
        selected_model: ModelOption,
    ) -> bool:
        return (
            provider.option_key == Provider.GENERIC.config_name
            and selected_model.spec == GENERIC_CUSTOM_MODEL_SENTINEL
        )

    @staticmethod
    def _is_llamacpp_import_model(
        provider: ProviderOption,
        selected_model: ModelOption,
    ) -> bool:
        return (
            provider.option_key == LLAMACPP_PROVIDER_KEY
            and selected_model.spec == LLAMACPP_IMPORT_SENTINEL
        )

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


async def run_model_picker_async(
    *,
    config_path: Path | None = None,
    config_payload: dict[str, object] | None = None,
    start_path: Path | None = None,
    initial_provider: str | None = None,
    initial_model_spec: str | None = None,
) -> ModelPickerResult | None:
    """Run the interactive model picker from within an active asyncio event loop."""
    picker = _SplitListPicker(
        config_path=config_path,
        config_payload=config_payload,
        start_path=start_path,
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
        if normalized_spec in (option.spec, option.preset_token):
            return index

    target_identity = model_identity(normalized_spec)
    if target_identity is None:
        return None

    for index, option in enumerate(options):
        if model_identity(option.spec) == target_identity:
            return index

    if target_identity[0] == Provider.GENERIC:
        for index, option in enumerate(options):
            if option.spec == GENERIC_CUSTOM_MODEL_SENTINEL:
                return index

    return None
