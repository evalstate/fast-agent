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
from prompt_toolkit.widgets import Frame

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.llm.provider_types import Provider
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
    model_options_for_option,
    provider_activation_action,
)
from fast_agent.ui.picker_theme import build_picker_style
from fast_agent.utils.async_utils import suppress_known_runtime_warnings

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
    source: ModelSource


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
        self._initial_model_spec = initial_model_spec.strip() if initial_model_spec else None

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

    def _provider_requires_docs_only(self) -> bool:
        provider = self.current_provider.provider
        return provider in REFER_TO_DOCS_PROVIDERS if provider is not None else False

    def _provider_activation_action(
        self,
        option: ProviderOption | None = None,
    ) -> ProviderActivationAction | None:
        provider_option = option or self.current_provider
        if provider_option.provider is None:
            return None
        return provider_activation_action(self.snapshot, provider_option.provider)

    @property
    def current_models(self) -> list[ModelOption]:
        return model_options_for_option(
            self.snapshot,
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
            models = model_options_for_option(
                self.snapshot,
                provider_option,
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
        availability: Literal["active", "attention", "inactive"],
    ) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        parts.append(f"class:{availability}")
        return " ".join(parts)

    def _provider_availability_label(self, option: ProviderOption) -> str:
        if option.overlay_group and not option.curated_entries:
            return "none yet"
        if option.active:
            return "available"
        if option.disabled_reason is not None:
            return "disabled"
        if self._provider_activation_action(option) is not None:
            return "sign in required"
        return "not configured"

    def _provider_availability_style(
        self,
        option: ProviderOption,
    ) -> Literal["active", "attention", "inactive"]:
        if option.overlay_group and not option.curated_entries:
            return "inactive"
        if option.active:
            return "active"
        if option.disabled_reason is not None:
            return "attention"
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
            return "Local (ollama)"
        if config_name == "fast-agent":
            return "fast-agent"

        return default_name

    @classmethod
    def _provider_display_name_for_option(cls, option: ProviderOption) -> str:
        if option.display_name is not None:
            return option.display_name
        provider = option.provider
        assert provider is not None
        return cls._provider_display_name(
            provider.config_name,
            provider.display_name,
        )

    @staticmethod
    def _provider_entry_count_label(option: ProviderOption) -> str:
        if option.overlay_group:
            entry_count = len(option.curated_entries)
            suffix = "overlay" if entry_count == 1 else "overlays"
            return f"{entry_count} {suffix}"
        return f"{len(option.curated_entries)} curated"

    def _overlay_models(self) -> list[ModelOption]:
        options: list[ModelOption] = []
        for entry in self.current_provider.curated_entries:
            tags: list[str] = []
            if entry.local:
                tags.append("local")
            if entry.fast:
                tags.append("fast")
            if not entry.current:
                tags.append("legacy")

            suffix = f" ({', '.join(tags)})" if tags else ""
            label = f"{(entry.display_label or entry.alias):<19} → {entry.model}{suffix}"
            if entry.description:
                label = f"{label} — {entry.description}"
            options.append(
                ModelOption(
                    spec=entry.model,
                    label=label,
                    preset_token=entry.alias,
                    fast=entry.fast,
                    curated=entry.current,
                )
            )
        return options

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
            count_label = self._provider_entry_count_label(option)
            text = (
                f"{cursor}{provider_name:<16} "
                f"[{availability}] ({count_label})\n"
            )
            fragments.append((line_style, text))
        return fragments

    def _render_model_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        models = self.current_models
        self._clamp_model_index()

        provider_available = self.current_provider.active
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
            fragments.append(
                (
                    line_style,
                    f"{cursor}{marker} "
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
            self._focus_providers()
            event.app.invalidate()

        @kb.add("right")
        def _right(event) -> None:
            self._focus_models()
            event.app.invalidate()

        @kb.add("tab")
        def _tab(event) -> None:
            event.app.layout.focus_next()
            event.app.invalidate()

        @kb.add("s-tab")
        def _shift_tab(event) -> None:
            event.app.layout.focus_previous()
            event.app.invalidate()

        @kb.add("up")
        def _up(event) -> None:
            if event.app.layout.has_focus(self.provider_window):
                self._move_provider(-1)
            else:
                self._move_model(-1)
            event.app.invalidate()

        @kb.add("down")
        def _down(event) -> None:
            if event.app.layout.has_focus(self.provider_window):
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
            selected_value = (
                selected_model.preset_token
                if provider.overlay_group and selected_model.preset_token is not None
                else selected_model.spec
            )
            if selected_model.activation_action is not None:
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.option_key,
                        provider_available=provider.active,
                        selected_model=selected_value,
                        resolved_model=None,
                        source=self.state.source,
                        refer_to_docs=False,
                        activation_action=selected_model.activation_action,
                    )
                )
                return

            if (
                provider.option_key == Provider.GENERIC.config_name
                and selected_model.spec == GENERIC_CUSTOM_MODEL_SENTINEL
            ):
                event.app.exit(
                    result=ModelPickerResult(
                        provider=provider.option_key,
                        provider_available=provider.active,
                        selected_model=selected_value,
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
                        provider=provider.option_key,
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
                    provider=provider.option_key,
                    provider_available=provider.active,
                    selected_model=selected_value,
                    resolved_model=selected_value,
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
        with suppress_known_runtime_warnings():
            result = self.app.run()
        if result is None:
            return None
        if isinstance(result, ModelPickerResult):
            return result
        return None

    async def run_async(self) -> ModelPickerResult | None:
        with suppress_known_runtime_warnings():
            result = await self.app.run_async()
        if result is None:
            return None
        if isinstance(result, ModelPickerResult):
            return result
        return None


def run_model_picker(
    *,
    config_path: Path | None = None,
    start_path: Path | None = None,
    initial_provider: str | None = None,
) -> ModelPickerResult | None:
    """Run the interactive model picker and return the selected model configuration."""
    picker = _SplitListPicker(
        config_path=config_path,
        start_path=start_path,
        initial_provider=initial_provider,
    )
    return picker.run()


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
        if option.spec == normalized_spec or option.preset_token == normalized_spec:
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
