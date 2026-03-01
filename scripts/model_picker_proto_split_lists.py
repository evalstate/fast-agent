from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _model_picker_common import (
    DEFAULT_VALUE,
    KEEP_VALUE,
    REFER_TO_DOCS_PROVIDERS,
    ModelOption,
    ModelSource,
    apply_option_overrides,
    build_snapshot,
    model_capabilities,
    model_options_for_provider,
    web_search_display,
)
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

StyleFragments = list[tuple[str, str]]


@dataclass
class PickerState:
    provider_index: int
    model_index: int
    focus: Literal["providers", "models"]
    source: ModelSource
    reasoning_choice: str
    web_search_choice: str
    context_choice: str
    cache_ttl_choice: str


class SplitListPicker:
    def __init__(self, *, config_path: Path | None) -> None:
        self.snapshot = build_snapshot(config_path)
        if not self.snapshot.providers:
            raise ValueError("No providers found in model catalog.")

        self.state = PickerState(
            provider_index=self._initial_provider_index(),
            model_index=0,
            focus="providers",
            source="curated",
            reasoning_choice=KEEP_VALUE,
            web_search_choice=KEEP_VALUE,
            context_choice=KEEP_VALUE,
            cache_ttl_choice=KEEP_VALUE,
        )

        self.provider_control = FormattedTextControl(self._render_provider_panel)
        self.model_control = FormattedTextControl(self._render_model_panel)
        self.options_control = FormattedTextControl(self._render_options_panel)
        self.status_control = FormattedTextControl(self._render_status_bar)

        picker_columns = VSplit(
            [
                Frame(Window(self.provider_control, wrap_lines=False), title="Providers"),
                HSplit(
                    [
                        Frame(Window(self.model_control, wrap_lines=False), title="Models"),
                        Frame(Window(self.options_control, height=8), title="Options"),
                    ],
                    padding=1,
                ),
            ],
            padding=1,
        )

        body = HSplit(
            [
                Frame(picker_columns, title="Model Picker"),
                Window(height=1, char="─", style="class:muted"),
                Window(self.status_control, height=2),
            ]
        )

        self.app = Application(
            layout=Layout(body),
            key_bindings=self._create_key_bindings(),
            style=Style.from_dict(
                {
                    "title": "bold",
                    "selected": "reverse",
                    "active": "ansigreen",
                    "inactive": "ansired",
                    "muted": "ansibrightblack",
                    "focus": "ansicyan",
                }
            ),
            full_screen=False,
            mouse_support=False,
        )

    @property
    def current_provider(self):
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
        return models[self.state.model_index]

    def _selected_capabilities(self):
        if self._provider_requires_docs_only():
            return None
        selected = self._selected_model()
        if selected is None:
            return None
        try:
            return model_capabilities(selected.spec)
        except Exception:
            return None

    def _resolved_model(self) -> str | None:
        if self._provider_requires_docs_only():
            return None

        selected = self._selected_model()
        if selected is None:
            return None

        reasoning_value = None if self.state.reasoning_choice == KEEP_VALUE else self.state.reasoning_choice
        web_search_value = (
            None if self.state.web_search_choice == KEEP_VALUE else self.state.web_search_choice
        )
        context_value = None if self.state.context_choice == KEEP_VALUE else self.state.context_choice
        return apply_option_overrides(
            selected.spec,
            reasoning_value=reasoning_value,
            web_search_value=web_search_value,
            context_value=context_value,
        )

    def _reasoning_display(self, value: str, *, capabilities) -> str:
        if value == "auto" and capabilities.provider.config_name == "anthropic":
            return "adaptive"
        return value

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

    def _reset_overrides(self) -> None:
        self.state.reasoning_choice = KEEP_VALUE
        self.state.web_search_choice = KEEP_VALUE
        self.state.context_choice = KEEP_VALUE
        self.state.cache_ttl_choice = KEEP_VALUE

    def _move_provider(self, delta: int) -> None:
        count = len(self.snapshot.providers)
        self.state.provider_index = (self.state.provider_index + delta) % count
        self.state.model_index = 0
        self._reset_overrides()

    def _move_model(self, delta: int) -> None:
        models = self.current_models
        if not models:
            self.state.model_index = 0
            return
        self.state.model_index = (self.state.model_index + delta) % len(models)
        self._reset_overrides()

    def _toggle_source(self) -> None:
        self.state.source = "all" if self.state.source == "curated" else "curated"
        self.state.model_index = 0
        self._reset_overrides()

    def _row_style(self, *, selected: bool, available: bool) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        if available:
            parts.append("class:active")
        else:
            parts.append("class:inactive")
        return " ".join(parts)

    def _reasoning_choices(self) -> list[tuple[str, str]]:
        capabilities = self._selected_capabilities()
        if capabilities is None or not capabilities.reasoning_values:
            return []

        default_reasoning = self._reasoning_display(capabilities.default_reasoning, capabilities=capabilities)

        values = [
            (
                KEEP_VALUE,
                f"keep ({self._reasoning_display(capabilities.current_reasoning, capabilities=capabilities)})",
            ),
            (DEFAULT_VALUE, f"default ({default_reasoning})"),
        ]
        values.extend((value, self._reasoning_display(value, capabilities=capabilities)) for value in capabilities.reasoning_values)
        return values

    def _web_search_choices(self) -> list[tuple[str, str]]:
        capabilities = self._selected_capabilities()
        if capabilities is None or not capabilities.web_search_supported:
            return []

        current = web_search_display(capabilities.current_web_search)
        return [
            (KEEP_VALUE, f"keep ({current})"),
            (DEFAULT_VALUE, "default"),
            ("on", "on"),
            ("off", "off"),
        ]

    def _context_choices(self) -> list[tuple[str, str]]:
        capabilities = self._selected_capabilities()
        if capabilities is None or not capabilities.supports_long_context:
            return []

        current = "1m" if capabilities.current_long_context else "standard"
        return [
            (KEEP_VALUE, f"keep ({current})"),
            (DEFAULT_VALUE, "default (standard)"),
            ("1m", "1m"),
        ]

    def _cache_ttl_choices(self) -> list[tuple[str, str]]:
        capabilities = self._selected_capabilities()
        if capabilities is None:
            return []
        if capabilities.provider.config_name != "anthropic":
            return []
        if capabilities.cache_ttl_default is None:
            return []

        default_ttl = capabilities.cache_ttl_default
        return [
            (KEEP_VALUE, f"keep ({default_ttl})"),
            (DEFAULT_VALUE, f"default ({default_ttl})"),
            ("5m", "5m"),
            ("1h", "1h"),
        ]

    @staticmethod
    def _cycle_value(current: str, choices: list[tuple[str, str]]) -> str:
        if not choices:
            return current

        values = [value for value, _ in choices]
        if current not in values:
            return values[0]

        index = values.index(current)
        return values[(index + 1) % len(values)]

    def _render_provider_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        for index, option in enumerate(self.snapshot.providers):
            selected = index == self.state.provider_index
            cursor = "❯ " if self.state.focus == "providers" and selected else "  "
            line_style = self._row_style(selected=selected, available=option.active)
            availability = "available" if option.active else "not configured"
            text = (
                f"{cursor}{option.provider.display_name:<16} "
                f"[{availability}] ({len(option.curated_entries)} curated)\n"
            )
            fragments.append((line_style, text))
        return fragments

    def _render_model_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        models = self.current_models
        self._clamp_model_index()

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

    def _render_options_panel(self) -> StyleFragments:
        fragments: StyleFragments = []
        selected_model = self._selected_model()
        if selected_model is None:
            fragments.append(("class:muted", "No model selected.\n"))
            return fragments

        if self._provider_requires_docs_only():
            fragments.append(("class:muted", "See provider docs for model IDs/options.\n"))
            return fragments

        capabilities = self._selected_capabilities()

        reasoning_choices = self._reasoning_choices()
        web_search_choices = self._web_search_choices()
        context_choices = self._context_choices()
        cache_ttl_choices = self._cache_ttl_choices()
        has_known_options = False

        if reasoning_choices:
            reasoning_label = dict(reasoning_choices).get(self.state.reasoning_choice, "keep")
            fragments.append(("class:focus", f"Reasoning [r]: {reasoning_label}\n"))
            has_known_options = True

        if web_search_choices:
            web_search_label = dict(web_search_choices).get(self.state.web_search_choice, "keep")
            fragments.append(("class:focus", f"Web search [w]: {web_search_label}\n"))
            has_known_options = True

        if context_choices:
            context_label = dict(context_choices).get(self.state.context_choice, "keep")
            fragments.append(("class:focus", f"Context [x]: {context_label}\n"))
            if capabilities is not None and capabilities.long_context_window is not None:
                window = f"{capabilities.long_context_window:,}"
                fragments.append(("class:muted", f"  long-context target: {window} tokens\n"))
            has_known_options = True

        if cache_ttl_choices:
            cache_ttl_label = dict(cache_ttl_choices).get(self.state.cache_ttl_choice, "keep")
            fragments.append(("class:focus", f"Cache TTL [t]: {cache_ttl_label}\n"))
            has_known_options = True

        if not has_known_options:
            fragments.append(("class:muted", "No extra options for this model.\n"))

        resolved = self._resolved_model()
        if resolved is not None and has_known_options:
            fragments.append(("class:active", "Resolved model:\n"))
            fragments.append(("", f"{resolved}\n"))

            if self.state.cache_ttl_choice not in {KEEP_VALUE, DEFAULT_VALUE}:
                fragments.append(("class:active", f"anthropic.cache_ttl = {self.state.cache_ttl_choice}\n"))

        return fragments

    def _render_status_bar(self) -> StyleFragments:
        provider = self.current_provider
        scope = "curated" if self.state.source == "curated" else "all static"
        status = "available" if provider.active else "not configured"
        warning = ""
        if self._provider_requires_docs_only():
            warning = " · see docs"

        return [
            (
                "class:focus",
                (
                    f"Provider: {provider.provider.display_name} ({status}) | "
                    f"Scope: {scope} | Focus: {self.state.focus}{warning}\n"
                ),
            ),
            (
                "class:muted",
                (
                    "Keys: ←/→ focus · ↑/↓ move · Tab swap · "
                    "c scope · r/w/x/t options · d reset · Enter select · q quit"
                ),
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

        @kb.add("r")
        def _toggle_reasoning(event) -> None:
            choices = self._reasoning_choices()
            if choices:
                self.state.reasoning_choice = self._cycle_value(self.state.reasoning_choice, choices)
                event.app.invalidate()

        @kb.add("w")
        def _toggle_web_search(event) -> None:
            choices = self._web_search_choices()
            if choices:
                self.state.web_search_choice = self._cycle_value(self.state.web_search_choice, choices)
                event.app.invalidate()

        @kb.add("x")
        def _toggle_context(event) -> None:
            choices = self._context_choices()
            if choices:
                self.state.context_choice = self._cycle_value(self.state.context_choice, choices)
                event.app.invalidate()

        @kb.add("t")
        def _toggle_cache_ttl(event) -> None:
            choices = self._cache_ttl_choices()
            if choices:
                self.state.cache_ttl_choice = self._cycle_value(self.state.cache_ttl_choice, choices)
                event.app.invalidate()

        @kb.add("d")
        def _reset(event) -> None:
            self._reset_overrides()
            event.app.invalidate()

        @kb.add("enter")
        def _accept(event) -> None:
            selected_model = self._selected_model()
            if selected_model is None:
                return

            provider = self.current_provider
            if self._provider_requires_docs_only():
                event.app.exit(
                    result={
                        "provider": provider.provider.config_name,
                        "provider_available": provider.active,
                        "selected_model": None,
                        "resolved_model": None,
                        "source": self.state.source,
                        "reasoning_override": None,
                        "web_search_override": None,
                        "context_override": None,
                        "anthropic_cache_ttl_override": None,
                        "refer_to_docs": True,
                    }
                )
                return

            resolved = self._resolved_model()
            if resolved is None:
                resolved = selected_model.spec

            reasoning_override = None
            if self.state.reasoning_choice == DEFAULT_VALUE:
                reasoning_override = "default"
            elif self.state.reasoning_choice != KEEP_VALUE:
                reasoning_override = self.state.reasoning_choice

            web_search_override = None
            if self.state.web_search_choice == DEFAULT_VALUE:
                web_search_override = "default"
            elif self.state.web_search_choice != KEEP_VALUE:
                web_search_override = self.state.web_search_choice

            context_override = None
            if self.state.context_choice == DEFAULT_VALUE:
                context_override = "default"
            elif self.state.context_choice != KEEP_VALUE:
                context_override = self.state.context_choice

            cache_ttl_override = None
            if self.state.cache_ttl_choice == DEFAULT_VALUE:
                cache_ttl_override = "default"
            elif self.state.cache_ttl_choice != KEEP_VALUE:
                cache_ttl_override = self.state.cache_ttl_choice

            event.app.exit(
                result={
                    "provider": provider.provider.config_name,
                    "provider_available": provider.active,
                    "selected_model": selected_model.spec,
                    "resolved_model": resolved,
                    "source": self.state.source,
                    "reasoning_override": reasoning_override,
                    "web_search_override": web_search_override,
                    "context_override": context_override,
                    "anthropic_cache_ttl_override": cache_ttl_override,
                    "refer_to_docs": False,
                }
            )

        @kb.add("q")
        @kb.add("escape")
        @kb.add("c-c")
        def _quit(event) -> None:
            event.app.exit(result=None)

        return kb

    def run(self) -> dict[str, object] | None:
        return self.app.run()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt Toolkit prototype #4b: windowed split-lists picker with inline reasoning, "
            "web_search, context, and cache options"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to fastagent.config.yaml",
    )
    args = parser.parse_args()

    try:
        picker = SplitListPicker(config_path=args.config)
    except ValueError as exc:
        print(str(exc))
        return 1

    result = picker.run()
    if result is None:
        print("Cancelled.")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
