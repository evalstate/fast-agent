from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _model_picker_common import (
    ModelOption,
    ModelSource,
    build_snapshot,
    model_options_for_provider,
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
        )

        self.provider_control = FormattedTextControl(self._render_provider_panel)
        self.model_control = FormattedTextControl(self._render_model_panel)
        self.status_control = FormattedTextControl(self._render_status_bar)

        body = HSplit(
            [
                VSplit(
                    [
                        Frame(Window(self.provider_control, wrap_lines=False), title="Providers"),
                        Frame(Window(self.model_control, wrap_lines=False), title="Models"),
                    ],
                    padding=1,
                ),
                Window(height=1, char="─", style="class:muted"),
                Window(self.status_control, height=3),
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

    @property
    def current_models(self) -> list[ModelOption]:
        models = model_options_for_provider(
            self.snapshot,
            self.current_provider.provider,
            source=self.state.source,
        )
        return models

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

    def _row_style(self, *, selected: bool, available: bool) -> str:
        parts: list[str] = []
        if selected:
            parts.append("class:selected")
        if available:
            parts.append("class:active")
        else:
            parts.append("class:inactive")
        return " ".join(parts)

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
            fragments.append(("class:muted", "  No models for this provider/scope.\n"))
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
        provider_status = "available" if provider.active else "not configured"
        model_scope = "curated" if self.state.source == "curated" else "all catalog"

        focus_hint = "providers" if self.state.focus == "providers" else "models"
        warning = ""
        if not provider.active:
            warning = " · selected provider has no detected API credentials"

        fragments: StyleFragments = [
            (
                "class:focus",
                (
                    f"Provider: {provider.provider.display_name} ({provider_status}) "
                    f"| Scope: {model_scope} | Focus: {focus_hint}{warning}\n"
                ),
            ),
            (
                "class:muted",
                "Keys: ←/→ switch column · ↑/↓ move · Tab switch · c toggle scope · Enter select · q quit\n",
            ),
        ]
        return fragments

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
        def _toggle(event) -> None:
            self._toggle_source()
            event.app.invalidate()

        @kb.add("enter")
        def _accept(event) -> None:
            models = self.current_models
            if not models:
                return

            self._clamp_model_index()
            model = models[self.state.model_index]
            provider = self.current_provider
            event.app.exit(
                result={
                    "provider": provider.provider.config_name,
                    "provider_available": provider.active,
                    "model": model.spec,
                    "source": self.state.source,
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
            "Prompt Toolkit prototype #4: non-fullscreen split lists "
            "for providers/models availability"
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
