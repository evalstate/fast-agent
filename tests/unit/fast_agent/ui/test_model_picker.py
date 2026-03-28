from __future__ import annotations

import types
from typing import Any, cast

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from fast_agent.llm.model_selection import CatalogModelEntry
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker import _find_initial_model_index, _SplitListPicker
from fast_agent.ui.model_picker_common import (
    ANTHROPIC_VERTEX_PROVIDER_KEY,
    GENERIC_CUSTOM_MODEL_SENTINEL,
    ModelOption,
    ModelPickerSnapshot,
    ProviderOption,
    build_snapshot,
    model_options_for_provider,
    provider_activation_action,
)


def test_prompt_toolkit_window_scrolls_to_keep_cursor_visible() -> None:
    visible_rows = 5
    selected_index = 8
    control = FormattedTextControl(
        text=[("", "".join(f"item {index}\n" for index in range(12)))],
        show_cursor=False,
        get_cursor_position=lambda: Point(x=0, y=selected_index),
    )
    window = Window(
        control,
        wrap_lines=False,
        height=Dimension.exact(visible_rows),
        dont_extend_height=True,
    )

    content = control.create_content(width=40, height=visible_rows)

    window._scroll_without_linewrapping(content, width=40, height=visible_rows)

    assert window.vertical_scroll == selected_index - visible_rows + 1


def test_picker_uses_cursor_position_for_model_scrolling() -> None:
    picker = _SplitListPicker(config_path=None)
    picker.state.source = "all"

    provider_index: int | None = None
    for index, option in enumerate(picker.snapshot.providers):
        if option.provider is None:
            continue
        if len(model_options_for_provider(picker.snapshot, option.provider, source="all")) > picker.LIST_VISIBLE_ROWS:
            provider_index = index
            break

    assert provider_index is not None

    picker.state.provider_index = provider_index
    picker.state.model_index = picker.LIST_VISIBLE_ROWS + 1

    cursor = picker._model_cursor_position()

    assert cursor is not None
    assert cursor.y == picker.state.model_index


def test_picker_uses_cursor_position_for_provider_scrolling() -> None:
    picker = _SplitListPicker(config_path=None)
    picker.snapshot = ModelPickerSnapshot(
        providers=tuple(
            ProviderOption(
                provider=provider,
                active=True,
                curated_entries=(
                    CatalogModelEntry(alias=f"{provider.config_name}-demo", model=f"{provider.config_name}.demo"),
                ),
            )
            for provider in Provider
        ),
        config_payload={},
    )
    picker.state.provider_index = len(picker.snapshot.providers) - 1

    cursor = picker._provider_cursor_position()

    assert cursor is not None
    assert cursor.y == picker.state.provider_index


def test_picker_uses_prompt_toolkit_layout_focus() -> None:
    picker = _SplitListPicker(config_path=None)

    assert picker.app.layout.has_focus(picker.provider_window)
    assert not picker.app.layout.has_focus(picker.model_window)
    assert "Focus: providers" in picker._render_status_bar()[0][1]

    picker._focus_models()

    assert picker.app.layout.has_focus(picker.model_window)
    assert not picker.app.layout.has_focus(picker.provider_window)
    assert "Focus: models" in picker._render_status_bar()[0][1]


def test_provider_display_name_uses_local_generic_label() -> None:
    assert _SplitListPicker._provider_display_name("generic", "Generic") == "Local (ollama)"


def test_provider_display_name_uses_overlays_label_for_overlay_group() -> None:
    option = ProviderOption(
        provider=None,
        active=True,
        curated_entries=(
            CatalogModelEntry(
                alias="qwen-local",
                model="openresponses.unsloth/Qwen3.5-9B-GGUF",
                local=True,
            ),
        ),
        key="overlays",
        display_name="Overlays",
        overlay_group=True,
    )

    assert _SplitListPicker._provider_display_name_for_option(option) == "Overlays"
    assert _SplitListPicker._provider_entry_count_label(option) == "1 overlay"


def test_overlay_group_without_entries_renders_empty_message() -> None:
    picker = _SplitListPicker(config_path=None)
    picker.snapshot = ModelPickerSnapshot(
        providers=(
            ProviderOption(
                provider=None,
                active=False,
                curated_entries=(),
                key="overlays",
                display_name="Overlays",
                overlay_group=True,
            ),
        ),
        config_payload={},
    )
    picker.state.provider_index = 0

    rendered = "".join(fragment for _, fragment in picker._render_model_panel())

    assert "No local overlays found." in rendered
    assert picker._provider_availability_label(picker.current_provider) == "none yet"


def test_codex_inactive_provider_uses_activation_option() -> None:
    snapshot = ModelPickerSnapshot(
        providers=(
            ProviderOption(
                provider=Provider.CODEX_RESPONSES,
                active=False,
                curated_entries=(
                    CatalogModelEntry(alias="codexplan", model="codexresponses.o4-mini"),
                ),
            ),
        ),
        config_payload={},
    )

    assert provider_activation_action(snapshot, Provider.CODEX_RESPONSES) == "codex-login"

    options = model_options_for_provider(
        snapshot,
        Provider.CODEX_RESPONSES,
        source="curated",
    )

    assert options == [
        ModelOption(
            spec="codexresponses.__login__",
            label="Log in to enable Codex (Plan)",
            activation_action="codex-login",
        )
    ]


def test_codex_inactive_provider_is_shown_as_sign_in_required() -> None:
    picker = _SplitListPicker(config_path=None, initial_provider="codexresponses")
    picker.snapshot = ModelPickerSnapshot(
        providers=(
            ProviderOption(
                provider=Provider.CODEX_RESPONSES,
                active=False,
                curated_entries=(
                    CatalogModelEntry(alias="codexplan", model="codexresponses.o4-mini"),
                ),
            ),
        ),
        config_payload={},
    )
    picker.state.provider_index = 0
    picker.state.model_index = 0

    provider = picker.current_provider
    assert picker._provider_availability_label(provider) == "sign in required"
    status_line = picker._render_status_bar()[0][1]
    assert "press Enter to log in" in status_line


def test_find_initial_model_index_matches_model_identity() -> None:
    options = [
        ModelOption(
            spec="responses.chatgpt-5.3-instant",
            label="chatgpt → responses.chatgpt-5.3-instant",
            preset_token="chatgpt",
        ),
        ModelOption(
            spec="responses.gpt-5.4",
            label="gpt-5.4 → responses.gpt-5.4",
            preset_token="gpt-5.4",
        ),
    ]

    assert _find_initial_model_index(options, "chatgpt") == 0


def test_find_initial_model_index_maps_generic_model_to_custom_entry() -> None:
    options = [
        ModelOption(
            spec=GENERIC_CUSTOM_MODEL_SENTINEL,
            label="Enter local model string (e.g. llama3.2)",
        )
    ]

    assert _find_initial_model_index(options, "generic.llama3.2") == 0


def test_tabulate_model_label_uses_compact_columns() -> None:
    formatted = _SplitListPicker._tabulate_model_label(
        "qwen-local             → openresponses.unsloth/Qwen3.5-9B-GGUF (local, fast) — Imported from llama.cpp",
        panel_width=58,
    )

    assert "qwen-local" in formatted
    assert "openresponses.unsloth" in formatted
    assert len(formatted) <= 58


def test_picker_returns_overlay_token_as_resolved_model() -> None:
    class _FakeApp:
        def __init__(self) -> None:
            self.result = None

        def exit(self, *, result) -> None:
            self.result = result

    class _FakeEvent:
        def __init__(self, app: _FakeApp) -> None:
            self.app = app

    picker = _SplitListPicker(config_path=None)
    picker.snapshot = ModelPickerSnapshot(
        providers=(
            ProviderOption(
                provider=None,
                active=True,
                curated_entries=(
                    CatalogModelEntry(
                        alias="haikutiny",
                        model="anthropic.claude-haiku-4-5?temperature=0.5",
                        local=True,
                    ),
                ),
                key="overlays",
                display_name="Overlays",
                overlay_group=True,
            ),
        ),
        config_payload={},
    )
    picker.state.provider_index = 0
    picker.state.model_index = 0

    enter_binding = next(
        binding
        for binding in picker._create_key_bindings().bindings
        if getattr(binding.handler, "__name__", "") == "_accept"
    )

    app = _FakeApp()
    cast("Any", enter_binding.handler)(_FakeEvent(app))

    assert app.result is not None
    assert app.result.selected_model == "haikutiny"
    assert app.result.resolved_model == "haikutiny"


def test_snapshot_adds_anthropic_vertex_group_when_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.vertex_config.detect_google_adc",
        lambda: types.SimpleNamespace(
            available=True,
            project_id="proj",
            credentials=object(),
        ),
    )

    snapshot = build_snapshot(
        config_payload={
            "anthropic": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )

    option = next(
        provider
        for provider in snapshot.providers
        if provider.option_key == ANTHROPIC_VERTEX_PROVIDER_KEY
    )

    assert option.active is True
    assert option.option_display_name == "Anthropic (Vertex)"
    assert all(entry.model.startswith("anthropic-vertex.") for entry in option.curated_entries)


def test_snapshot_disables_anthropic_vertex_group_when_adc_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.vertex_config.detect_google_adc",
        lambda: types.SimpleNamespace(
            available=False,
            project_id=None,
            error=RuntimeError("missing"),
            credentials=None,
        ),
    )

    snapshot = build_snapshot(
        config_payload={
            "anthropic": {
                "vertex_ai": {
                    "enabled": True,
                    "project_id": "proj",
                    "location": "global",
                }
            }
        }
    )

    option = next(
        provider
        for provider in snapshot.providers
        if provider.option_key == ANTHROPIC_VERTEX_PROVIDER_KEY
    )

    assert option.active is False
    assert option.disabled_reason == "Google ADC not found"


def test_snapshot_adds_anthropic_vertex_group_for_env_only_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.llm.provider.anthropic.vertex_config.detect_google_adc",
        lambda: types.SimpleNamespace(
            available=True,
            project_id="proj",
            credentials=object(),
        ),
    )
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "proj")

    snapshot = build_snapshot(config_payload={})

    option = next(
        provider
        for provider in snapshot.providers
        if provider.option_key == ANTHROPIC_VERTEX_PROVIDER_KEY
    )

    assert option.active is True
    assert all(entry.model.startswith("anthropic-vertex.") for entry in option.curated_entries)
