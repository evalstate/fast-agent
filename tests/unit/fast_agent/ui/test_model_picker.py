from __future__ import annotations

from fast_agent.llm.model_selection import CatalogModelEntry
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker import _SplitListPicker
from fast_agent.ui.model_picker_common import (
    ModelOption,
    ModelPickerSnapshot,
    ProviderOption,
    model_options_for_provider,
    provider_activation_action,
)


def test_models_window_vertical_scroll_tracks_picker_scroll_state() -> None:
    picker = _SplitListPicker(config_path=None)
    picker.state.source = "all"

    provider_index: int | None = None
    for index, _ in enumerate(picker.snapshot.providers):
        picker.state.provider_index = index
        picker.state.model_index = 0
        picker.state.model_scroll_top = 0
        picker._sync_model_scroll()
        if len(picker.current_models) > picker.LIST_VISIBLE_ROWS:
            provider_index = index
            break

    assert provider_index is not None

    picker.state.provider_index = provider_index
    picker.state.model_index = 0
    picker.state.model_scroll_top = 0
    picker._sync_model_scroll()

    assert picker.model_window.vertical_scroll == 0

    for _ in range(picker.LIST_VISIBLE_ROWS + 1):
        picker._move_model(1)

    assert picker.state.model_scroll_top > 0
    assert picker.model_window.vertical_scroll == picker.state.model_scroll_top

    cursor = picker._model_cursor_position()
    assert cursor is not None
    assert cursor.y == picker.state.model_index


def test_provider_display_name_uses_local_generic_label() -> None:
    assert _SplitListPicker._provider_display_name("generic", "Generic") == "Local (Generic)"


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
