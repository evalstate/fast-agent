from __future__ import annotations

from fast_agent.utils.filename import sanitize_filename_component, sanitize_filename_suffix


def test_sanitize_filename_component_preserves_common_safe_characters() -> None:
    assert sanitize_filename_component("agent-1_run.trace", fallback="item") == (
        "agent-1_run.trace"
    )


def test_sanitize_filename_component_preserves_unicode_alphanumerics() -> None:
    assert sanitize_filename_component("café-東京", fallback="item") == "café-東京"


def test_sanitize_filename_component_replaces_unsafe_characters() -> None:
    assert sanitize_filename_component("My Agent/row:1", fallback="item") == (
        "My_Agent_row_1"
    )


def test_sanitize_filename_component_strips_outer_separators() -> None:
    assert sanitize_filename_component("  ../agent!!!  ", fallback="item") == "agent"
    assert sanitize_filename_component("__agent__", fallback="item") == "agent"


def test_sanitize_filename_component_uses_fallback_for_empty_result() -> None:
    assert sanitize_filename_component("", fallback="item") == "item"


def test_sanitize_filename_component_uses_fallback_for_only_separators() -> None:
    assert sanitize_filename_component("///", fallback="item") == "item"
    assert sanitize_filename_component("...", fallback="item") == "item"


def test_sanitize_filename_component_uses_fallback_for_windows_reserved_names() -> None:
    assert sanitize_filename_component("CON", fallback="item") == "item"
    assert sanitize_filename_component("nul.txt", fallback="item") == "item"
    assert sanitize_filename_component("COM1", fallback="item") == "item"


def test_sanitize_filename_suffix_normalizes_path_and_space_separators() -> None:
    assert sanitize_filename_suffix("openai/gpt-4o mini") == "openai_gpt-4o_mini"


def test_sanitize_filename_suffix_preserves_unicode_alphanumerics() -> None:
    assert sanitize_filename_suffix("café/東京 agent") == "café_東京_agent"


def test_sanitize_filename_suffix_collapses_and_strips_separators() -> None:
    assert sanitize_filename_suffix("  ../agent!!!  ") == "agent"


def test_sanitize_filename_suffix_uses_fallback_for_empty_result() -> None:
    assert sanitize_filename_suffix("__", fallback="item") == "item"
    assert sanitize_filename_suffix("!!!", fallback="item") == "item"


def test_sanitize_filename_suffix_uses_fallback_for_windows_reserved_names() -> None:
    assert sanitize_filename_suffix("aux", fallback="item") == "item"
    assert sanitize_filename_suffix("LPT9.log", fallback="item") == "item"
