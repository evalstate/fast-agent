from __future__ import annotations

from fast_agent.commands.handlers.shared import (
    load_prompt_messages_from_file,
    load_prompt_messages_result,
    unique_selection_options,
)


def test_unique_selection_options_preserves_first_nonempty_case_insensitive_spelling() -> None:
    options = [" Finder ", "", "finder", "FINDER", "other"]

    assert unique_selection_options(options) == ["Finder", "other"]


def test_unique_selection_options_uses_unicode_casefolding() -> None:
    assert unique_selection_options(["Straße", "STRASSE", "alpha"]) == ["Straße", "alpha"]


def test_unique_selection_options_accepts_generators() -> None:
    options = (option for option in [" Alpha ", "alpha", "beta"])

    assert unique_selection_options(options) == ["Alpha", "beta"]


def test_load_prompt_messages_from_file_prints_bracketed_filename_literally(capsys) -> None:
    result = load_prompt_messages_from_file("[draft].json", label="prompt")

    assert result is None
    assert "File not found: [draft].json" in capsys.readouterr().out


def test_load_prompt_messages_result_returns_error_without_printing(capsys) -> None:
    result = load_prompt_messages_result("[draft].json", label="prompt")

    assert result.messages is None
    assert result.error == "File not found: [draft].json"
    assert capsys.readouterr().out == ""
