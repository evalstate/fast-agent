from __future__ import annotations

from fast_agent.utils.text import (
    casefold_text,
    collapse_whitespace,
    format_english_list,
    starts_with_casefold,
    strip_casefold,
    strip_str_to_none,
    strip_to_none,
)


def test_collapse_whitespace_handles_missing_and_empty_text() -> None:
    assert collapse_whitespace(None) == ""
    assert collapse_whitespace("") == ""


def test_collapse_whitespace_collapses_all_whitespace_runs() -> None:
    assert collapse_whitespace(" one\t two \n\n three ") == "one two three"


def test_strip_to_none_preserves_non_empty_stripped_text() -> None:
    assert strip_to_none("  value  ") == "value"


def test_strip_to_none_omits_missing_and_blank_text() -> None:
    assert strip_to_none(None) is None
    assert strip_to_none(" \t\n ") is None


def test_strip_str_to_none_requires_string_values() -> None:
    assert strip_str_to_none("  value  ") == "value"
    assert strip_str_to_none(" \t\n ") is None
    assert strip_str_to_none(None) is None
    assert strip_str_to_none(123) is None


def test_strip_casefold_strips_and_casefolds_text() -> None:
    assert strip_casefold("  stra\u00dfe  ") == "strasse"


def test_casefold_text_preserves_spacing() -> None:
    assert casefold_text("  /HISTORY STRA\u00dfE  ") == "  /history strasse  "


def test_starts_with_casefold_handles_case_insensitive_prefixes() -> None:
    assert starts_with_casefold("Project", "pro")
    assert starts_with_casefold("stra\u00dfe", "STRASS")
    assert not starts_with_casefold("Project", "env")


def test_format_english_list_handles_empty_and_single_item_lists() -> None:
    assert format_english_list([]) == ""
    assert format_english_list(["--flag"]) == "--flag"


def test_format_english_list_uses_oxford_comma_for_multiple_items() -> None:
    assert format_english_list(["--one", "--two"]) == "--one, and --two"
    assert format_english_list(["--one", "--two", "--three"]) == "--one, --two, and --three"
