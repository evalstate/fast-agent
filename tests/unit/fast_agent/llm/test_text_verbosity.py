from typing import Any

import pytest

from fast_agent.llm.text_verbosity import (
    TextVerbositySpec,
    available_text_verbosity_values,
    format_text_verbosity,
    parse_text_verbosity,
    validate_text_verbosity,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("low", "low"),
        (" MED ", "medium"),
        ("medium", "medium"),
        ("HIGH", "high"),
    ],
)
def test_parse_text_verbosity_accepts_levels_and_aliases(
    value: str,
    expected: str,
) -> None:
    assert parse_text_verbosity(value) == expected


@pytest.mark.parametrize("value", [None, "", "loud"])
def test_parse_text_verbosity_rejects_unknown_values(value: str | None) -> None:
    assert parse_text_verbosity(value) is None


def test_parse_text_verbosity_rejects_non_string_values() -> None:
    value: Any = 123

    assert parse_text_verbosity(value) is None


def test_validate_text_verbosity_rejects_values_outside_spec() -> None:
    with pytest.raises(ValueError, match="allowed: low"):
        validate_text_verbosity("high", TextVerbositySpec(allowed=("low",), default="low"))


def test_format_text_verbosity_and_available_values() -> None:
    spec = TextVerbositySpec(allowed=("low", "high"), default="low")

    assert format_text_verbosity(None) == "unset"
    assert format_text_verbosity("high") == "high"
    assert available_text_verbosity_values(spec) == ["low", "high"]
    assert available_text_verbosity_values(None) == []
