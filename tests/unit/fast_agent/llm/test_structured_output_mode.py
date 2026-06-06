from typing import Any

import pytest

from fast_agent.llm.structured_output_mode import parse_structured_output_mode


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("json", "json"),
        (" JSON ", "json"),
        ("tool_use", "tool_use"),
        ("tool-use", "tool_use"),
        ("TOOL-USE", "tool_use"),
    ],
)
def test_parse_structured_output_mode_accepts_aliases(value: str, expected: str) -> None:
    assert parse_structured_output_mode(value) == expected


@pytest.mark.parametrize("value", [None, "", "xml", "tool use"])
def test_parse_structured_output_mode_rejects_unknown_values(value: str | None) -> None:
    assert parse_structured_output_mode(value) is None


def test_parse_structured_output_mode_rejects_non_string_values() -> None:
    value: Any = 123

    assert parse_structured_output_mode(value) is None
