from typing import Any

import pytest

from fast_agent.llm.task_budget import (
    format_task_budget_tokens,
    parse_task_budget_tokens,
    validate_task_budget_tokens,
)


def test_parse_task_budget_tokens_accepts_shorthand_values() -> None:
    assert parse_task_budget_tokens("20k") == 20_000
    assert parse_task_budget_tokens("1M") == 1_000_000
    assert parse_task_budget_tokens(" 64000 ") == 64_000


def test_parse_task_budget_tokens_accepts_disabled_aliases() -> None:
    assert parse_task_budget_tokens(None) is None
    assert parse_task_budget_tokens("OFF") is None
    assert parse_task_budget_tokens("default") is None


def test_parse_task_budget_tokens_rejects_boolean_values() -> None:
    value: Any = True

    with pytest.raises(ValueError, match="integer token count"):
        parse_task_budget_tokens(value)


def test_parse_task_budget_tokens_rejects_non_string_objects() -> None:
    value: Any = {"tokens": "64k"}

    with pytest.raises(ValueError, match="integer token count"):
        parse_task_budget_tokens(value)


def test_validate_task_budget_tokens_rejects_boolean_values() -> None:
    value: Any = False

    with pytest.raises(ValueError, match="integer token count"):
        validate_task_budget_tokens(value)


def test_validate_task_budget_tokens_rejects_values_below_minimum() -> None:
    with pytest.raises(ValueError, match="20,000"):
        validate_task_budget_tokens(19_999)


def test_format_task_budget_tokens_uses_compact_suffixes() -> None:
    assert format_task_budget_tokens(None) == "default"
    assert format_task_budget_tokens(64_000) == "64k"
    assert format_task_budget_tokens(1_000_000) == "1m"


def test_format_task_budget_tokens_does_not_suffix_invalid_values() -> None:
    assert format_task_budget_tokens(0) == "0"
    assert format_task_budget_tokens(-1_000) == "-1000"
