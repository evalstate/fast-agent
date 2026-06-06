from __future__ import annotations

import pytest

from fast_agent.llm.reasoning_effort import (
    ReasoningEffortSetting,
    ReasoningEffortSpec,
    available_reasoning_values,
    parse_boolean_alias,
    parse_reasoning_setting,
    validate_reasoning_setting,
)


@pytest.mark.parametrize("value", ["on", "true", "1", "yes", "enable", "enabled", " ON "])
def test_parse_boolean_alias_accepts_true_aliases(value: str) -> None:
    assert parse_boolean_alias(value) is True


@pytest.mark.parametrize("value", ["off", "false", "0", "no", "disable", "disabled", " OFF "])
def test_parse_boolean_alias_accepts_false_aliases(value: str) -> None:
    assert parse_boolean_alias(value) is False


@pytest.mark.parametrize("value", ["", "auto", "default", "high"])
def test_parse_boolean_alias_returns_none_for_non_boolean_values(value: str) -> None:
    assert parse_boolean_alias(value) is None


def test_effort_spec_without_disable_rejects_off() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
    )
    setting = parse_reasoning_setting(" OFF ")
    assert setting == ReasoningEffortSetting(kind="toggle", value=False)
    assert setting is not None
    assert "off" not in available_reasoning_values(spec)

    with pytest.raises(ValueError, match="disable is not supported"):
        validate_reasoning_setting(setting, spec)


def test_effort_spec_with_none_allows_off() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["none", "low", "medium", "high"],
    )
    setting = ReasoningEffortSetting(kind="toggle", value=False)

    assert "off" in available_reasoning_values(spec)
    assert validate_reasoning_setting(setting, spec) == setting


def test_effort_spec_with_toggle_disable_allows_off() -> None:
    spec = ReasoningEffortSpec(
        kind="effort",
        allowed_efforts=["low", "medium", "high"],
        allow_toggle_disable=True,
    )
    setting = ReasoningEffortSetting(kind="toggle", value=False)

    assert "off" in available_reasoning_values(spec)
    assert validate_reasoning_setting(setting, spec) == setting
