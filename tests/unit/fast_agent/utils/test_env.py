from __future__ import annotations

import pytest

from fast_agent.utils.env import env_flag, is_truthy_env_value, optional_env_flag


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_is_truthy_env_value_accepts_truthy_values(value: str) -> None:
    assert is_truthy_env_value(value)


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "no"])
def test_is_truthy_env_value_rejects_missing_and_falsey_values(value: str | None) -> None:
    assert not is_truthy_env_value(value)


def test_env_flag_uses_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAST_AGENT_TEST_FLAG", raising=False)

    assert not env_flag("FAST_AGENT_TEST_FLAG")
    assert env_flag("FAST_AGENT_TEST_FLAG", default=True)


def test_optional_env_flag_distinguishes_missing_from_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FAST_AGENT_TEST_FLAG", raising=False)
    assert optional_env_flag("FAST_AGENT_TEST_FLAG") is None

    monkeypatch.setenv("FAST_AGENT_TEST_FLAG", "0")
    assert optional_env_flag("FAST_AGENT_TEST_FLAG") is False
