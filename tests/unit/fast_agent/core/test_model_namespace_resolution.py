"""Tests for namespaced model alias resolution."""

import os

import pytest
from pydantic import ValidationError

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import resolve_model_alias, resolve_model_spec


def _build_context() -> Context:
    return Context(
        config=Settings(
            default_model="$system.default",
            model_aliases={
                "system": {
                    "default": "passthrough",
                    "plan": "codexplan",
                    "fast": "claude-haiku-4-5",
                },
                "custom": {
                    "indirect": "$system.fast",
                },
            },
        )
    )


def test_resolve_model_alias_passthrough() -> None:
    assert resolve_model_alias("gpt-5-mini.low", {"system": {"fast": "haiku"}}) == "gpt-5-mini.low"


def test_resolve_model_alias_happy_path() -> None:
    context = _build_context()
    assert context.config is not None
    assert resolve_model_alias("$system.fast", context.config.model_aliases) == "claude-haiku-4-5"


def test_resolve_model_alias_recursive() -> None:
    context = _build_context()
    assert context.config is not None
    assert resolve_model_alias("$custom.indirect", context.config.model_aliases) == "claude-haiku-4-5"


def test_resolve_model_alias_unknown_namespace() -> None:
    with pytest.raises(ModelConfigError, match="Unknown namespace"):
        resolve_model_alias("$unknown.fast", {"system": {"fast": "haiku"}})


def test_resolve_model_alias_unknown_key() -> None:
    with pytest.raises(ModelConfigError, match="Unknown key"):
        resolve_model_alias("$system.unknown", {"system": {"fast": "haiku"}})


def test_resolve_model_alias_cycle_detected() -> None:
    with pytest.raises(ModelConfigError, match="cycle"):
        resolve_model_alias(
            "$system.fast",
            {
                "system": {
                    "fast": "$custom.plan",
                },
                "custom": {
                    "plan": "$system.fast",
                },
            },
        )


def test_resolve_model_alias_rejects_non_exact_token_format() -> None:
    with pytest.raises(ModelConfigError, match="exact tokens"):
        resolve_model_alias("$system.fast?reasoning=low", {"system": {"fast": "haiku"}})


def test_resolve_model_spec_resolves_default_alias_from_context() -> None:
    context = _build_context()
    model, source = resolve_model_spec(context, hardcoded_default="playback")
    assert model == "passthrough"
    assert source == "config file"


def test_resolve_model_spec_precedence_with_aliases() -> None:
    context = _build_context()
    model, source = resolve_model_spec(
        context,
        model="$system.fast",
        cli_model="gpt-5-mini.low",
        hardcoded_default="playback",
    )
    assert model == "claude-haiku-4-5"
    assert source == "explicit model"


def test_resolve_model_spec_env_alias() -> None:
    context = Context(
        config=Settings(
            default_model=None,
            model_aliases={
                "system": {
                    "plan": "codexplan",
                }
            },
        )
    )
    original = os.environ.get("FAST_AGENT_MODEL")
    try:
        os.environ["FAST_AGENT_MODEL"] = "$system.plan"
        model, source = resolve_model_spec(
            context,
            default_model=None,
            fallback_to_hardcoded=False,
        )
    finally:
        if original is None:
            os.environ.pop("FAST_AGENT_MODEL", None)
        else:
            os.environ["FAST_AGENT_MODEL"] = original

    assert model == "codexplan"
    assert source == "environment variable FAST_AGENT_MODEL"


def test_settings_rejects_invalid_model_alias_namespace() -> None:
    with pytest.raises(ValidationError, match="namespace"):
        Settings(model_aliases={"bad.namespace": {"default": "passthrough"}})


def test_settings_rejects_invalid_model_alias_key() -> None:
    with pytest.raises(ValidationError, match="keys"):
        Settings(model_aliases={"system": {"bad.key": "passthrough"}})


def test_settings_rejects_empty_model_alias_value() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        Settings(model_aliases={"system": {"default": "   "}})
