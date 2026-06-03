"""Tests for namespaced model reference resolution."""

import os

import pytest
from pydantic import ValidationError

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.direct_factory import get_model_factory
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import (
    get_context_cli_model_override,
    get_context_model_references,
    resolve_model_reference,
    resolve_model_spec,
)
from fast_agent.llm.internal.passthrough import PassthroughLLM


def _build_context() -> Context:
    return Context(
        config=Settings(
            default_model="$system.default",
            model_references={
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


def test_resolve_model_reference_passthrough() -> None:
    assert resolve_model_reference(
        "gpt-5-mini?reasoning=low",
        {"system": {"fast": "haiku"}},
    ) == "gpt-5-mini?reasoning=low"


def test_resolve_model_reference_happy_path() -> None:
    context = _build_context()
    assert context.config is not None
    assert resolve_model_reference("$system.fast", context.config.model_references) == "claude-haiku-4-5"


def test_resolve_model_reference_normalizes_reference_values() -> None:
    assert (
        resolve_model_reference("  $system.fast  ", {"system": {"fast": "  claude-haiku-4-5  "}})
        == "claude-haiku-4-5"
    )


def test_resolve_model_reference_recursive() -> None:
    context = _build_context()
    assert context.config is not None
    assert resolve_model_reference("$custom.indirect", context.config.model_references) == "claude-haiku-4-5"


def test_resolve_model_reference_unknown_namespace() -> None:
    with pytest.raises(ModelConfigError, match="Unknown namespace"):
        resolve_model_reference("$unknown.fast", {"system": {"fast": "haiku"}})


def test_resolve_model_reference_unknown_key() -> None:
    with pytest.raises(ModelConfigError, match="Unknown key"):
        resolve_model_reference("$system.unknown", {"system": {"fast": "haiku"}})


def test_resolve_model_reference_cycle_detected() -> None:
    with pytest.raises(ModelConfigError, match="cycle"):
        resolve_model_reference(
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


def test_resolve_model_reference_rejects_non_exact_token_format() -> None:
    with pytest.raises(ModelConfigError, match="exact tokens"):
        resolve_model_reference("$system.fast?reasoning=low", {"system": {"fast": "haiku"}})


def test_resolve_model_spec_resolves_default_alias_from_context() -> None:
    context = _build_context()
    resolved_model = resolve_model_spec(
        context,
        hardcoded_default="playback",
        env_var="FAST_AGENT_MODEL_TEST_UNSET",
    )
    assert resolved_model.model == "passthrough"
    assert resolved_model.source == "config file"


def test_resolve_model_spec_precedence_with_aliases() -> None:
    context = _build_context()
    resolved_model = resolve_model_spec(
        context,
        model="$system.fast",
        cli_model="gpt-5-mini?reasoning=low",
        hardcoded_default="playback",
    )
    assert resolved_model.model == "claude-haiku-4-5"
    assert resolved_model.source == "explicit model"


def test_resolve_model_spec_cli_overrides_explicit_system_default_alias() -> None:
    context = _build_context()
    resolved_model = resolve_model_spec(
        context,
        model="$system.default",
        cli_model="gpt-5-mini?reasoning=low",
        hardcoded_default="playback",
    )

    assert resolved_model.model == "gpt-5-mini?reasoning=low"
    assert resolved_model.source == "CLI --model"


def test_resolve_model_spec_ignores_blank_higher_precedence_candidates() -> None:
    context = _build_context()

    resolved_model = resolve_model_spec(
        context,
        model="   ",
        cli_model="   ",
        hardcoded_default="playback",
        env_var="FAST_AGENT_MODEL_TEST_UNSET",
    )

    assert resolved_model.model == "passthrough"
    assert resolved_model.source == "config file"


def test_get_context_cli_model_override_returns_normalized_string() -> None:
    context = Context(config=Settings())
    assert context.config is not None
    context.config.cli_model_override = "  passthrough  "  # type: ignore[attr-defined]

    assert get_context_cli_model_override(context) == "passthrough"

    context.config.cli_model_override = "   "  # type: ignore[attr-defined]
    assert get_context_cli_model_override(context) is None


def test_context_model_helpers_accept_missing_context() -> None:
    assert get_context_cli_model_override(None) is None
    assert get_context_model_references(None) is None

    context = Context(config=None)
    assert get_context_cli_model_override(context) is None
    assert get_context_model_references(context) is None


def test_context_model_helpers_tolerate_partial_config_without_optional_fields() -> None:
    class _PartialConfig:
        pass

    context = Context.model_construct(config=_PartialConfig())

    assert get_context_cli_model_override(context) is None
    assert get_context_model_references(context) is None


def test_get_context_model_references_returns_config_mapping() -> None:
    context = _build_context()
    assert context.config is not None

    assert get_context_model_references(context) is context.config.model_references


def test_get_model_factory_inherits_context_cli_override_for_system_default() -> None:
    context = Context(config=Settings(default_model="responses.gpt-5-mini"))
    assert context.config is not None
    context.config.cli_model_override = "passthrough"  # type: ignore[attr-defined]

    factory = get_model_factory(context, model="$system.default")
    llm = factory(agent=LlmAgent(AgentConfig(name="test")))

    assert isinstance(llm, PassthroughLLM)
    assert llm.model_name == "passthrough"


def test_resolve_model_spec_falls_back_when_explicit_alias_unresolved() -> None:
    context = _build_context()
    resolved_model = resolve_model_spec(
        context,
        model="$system.unknown",
        hardcoded_default="playback",
    )

    assert resolved_model.model == "passthrough"
    assert resolved_model.source == "config file"


def test_resolve_model_spec_falls_back_to_hardcoded_when_config_alias_unresolved() -> None:
    context = Context(
        config=Settings(
            default_model="$system.missing",
            model_references={
                "system": {
                    "fast": "claude-haiku-4-5",
                }
            },
        )
    )

    resolved_model = resolve_model_spec(
        context,
        hardcoded_default="playback",
        env_var="FAST_AGENT_MODEL_TEST_UNSET",
    )

    assert resolved_model.model == "playback"
    assert resolved_model.source == "hardcoded default"


def test_resolve_model_spec_env_alias() -> None:
    context = Context(
        config=Settings(
            default_model=None,
            model_references={
                "system": {
                    "plan": "codexplan",
                }
            },
        )
    )
    original = os.environ.get("FAST_AGENT_MODEL")
    try:
        os.environ["FAST_AGENT_MODEL"] = "$system.plan"
        resolved_model = resolve_model_spec(
            context,
            default_model=None,
            fallback_to_hardcoded=False,
        )
    finally:
        if original is None:
            os.environ.pop("FAST_AGENT_MODEL", None)
        else:
            os.environ["FAST_AGENT_MODEL"] = original

    assert resolved_model.model == "codexplan"
    assert resolved_model.source == "environment variable FAST_AGENT_MODEL"


def test_settings_rejects_invalid_model_alias_namespace() -> None:
    with pytest.raises(ValidationError, match="namespace"):
        Settings(model_references={"bad.namespace": {"default": "passthrough"}})


def test_settings_rejects_invalid_model_alias_key() -> None:
    with pytest.raises(ValidationError, match="keys"):
        Settings(model_references={"system": {"bad.key": "passthrough"}})


def test_settings_rejects_empty_model_alias_value() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        Settings(model_references={"system": {"default": "   "}})
