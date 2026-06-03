from __future__ import annotations

import pytest

from fast_agent.commands.model_capabilities import (
    available_service_tier_values,
    resolve_model_info,
    resolve_resolved_model,
    resolve_service_tier,
    resolve_task_budget_tokens,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    service_tier_command_values,
)


class _MalformedCapabilityLLM:
    web_search_supported = "false"
    web_search_enabled = 1
    task_budget_tokens = True
    available_service_tiers = None
    service_tier = "turbo"


class _ValidCapabilityLLM:
    web_search_supported = True
    web_search_enabled = True
    task_budget_tokens = 64_000
    available_service_tiers = ("fast", "turbo", "flex")
    service_tier = "flex"


class _DuplicateServiceTierLLM:
    available_service_tiers = ("flex", "fast", "flex", "turbo", "fast")


class _SupportedMalformedServiceTierLLM:
    service_tier_supported = True
    available_service_tiers = "flex"


class _MissingCapabilityLLM:
    pass


class _BrokenCapabilityLLM:
    @property
    def web_search_supported(self) -> bool:
        raise AttributeError("provider bug")


class _MissingMetadataPropertyLLM:
    @property
    def resolved_model(self) -> object:
        raise AttributeError("resolved_model")

    @property
    def model_info(self) -> object:
        raise AttributeError("model_info")


def test_bool_capabilities_require_real_true_values() -> None:
    assert not resolve_web_search_supported(_MalformedCapabilityLLM())
    assert not resolve_web_search_enabled(_MalformedCapabilityLLM())

    assert resolve_web_search_supported(_ValidCapabilityLLM())
    assert resolve_web_search_enabled(_ValidCapabilityLLM())


def test_missing_capabilities_still_default_to_false() -> None:
    assert not resolve_web_search_supported(_MissingCapabilityLLM())


def test_capability_property_attribute_errors_are_not_masked() -> None:
    with pytest.raises(AttributeError, match="provider bug"):
        resolve_web_search_supported(_BrokenCapabilityLLM())


def test_optional_model_metadata_attribute_errors_are_not_masked() -> None:
    llm = _MissingMetadataPropertyLLM()

    with pytest.raises(AttributeError, match="resolved_model"):
        resolve_resolved_model(llm)
    with pytest.raises(AttributeError, match="model_info"):
        resolve_model_info(llm)


def test_task_budget_tokens_require_real_int_values() -> None:
    assert resolve_task_budget_tokens(_MalformedCapabilityLLM()) is None
    assert resolve_task_budget_tokens(_ValidCapabilityLLM()) == 64_000


def test_available_service_tier_values_ignore_malformed_values() -> None:
    assert available_service_tier_values(_MalformedCapabilityLLM()) == ()
    assert available_service_tier_values(_ValidCapabilityLLM()) == ("fast", "flex")


def test_available_service_tier_values_deduplicate_preserving_order() -> None:
    assert available_service_tier_values(_DuplicateServiceTierLLM()) == ("flex", "fast")


def test_available_service_tier_values_fall_back_when_supported() -> None:
    assert available_service_tier_values(_SupportedMalformedServiceTierLLM()) == (
        "fast",
        "flex",
    )
    assert service_tier_command_values(_SupportedMalformedServiceTierLLM()) == (
        "on",
        "off",
        "flex",
        "status",
    )


def test_service_tier_requires_supported_literal_value() -> None:
    assert resolve_service_tier(_MalformedCapabilityLLM()) is None
    assert resolve_service_tier(_ValidCapabilityLLM()) == "flex"
