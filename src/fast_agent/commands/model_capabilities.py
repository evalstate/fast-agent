"""Runtime capability helpers for model command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.model_info import ModelInfo
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
    from fast_agent.llm.resolved_model import ResolvedModelSpec
    from fast_agent.llm.text_verbosity import TextVerbosityLevel, TextVerbositySpec


ServiceTierValue = Literal["fast", "flex"]
SERVICE_TIER_VALUES: tuple[ServiceTierValue, ...] = ("fast", "flex")


def resolve_web_search_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.web_search_enabled is True


def resolve_x_search_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.x_search_enabled is True


def resolve_web_fetch_enabled(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.web_fetch_enabled is True


def resolve_web_search_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.web_search_supported is True


def resolve_x_search_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.x_search_supported is True


def resolve_web_fetch_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.web_fetch_supported is True


def resolve_resolved_model(
    llm: "FastAgentLLMProtocol | None",
) -> "ResolvedModelSpec | None":
    return None if llm is None else llm.resolved_model


def resolve_model_name(llm: "FastAgentLLMProtocol | None") -> str | None:
    return None if llm is None else llm.model_name


def resolve_model_info(llm: "FastAgentLLMProtocol | None") -> "ModelInfo | None":
    return None if llm is None else llm.model_info


def resolve_reasoning_effort(
    llm: "FastAgentLLMProtocol | None",
) -> "ReasoningEffortSetting | None":
    return None if llm is None else llm.reasoning_effort


def resolve_reasoning_effort_spec(
    llm: "FastAgentLLMProtocol | None",
) -> "ReasoningEffortSpec | None":
    return None if llm is None else llm.reasoning_effort_spec


def set_reasoning_effort(
    llm: "FastAgentLLMProtocol",
    value: "ReasoningEffortSetting | None",
) -> None:
    llm.set_reasoning_effort(value)


def resolve_text_verbosity(
    llm: "FastAgentLLMProtocol | None",
) -> "TextVerbosityLevel | None":
    return None if llm is None else llm.text_verbosity


def resolve_text_verbosity_spec(
    llm: "FastAgentLLMProtocol | None",
) -> "TextVerbositySpec | None":
    return None if llm is None else llm.text_verbosity_spec


def set_text_verbosity(
    llm: "FastAgentLLMProtocol",
    value: "TextVerbosityLevel | None",
) -> None:
    llm.set_text_verbosity(value)


def set_web_search_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_web_search_enabled(value)


def set_x_search_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_x_search_enabled(value)


def set_web_fetch_enabled(llm: "FastAgentLLMProtocol", value: bool | None) -> None:
    llm.set_web_fetch_enabled(value)


def resolve_task_budget_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.task_budget_supported is True


def resolve_task_budget_tokens(llm: "FastAgentLLMProtocol | None") -> int | None:
    if llm is None or isinstance(llm.task_budget_tokens, bool):
        return None
    return llm.task_budget_tokens


def set_task_budget_tokens(llm: "FastAgentLLMProtocol", value: int | None) -> None:
    llm.set_task_budget_tokens(value)


def resolve_service_tier_supported(llm: "FastAgentLLMProtocol | None") -> bool:
    return llm is not None and llm.service_tier_supported is True


def available_service_tier_values(
    llm: "FastAgentLLMProtocol | None",
) -> tuple[ServiceTierValue, ...]:
    raw_values = () if llm is None else llm.available_service_tiers
    values = tuple(
        unique_preserve_order(value for value in raw_values if value in SERVICE_TIER_VALUES)
    )
    if values:
        return values
    if resolve_service_tier_supported(llm):
        return SERVICE_TIER_VALUES
    return ()


def service_tier_command_values(llm: "FastAgentLLMProtocol | None") -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def resolve_service_tier(llm: "FastAgentLLMProtocol | None") -> ServiceTierValue | None:
    value = None if llm is None else llm.service_tier
    return value if value in SERVICE_TIER_VALUES else None


def set_service_tier(
    llm: "FastAgentLLMProtocol",
    value: ServiceTierValue | None,
) -> None:
    llm.set_service_tier(value)


def describe_service_tier_state(llm: "FastAgentLLMProtocol | None") -> str:
    current_tier = resolve_service_tier(llm)
    return current_tier or "default"


def model_supports_web_search(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return resolve_web_search_supported(llm)


def model_supports_x_search(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports x_search runtime configuration."""
    return resolve_x_search_supported(llm)


def model_supports_web_fetch(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports web_fetch runtime configuration."""
    return resolve_web_fetch_supported(llm)


def model_supports_service_tier(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports service tier runtime configuration."""
    return resolve_service_tier_supported(llm)


def model_supports_task_budget(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model/provider supports task budget runtime configuration."""
    return resolve_task_budget_supported(llm)


def model_supports_text_verbosity(llm: "FastAgentLLMProtocol | None") -> bool:
    """Return True when model exposes text verbosity controls."""
    return resolve_text_verbosity_spec(llm) is not None
